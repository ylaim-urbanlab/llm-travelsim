"""
Pull ACS 5-year demographic estimates for Denver metro block groups.
Spatially join to H3 hexes via areal interpolation (centroid-in-polygon).
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests

from llm_travel_sim.data.h3_grid import HexCell

# ACS variable codes -> human-readable names
ACS_VARS = {
    "B19013_001E": "median_household_income",
    "B17001_002E": "pop_below_poverty",
    "B17001_001E": "pop_poverty_denom",
    "B22003_002E": "households_snap",
    "B22003_001E": "households_snap_denom",
    "B01001_003E": "pop_under_5",
    "B01001_027E": "pop_under_5_female",
    "B01002_001E": "median_age",
    "B09001_001E": "pop_under_18",
    "B01001_001E": "total_pop",
    "B11001_001E": "total_households",
    "B11001_002E": "family_households",
    "B25033_008E": "renter_occupied_units",
    "B25033_001E": "total_housing_units",
    "B25010_001E": "avg_household_size",
    "B08201_002E": "zero_vehicle_households",
    "B08201_001E": "vehicle_households_denom",
    "B08128_004E": "work_from_home",
    "B08128_001E": "work_from_home_denom",
    "B08303_001E": "mean_commute_time_sum",
    "B08303_013E": "commute_60_89min",
    "B24080_003E": "management_professional",
    "B24080_001E": "occupation_denom",
    "B24080_009E": "service_occupations",
    "B25058_001E": "median_contract_rent",
}

DENVER_STATE_FIPS = "08"
DENVER_COUNTY_FIPS = "031"

FEATURE_NAMES = [
    "median_hh_income_norm",
    "pct_below_poverty",
    "pct_snap",
    "median_age_norm",
    "pct_under_18",
    "pct_family_households",
    "pct_renter",
    "pct_zero_vehicle",
    "pct_work_from_home",
    "mean_commute_norm",
    "pct_mgmt_professional",
    "pct_service",
]


# ---------------------------------------------------------------------------
# ACS fetch
# ---------------------------------------------------------------------------

def _fetch_acs_block_groups(api_key: str, year: int, cache_dir: str) -> List[dict]:
    cache_path = Path(cache_dir) / f"acs_bg_{year}.json"
    if cache_path.exists():
        print(f"[census] Loading ACS data from cache: {cache_path}")
        return json.loads(cache_path.read_text())

    if not api_key:
        print("[census] WARNING: No Census API key set. Using synthetic fallback data.")
        return []

    var_list = ",".join(ACS_VARS.keys())
    url = (
        f"https://api.census.gov/data/{year}/acs/acs5"
        f"?get=NAME,{var_list}"
        f"&for=block+group:*"
        f"&in=state:{DENVER_STATE_FIPS}+county:{DENVER_COUNTY_FIPS}"
        f"&key={api_key}"
    )
    print(f"[census] Fetching ACS block groups from Census API...")
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            raw = resp.json()
            break
        except Exception as e:
            print(f"[census] Attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print("[census] Census API failed, using synthetic fallback.")
                return []

    headers = raw[0]
    records = [dict(zip(headers, row)) for row in raw[1:]]
    cache_path.write_text(json.dumps(records, indent=2))
    print(f"[census] Fetched {len(records)} block groups, cached to {cache_path}")
    return records


# ---------------------------------------------------------------------------
# TIGER centroid fetch
# ---------------------------------------------------------------------------

def _fetch_bg_centroids(year: int, cache_dir: str) -> Dict[str, Tuple[float, float]]:
    """
    Return {geoid: (lat, lon)} for all Denver County block groups.
    Uses TIGERweb Tracts_Blocks layer (no API key required).
    Fetches polygon geometry and computes centroid via Shapely.
    GEOID format: state(2) + county(3) + tract(6) + bg(1) = 12 chars.
    """
    cache_path = Path(cache_dir) / f"bg_centroids_{year}.json"
    if cache_path.exists():
        print(f"[census] Loading BG centroids from cache: {cache_path}")
        raw = json.loads(cache_path.read_text())
        return {k: tuple(v) for k, v in raw.items()}

    try:
        from shapely.geometry import shape as shapely_shape
    except ImportError:
        print("[census] shapely not available, centroids unavailable.")
        return {}

    base_url = (
        "https://tigerweb.geo.census.gov/arcgis/rest/services/"
        "TIGERweb/Tracts_Blocks/MapServer/11/query"
    )
    params = (
        "?where=STATE%3D%2708%27+AND+COUNTY%3D%27031%27"
        "&outFields=GEOID"
        "&returnGeometry=true"
        "&f=geojson"
    )

    print("[census] Fetching block group centroids from TIGERweb...")
    centroids: Dict[str, Tuple[float, float]] = {}
    offset = 0

    while True:
        url = base_url + params + f"&resultOffset={offset}&resultRecordCount=1000"
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                features = data.get("features", [])
                break
            except Exception as e:
                print(f"[census] TIGERweb attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    features = None

        if not features:
            break

        for feat in features:
            geoid = str(feat.get("properties", {}).get("GEOID", "")).zfill(12)
            try:
                geom = shapely_shape(feat["geometry"])
                c = geom.centroid
                centroids[geoid] = (c.y, c.x)  # (lat, lon)
            except Exception:
                continue

        # If fewer than 1000 returned we've got everything
        if len(features) < 1000:
            break
        offset += 1000

    if centroids:
        cache_path.write_text(json.dumps(centroids, indent=2))
        print(f"[census] Fetched {len(centroids)} BG centroids, cached to {cache_path}")
    else:
        print("[census] TIGERweb returned no centroids, falling back to nearest BG.")

    return centroids


def _geoid_from_record(r: dict) -> str:
    state  = str(r.get("state",  "")).zfill(2)
    county = str(r.get("county", "")).zfill(3)
    tract  = str(r.get("tract",  "")).zfill(6)
    bg     = str(r.get("block group", "")).zfill(1)
    return state + county + tract + bg


# ---------------------------------------------------------------------------
# Feature vector builder
# ---------------------------------------------------------------------------

def _build_feature_vector(record: dict) -> np.ndarray:
    def safe_div(a, b, default=0.0):
        try:
            av, bv = float(a), float(b)
            return av / bv if bv > 0 else default
        except (TypeError, ValueError):
            return default

    def safe_float(x, default=0.0):
        try:
            return float(x) if x not in (None, "", "null", "-666666666") else default
        except (TypeError, ValueError):
            return default

    features = np.array([
        safe_float(record.get("median_household_income"), 50000) / 200000,
        safe_div(record.get("pop_below_poverty", 0), record.get("pop_poverty_denom", 1)),
        safe_div(record.get("households_snap", 0), record.get("households_snap_denom", 1)),
        safe_float(record.get("median_age"), 35) / 80,
        safe_div(record.get("pop_under_18", 0), record.get("total_pop", 1)),
        safe_div(record.get("family_households", 0), record.get("total_households", 1)),
        safe_div(record.get("renter_occupied_units", 0), record.get("total_housing_units", 1)),
        safe_div(record.get("zero_vehicle_households", 0), record.get("vehicle_households_denom", 1)),
        safe_div(record.get("work_from_home", 0), record.get("work_from_home_denom", 1)),
        safe_float(record.get("mean_commute_time_sum"), 25) / 90,
        safe_div(record.get("management_professional", 0), record.get("occupation_denom", 1)),
        safe_div(record.get("service_occupations", 0), record.get("occupation_denom", 1)),
    ], dtype=np.float32)
    return np.clip(features, 0, 1)


def _safe_pop(record: dict) -> float:
    try:
        v = float(record.get("total_pop") or 0)
        return max(v, 0.0)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Synthetic fallbacks
# ---------------------------------------------------------------------------

def _synthetic_features(hex_id: str, lat: float, lon: float) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(hex_id)) % (2**32))
    cbd_lat, cbd_lon = 39.7392, -104.9903
    dist = ((lat - cbd_lat)**2 + (lon - cbd_lon)**2) ** 0.5
    urban_score = max(0, 1 - dist / 0.15)
    base = rng.uniform(0.2, 0.8, size=12).astype(np.float32)
    base[0] = float(np.clip(0.3 + urban_score * 0.4 + rng.uniform(-0.1, 0.1), 0, 1))
    base[6] = float(np.clip(0.3 + urban_score * 0.5 + rng.uniform(-0.1, 0.1), 0, 1))
    base[7] = float(np.clip(0.1 + urban_score * 0.3 + rng.uniform(-0.05, 0.05), 0, 1))
    return base


# ---------------------------------------------------------------------------
# Areal interpolation core
# ---------------------------------------------------------------------------

def _interpolate_hex(
    hex_id: str,
    cell: HexCell,
    bg_records: List[dict],         # renamed ACS records
    bg_geoids: List[str],           # geoid per record
    centroid_map: Dict[str, Tuple[float, float]],  # geoid -> (lat, lon)
) -> Tuple[np.ndarray, float]:
    """
    For one hex: find all BG centroids that fall inside the H3 boundary,
    blend their feature vectors and population counts by population weight.
    Falls back to nearest centroid if none are inside.
    Returns (feature_vector, population).
    """
    try:
        import h3
        from shapely.geometry import Point, Polygon

        boundary = h3.cell_to_boundary(hex_id)  # list of (lat, lon)
        hex_poly = Polygon([(lon, lat) for lat, lon in boundary])

        inside_indices = []
        for i, geoid in enumerate(bg_geoids):
            coords = centroid_map.get(geoid)
            if coords is None:
                continue
            lat, lon = coords
            if hex_poly.contains(Point(lon, lat)):
                inside_indices.append(i)

        if inside_indices:
            recs = [bg_records[i] for i in inside_indices]
            pops = np.array([_safe_pop(r) for r in recs], dtype=np.float32)
            weights = pops if pops.sum() > 0 else np.ones(len(recs), dtype=np.float32)
            weights = weights / weights.sum()
            features = np.stack([_build_feature_vector(r) for r in recs])
            blended = (weights[:, None] * features).sum(axis=0)
            population = float(pops.sum())
            return blended, population

        # Fallback: nearest centroid by Euclidean distance
        best_idx, best_dist = 0, float("inf")
        for i, geoid in enumerate(bg_geoids):
            coords = centroid_map.get(geoid)
            if coords is None:
                continue
            dist = (coords[0] - cell.lat)**2 + (coords[1] - cell.lon)**2
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        return _build_feature_vector(bg_records[best_idx]), _safe_pop(bg_records[best_idx])

    except Exception:
        # If shapely/h3 fail for any reason, use synthetic
        return _synthetic_features(hex_id, cell.lat, cell.lon), 500.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_hex_features(
    cells: Dict[str, HexCell],
    api_key: str,
    year: int,
    cache_dir: str,
) -> Dict[str, np.ndarray]:
    """Return per-hex demographic feature vectors via areal interpolation."""
    feature_cache = Path(cache_dir) / f"hex_features_{year}.json"
    if feature_cache.exists():
        raw = json.loads(feature_cache.read_text())
        cached = {k: np.array(v, dtype=np.float32) for k, v in raw.items()}
        missing = set(cells.keys()) - set(cached.keys())
        if not missing:
            print(f"[census] Loading hex features from cache: {feature_cache}")
            return cached
        print(f"[census] Cache missing {len(missing)} hexes, regenerating...")

    records = _fetch_acs_block_groups(api_key, year, cache_dir)

    if not records:
        print("[census] Building synthetic demographic features for all hexes...")
        result = {hid: _synthetic_features(hid, c.lat, c.lon) for hid, c in cells.items()}
        feature_cache.write_text(json.dumps({k: v.tolist() for k, v in result.items()}, indent=2))
        return result

    # Rename to human-readable and build geoid list
    renamed = []
    geoids = []
    for r in records:
        nr = {name: r.get(code) for code, name in ACS_VARS.items()}
        nr["state"] = r.get("state")
        nr["county"] = r.get("county")
        nr["tract"] = r.get("tract")
        nr["block group"] = r.get("block group")
        renamed.append(nr)
        geoids.append(_geoid_from_record(r))

    centroids = _fetch_bg_centroids(year, cache_dir)
    n_with_centroids = sum(1 for g in geoids if g in centroids)
    print(f"[census] Areal interpolation: {len(renamed)} BGs, "
          f"{n_with_centroids} with centroids")

    hex_features: Dict[str, np.ndarray] = {}
    for hid, cell in cells.items():
        feat, _ = _interpolate_hex(hid, cell, renamed, geoids, centroids)
        hex_features[hid] = feat

    feature_cache.write_text(json.dumps(
        {k: v.tolist() for k, v in hex_features.items()}, indent=2
    ))
    print(f"[census] Built features for {len(hex_features)} hexes via areal interpolation, "
          f"cached to {feature_cache}")
    return hex_features


def build_hex_populations(
    cells: Dict[str, HexCell],
    api_key: str,
    year: int,
    cache_dir: str,
) -> Dict[str, float]:
    """Return residential population per hex via areal interpolation."""
    pop_cache = Path(cache_dir) / f"hex_populations_{year}.json"
    if pop_cache.exists():
        raw = json.loads(pop_cache.read_text())
        cached = {k: float(v) for k, v in raw.items()}
        missing = set(cells.keys()) - set(cached.keys())
        if not missing:
            print(f"[census] Loading hex populations from cache: {pop_cache}")
            return cached
        print(f"[census] Population cache missing {len(missing)} hexes, regenerating...")

    records = _fetch_acs_block_groups(api_key, year, cache_dir)

    if not records:
        print("[census] No ACS data — using synthetic population (500/hex)")
        result = {hid: 500.0 for hid in cells}
        pop_cache.write_text(json.dumps(result, indent=2))
        return result

    renamed = []
    geoids = []
    for r in records:
        nr = {name: r.get(code) for code, name in ACS_VARS.items()}
        nr["state"] = r.get("state")
        nr["county"] = r.get("county")
        nr["tract"] = r.get("tract")
        nr["block group"] = r.get("block group")
        renamed.append(nr)
        geoids.append(_geoid_from_record(r))

    centroids = _fetch_bg_centroids(year, cache_dir)

    hex_pops: Dict[str, float] = {}
    for hid, cell in cells.items():
        _, pop = _interpolate_hex(hid, cell, renamed, geoids, centroids)
        hex_pops[hid] = pop if pop > 0 else 500.0

    pop_cache.write_text(json.dumps(hex_pops, indent=2))
    total = sum(hex_pops.values())
    print(f"[census] Built populations for {len(hex_pops)} hexes "
          f"(total: {total:,.0f}), cached to {pop_cache}")
    return hex_pops
