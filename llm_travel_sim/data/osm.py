"""
Pull OpenStreetMap attractor data and place context for Denver hexes.
Uses OSMnx (Overpass under the hood; no API key required).
Rate-limited and retry-friendly for public Overpass instances.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from llm_travel_sim.data.h3_grid import HexCell

# Radius (m) for POI/attractor counts around hex centroid. ~500m approximates resolution-8 hex.
ATTRACTOR_DIST_M = 500

# Radius (m) for place-context: neighbourhoods, major roads, trails
NEIGHBOURHOOD_RADIUS_M = 150
MAJOR_ROADS_RADIUS_M = 1000
TRAIL_PROXIMITY_M = 400  # walking distance to trail/greenway for active mobility

# Delay between API calls per hex (osmnx uses Overpass; be polite)
OSM_DELAY_SECONDS = 1.5

# Retries with backoff when server is overloaded
OSM_RETRY_ATTEMPTS = 4
OSM_RETRY_BACKOFF = 3

# Highway class order for sorting (most important first)
HIGHWAY_ORDER = ("motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary", "tertiary")

ATTRACTOR_NAMES = [
    "employment",   # offices, commercial, industrial
    "retail",       # shops, restaurants, services
    "education",    # schools, universities
    "recreation",   # parks, gyms
    "transit",      # bus stops, light rail
]

# OSMnx tag sets per attractor category (union of matches)
ATTRACTOR_TAGS = [
    {"office": True, "building": ["commercial", "office", "industrial"]},  # employment
    {"shop": True, "amenity": ["restaurant", "fast_food", "cafe", "bar", "pub"]},  # retail
    {"amenity": ["school", "university", "college", "kindergarten"]},  # education
    {"leisure": ["park", "fitness_centre", "sports_centre", "pitch"]},  # recreation
    {"public_transport": True, "highway": "bus_stop", "railway": ["station", "halt", "tram_stop"]},  # transit
]


def _run_with_retry(fn, *args, label: str = "OSM", **kwargs):
    """Run fn(*args, **kwargs) with retries and exponential backoff on overload/timeout."""
    last_err = None
    for attempt in range(OSM_RETRY_ATTEMPTS):
        try:
            time.sleep(OSM_DELAY_SECONDS)
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            retryable = (
                "load too high" in msg or "timeout" in msg or "rate" in msg
                or "busy" in msg or "try again" in msg or "429" in msg
                or "overpass" in msg
            )
            wait = OSM_DELAY_SECONDS * (OSM_RETRY_BACKOFF ** attempt)
            if attempt < OSM_RETRY_ATTEMPTS - 1 and retryable:
                print(f"[osm] {label}: {e} — retrying in {wait:.0f}s (attempt {attempt + 1}/{OSM_RETRY_ATTEMPTS})")
                time.sleep(wait)
            else:
                break
    if last_err:
        print(f"[osm] {label} failed: {last_err}")
    return None


def _get_ox():
    """Import osmnx; support both ox.features_from_point and ox.features.features_from_point."""
    import osmnx as ox
    return ox


def _count_features_at_point(lat: float, lon: float, tags: dict, dist: int = ATTRACTOR_DIST_M) -> int:
    """Return number of OSM features matching tags within dist (m) of (lat, lon)."""
    try:
        ox = _get_ox()
        fn = getattr(ox, "features_from_point", None) or getattr(ox.features, "features_from_point", None)
        if fn is None:
            return 0
        gdf = _run_with_retry(
            fn,
            (lat, lon),
            tags,
            dist=dist,
            label="features",
        )
        if gdf is None or gdf.empty:
            return 0
        return len(gdf)
    except ImportError:
        return 0
    except Exception as e:
        print(f"[osm] count_features: {e}")
        return 0


def _synthetic_attractors(hex_id: str, lat: float, lon: float) -> np.ndarray:
    """Plausible synthetic attractor counts based on CBD proximity."""
    rng = np.random.default_rng(abs(hash(hex_id + "osm")) % (2**32))
    cbd_lat, cbd_lon = 39.7392, -104.9903
    dist = ((lat - cbd_lat)**2 + (lon - cbd_lon)**2) ** 0.5
    urban_score = max(0, 1 - dist / 0.15)

    employment = int(rng.poisson(max(1, 20 * urban_score + 2)))
    retail = int(rng.poisson(max(1, 15 * urban_score + 3)))
    education = int(rng.poisson(max(1, 3 * (1 - urban_score) + 1)))
    recreation = int(rng.poisson(max(1, 4 * urban_score + 1)))
    transit = int(rng.poisson(max(1, 8 * urban_score + 1)))

    raw = np.array([employment, retail, education, recreation, transit], dtype=np.float32)
    norms = np.array([50.0, 40.0, 10.0, 15.0, 20.0])
    return np.clip(raw / norms, 0, 1)


def _get_neighbourhoods_osmnx(lat: float, lon: float) -> List[str]:
    """Return named neighbourhoods (place=neighbourhood) within radius of (lat, lon)."""
    try:
        ox = _get_ox()
        fn = getattr(ox, "features_from_point", None) or getattr(ox.features, "features_from_point", None)
        if fn is None:
            return []
        gdf = _run_with_retry(
            fn,
            (lat, lon),
            {"place": "neighbourhood"},
            dist=NEIGHBOURHOOD_RADIUS_M,
            label="neighbourhoods",
        )
        if gdf is None or gdf.empty:
            return []
        names = []
        for col in ("name", "name:en"):
            if col in gdf.columns:
                for v in gdf[col].dropna().unique():
                    if v and str(v).strip() and str(v) not in names:
                        names.append(str(v).strip())
        return names[:10]  # cap
    except ImportError:
        return []
    except Exception as e:
        print(f"[osm] neighbourhoods: {e}")
        return []


def _get_major_roads_osmnx(lat: float, lon: float) -> List[str]:
    """Return named major roads within radius; deduped by name, sorted by highway importance."""
    try:
        ox = _get_ox()
        fn = getattr(ox, "graph_from_point", None) or getattr(ox.graph, "graph_from_point", None)
        if fn is None:
            return []
        G = _run_with_retry(
            fn,
            (lat, lon),
            MAJOR_ROADS_RADIUS_M,
            network_type="drive",
            simplify=True,
            label="roads",
        )
        if G is None or G.number_of_edges() == 0:
            return []
        by_name: Dict[str, str] = {}
        order_idx = {t: i for i, t in enumerate(HIGHWAY_ORDER)}
        for _u, _v, data in G.edges(data=True):
            name = data.get("name")
            if isinstance(name, list):
                name = name[0] if name else None
            if not name or not str(name).strip():
                continue
            name = str(name).strip()
            hw = data.get("highway")
            if isinstance(hw, list):
                hw = hw[0] if hw else ""
            hw = str(hw) if hw else ""
            if hw not in order_idx:
                continue
            if name not in by_name or order_idx.get(hw, 99) < order_idx.get(by_name[name], 99):
                by_name[name] = hw
        names = sorted(by_name.keys(), key=lambda n: order_idx.get(by_name[n], 99))
        return names[:12]  # cap
    except ImportError:
        return []
    except Exception as e:
        print(f"[osm] major roads: {e}")
        return []


def _get_trail_proximity_osmnx(lat: float, lon: float) -> dict:
    """
    Return trail/greenway proximity for active mobility context.
    Keys: near_trail (bool), near_cherry_creek (bool), trail_names (list of str).
    """
    out = {"near_trail": False, "near_cherry_creek": False, "trail_names": []}
    try:
        ox = _get_ox()
        fn = getattr(ox, "features_from_point", None) or getattr(ox.features, "features_from_point", None)
        if fn is None:
            return out
        # Paths, footways, cycleways, pedestrian ways, and leisure tracks (e.g. running/cycling)
        gdf = _run_with_retry(
            fn,
            (lat, lon),
            {"highway": ["path", "footway", "cycleway", "pedestrian"], "leisure": "track"},
            dist=TRAIL_PROXIMITY_M,
            label="trails",
        )
        if gdf is None or gdf.empty:
            return out
        names = []
        cherry = False
        for col in ("name", "name:en"):
            if col not in gdf.columns:
                continue
            for v in gdf[col].dropna().unique():
                s = str(v).strip()
                if not s or s in names:
                    continue
                names.append(s)
                if "cherry creek" in s.lower():
                    cherry = True
        out["near_trail"] = len(names) > 0
        out["near_cherry_creek"] = cherry
        out["trail_names"] = names[:6]
        return out
    except ImportError:
        return out
    except Exception as e:
        print(f"[osm] trail proximity: {e}")
        return out


def build_hex_place_context(
    cells: Dict[str, HexCell],
    cache_dir: str,
    use_api: bool = True,
) -> Dict[str, dict]:
    """
    Return per-hex place context: neighbourhood names and major road names for LLM prompts.
    Cached to hex_place_context.json. When use_api=False, returns empty lists for all hexes.
    """
    cache_path = Path(cache_dir) / "hex_place_context.json"
    if cache_path.exists():
        raw = json.loads(cache_path.read_text())
        missing = set(cells.keys()) - set(raw.keys())
        if not missing:
            print(f"[osm] Loading place context from cache: {cache_path}")
            return raw
        print(f"[osm] Place context cache missing {len(missing)} hexes, regenerating...")

    result: Dict[str, dict] = {}

    if use_api:
        try:
            print("[osm] Fetching neighbourhood, roads, and trail proximity (OSMnx)...")
            for i, (hid, cell) in enumerate(cells.items()):
                neighborhoods = _get_neighbourhoods_osmnx(cell.lat, cell.lon)
                roads = _get_major_roads_osmnx(cell.lat, cell.lon)
                trail_info = _get_trail_proximity_osmnx(cell.lat, cell.lon)
                result[hid] = {
                    "neighborhoods": neighborhoods,
                    "roads": roads,
                    "near_trail": trail_info["near_trail"],
                    "near_cherry_creek": trail_info["near_cherry_creek"],
                    "trail_names": trail_info["trail_names"],
                }
                if (i + 1) % 10 == 0:
                    print(f"[osm] Place context: {i + 1}/{len(cells)} hexes...")
        except ImportError:
            print("[osm] osmnx not available, skipping place context")
            result = {
                hid: {"neighborhoods": [], "roads": [], "near_trail": False, "near_cherry_creek": False, "trail_names": []}
                for hid in cells
            }
    else:
        result = {
            hid: {"neighborhoods": [], "roads": [], "near_trail": False, "near_cherry_creek": False, "trail_names": []}
            for hid in cells
        }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, indent=2))
    print(f"[osm] Place context cached to {cache_path}")
    return result


def build_hex_attractors(
    cells: Dict[str, HexCell],
    cache_dir: str,
    use_api: bool = True,
) -> Dict[str, np.ndarray]:
    """Return per-hex attractor feature vectors (normalized 0-1) using OSMnx features_from_point."""
    cache_path = Path(cache_dir) / "hex_attractors.json"
    if cache_path.exists():
        raw = json.loads(cache_path.read_text())
        cached = {k: np.array(v, dtype=np.float32) for k, v in raw.items()}
        missing = set(cells.keys()) - set(cached.keys())
        if not missing:
            print(f"[osm] Loading attractors from cache: {cache_path}")
            return cached
        print(f"[osm] Cache missing {len(missing)} hexes, regenerating...")

    hex_attractors: Dict[str, np.ndarray] = {}

    if use_api:
        try:
            print("[osm] Pulling OSM attractor data via OSMnx...")
            success_count = 0
            for hid, cell in cells.items():
                counts = {name: 0 for name in ATTRACTOR_NAMES}
                for name, tags in zip(ATTRACTOR_NAMES, ATTRACTOR_TAGS):
                    n = _count_features_at_point(cell.lat, cell.lon, tags)
                    counts[name] = n
                if any(counts.values()):
                    raw = np.array([
                        counts["employment"],
                        counts["retail"],
                        counts["education"],
                        counts["recreation"],
                        counts["transit"],
                    ], dtype=np.float32)
                    norms = np.array([50.0, 40.0, 10.0, 15.0, 20.0])
                    hex_attractors[hid] = np.clip(raw / norms, 0, 1)
                    success_count += 1
                else:
                    hex_attractors[hid] = _synthetic_attractors(hid, cell.lat, cell.lon)

                if len(hex_attractors) % 20 == 0:
                    print(f"[osm] Processed {len(hex_attractors)}/{len(cells)} hexes ({success_count} from API)...")
        except ImportError:
            print("[osm] osmnx not available, using synthetic attractors")
            for hid, cell in cells.items():
                hex_attractors[hid] = _synthetic_attractors(hid, cell.lat, cell.lon)
    else:
        print("[osm] Building synthetic attractor data for all hexes...")
        for hid, cell in cells.items():
            hex_attractors[hid] = _synthetic_attractors(hid, cell.lat, cell.lon)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(
        {k: v.tolist() for k, v in hex_attractors.items()}, indent=2
    ))
    print(f"[osm] Built attractors for {len(hex_attractors)} hexes, cached to {cache_path}")
    return hex_attractors
