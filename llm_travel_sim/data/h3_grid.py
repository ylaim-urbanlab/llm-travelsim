"""
Generate the H3 hexagonal grid centered on Denver CBD.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import h3


@dataclass
class HexCell:
    hex_id: str
    lat: float
    lon: float
    neighbors: List[str] = field(default_factory=list)


def build_grid(
    center_lat: float,
    center_lon: float,
    resolution: int,
    ring_radius: int,
    cache_dir: str = "cache",
) -> Dict[str, HexCell]:
    """Return a dict mapping hex_id -> HexCell for all hexes within ring_radius of center."""
    cache_path = Path(cache_dir) / f"grid_r{resolution}_k{ring_radius}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"[h3_grid] Loading grid from cache: {cache_path}")
        raw = json.loads(cache_path.read_text())
        return {k: HexCell(**v) for k, v in raw.items()}

    print(f"[h3_grid] Building H3 grid (resolution={resolution}, rings={ring_radius})")
    center_hex = h3.latlng_to_cell(center_lat, center_lon, resolution)
    hex_ids = h3.grid_disk(center_hex, ring_radius)

    cells: Dict[str, HexCell] = {}
    for hid in hex_ids:
        lat, lon = h3.cell_to_latlng(hid)
        neighbors = [n for n in h3.grid_disk(hid, 1) if n != hid and n in hex_ids]
        cells[hid] = HexCell(hex_id=hid, lat=lat, lon=lon, neighbors=neighbors)

    cache_path.write_text(json.dumps({k: vars(v) for k, v in cells.items()}, indent=2))
    print(f"[h3_grid] Built {len(cells)} hexes, cached to {cache_path}")
    return cells
