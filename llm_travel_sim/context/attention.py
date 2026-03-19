"""
Build context vectors for each hex at each timestep.
Attention weights are uniform now but structured for nn.Parameter swap-in.
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def _distance_and_direction(
    lat: float, lon: float, cbd_lat: float, cbd_lon: float
) -> Tuple[float, str]:
    """Rough distance (mi) and 8-point direction from CBD to (lat, lon)."""
    # Approx mi per degree at Denver's latitude (~39°): 1 deg lat ~ 69 mi, 1 deg lon ~ 53 mi
    dlat = (lat - cbd_lat) * 69.0
    dlon = (lon - cbd_lon) * 53.0
    dist_mi = (dlat**2 + dlon**2) ** 0.5
    # Bearing: 0 = N, 90 = E, etc.; convert to compass
    bearing_deg = math.degrees(math.atan2(dlon, dlat))
    if bearing_deg < 0:
        bearing_deg += 360
    idx = round(bearing_deg / 45) % 8
    direction = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"][idx]
    return dist_mi, direction


class AttentionWeights(nn.Module):
    """
    Placeholder attention weight module.
    Currently returns uniform weights, but stores a nn.Parameter
    so it can be trained in a future backprop loop.
    """
    def __init__(self, n_features: int = 1):
        super().__init__()
        # Scalar logit per neighbor position (uniform init -> softmax -> uniform weights)
        self.logit = nn.Parameter(torch.zeros(1))

    def forward(self, n_neighbors: int) -> torch.Tensor:
        """Return uniform attention weights of shape (n_neighbors,)."""
        return torch.ones(n_neighbors) / n_neighbors


_GLOBAL_ATTENTION = AttentionWeights()


def _time_encoding(hour: float) -> Dict[str, float]:
    """Sin/cos encoding of hour of day."""
    angle = 2 * math.pi * hour / 24.0
    return {"sin_hour": math.sin(angle), "cos_hour": math.cos(angle)}


def _day_period(hour: float) -> str:
    if 6.5 <= hour < 9.5:
        return "morning_peak"
    elif 9.5 <= hour < 15.5:
        return "midday"
    elif 15.5 <= hour < 19.0:
        return "evening_peak"
    else:
        return "night"


def build_context(
    hex_id: str,
    neighbors: List[str],
    demographic_features: Dict[str, np.ndarray],
    attractor_features: Dict[str, np.ndarray],
    population: Dict[str, float],
    timestep_wall_clock: float,  # fractional hours, e.g. 8.5 = 8:30am
    attention_weights: Optional[AttentionWeights] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    cbd_center: Optional[Tuple[float, float]] = None,
    place_context: Optional[Dict[str, dict]] = None,
) -> dict:
    """
    Build a structured context dict for one hex at one timestep.

    Returns a dict that can be serialized directly into a prompt.
    If lat, lon, cbd_center are provided, adds location_description for the prompt.
    If place_context is provided, adds neighborhoods and roads (from OSM) for this hex.
    """
    if attention_weights is None:
        attention_weights = _GLOBAL_ATTENTION

    n = len(neighbors)
    time_enc = _time_encoding(timestep_wall_clock)
    period = _day_period(timestep_wall_clock)

    if n == 0:
        neighbor_demo = np.zeros_like(demographic_features[hex_id])
        neighbor_attr = np.zeros_like(attractor_features[hex_id])
        neighbor_pop = 0.0
    else:
        weights = attention_weights(n).detach().numpy()  # (n,)

        neighbor_demo_stack = np.stack([
            demographic_features.get(nb, np.zeros_like(demographic_features[hex_id]))
            for nb in neighbors
        ])  # (n, d_demo)

        neighbor_attr_stack = np.stack([
            attractor_features.get(nb, np.zeros_like(attractor_features[hex_id]))
            for nb in neighbors
        ])  # (n, d_attr)

        neighbor_pop_vals = np.array([population.get(nb, 0.0) for nb in neighbors])

        neighbor_demo = (weights[:, None] * neighbor_demo_stack).sum(axis=0)
        neighbor_attr = (weights[:, None] * neighbor_attr_stack).sum(axis=0)
        neighbor_pop = float((weights * neighbor_pop_vals).sum())

    hour_int = int(timestep_wall_clock)
    minute = int((timestep_wall_clock - hour_int) * 60)
    time_str = f"{hour_int:02d}:{minute:02d}"

    out = {
        "hex_id": hex_id,
        "time": time_str,
        "hour": timestep_wall_clock,
        "period": period,
        "time_encoding": time_enc,
        "own_demographics": demographic_features[hex_id].tolist(),
        "own_attractors": attractor_features[hex_id].tolist(),
        "own_population": float(population.get(hex_id, 0.0)),
        "neighbor_demographics_avg": neighbor_demo.tolist(),
        "neighbor_attractors_avg": neighbor_attr.tolist(),
        "neighbor_population_avg": neighbor_pop,
        "n_neighbors": n,
    }

    if lat is not None and lon is not None and cbd_center is not None:
        dist_mi, direction = _distance_and_direction(lat, lon, cbd_center[0], cbd_center[1])
        out["lat"] = lat
        out["lon"] = lon
        out["location_description"] = f"~{dist_mi:.1f} mi {direction} of downtown Denver (lat {lat:.4f}, lon {lon:.4f})"

    if place_context:
        place_info = place_context.get(hex_id, {})
        out["neighborhoods"] = place_info.get("neighborhoods", [])
        out["roads"] = place_info.get("roads", [])
        out["near_trail"] = place_info.get("near_trail", False)
        out["near_cherry_creek"] = place_info.get("near_cherry_creek", False)
        out["trail_names"] = place_info.get("trail_names", [])

    return out
