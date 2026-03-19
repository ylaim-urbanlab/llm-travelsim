"""
State transition: convert trip flows into a population transition matrix.
Written in PyTorch for future differentiability.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch

from llm_travel_sim.llm.trip_generator import Trip


# Approximate travel time (in simulation timesteps of 30min) by mode
# These hold a fraction of the trip in-flight for delivery at next tick
MODE_TRAVEL_TIMESTEPS = {
    "walking": 1,
    "cycling": 1,
    "transit": 1,
    "driving": 1,
    "rideshare": 1,
}


def build_transition_matrix(
    trips: List[Trip],
    hex_index: Dict[str, int],
    n_hexes: int,
    population: torch.Tensor,
) -> Tuple[torch.Tensor, List[dict]]:
    """
    Construct a sparse transition matrix from trip flows.

    Returns:
    - transition: torch.Tensor shape (n_hexes, n_hexes)
      transition[i, j] = fraction of population moving from hex i to hex j
    - in_flight: list of {dest_idx, amount, arrival_offset_timesteps}
      for trips that take >1 timestep (currently all single-step, scaffold for multi-step)
    """
    # Accumulate weighted flows
    # flow_matrix[i, j] = total weighted trip count from i to j
    flow_matrix = torch.zeros(n_hexes, n_hexes, dtype=torch.float32)

    for trip in trips:
        orig_idx = hex_index.get(trip.origin, -1)
        dest_idx = hex_index.get(trip.destination, -1)
        if orig_idx < 0 or dest_idx < 0:
            continue
        flow_matrix[orig_idx, dest_idx] += trip.weight

    # Convert raw flows to fractions of origin population
    # Avoid dividing by zero: clip population to minimum 1
    pop_safe = torch.clamp(population, min=1.0)

    # fraction_matrix[i, j] = what fraction of hex i's population moves to j
    # Cap total outflow per hex at some reasonable fraction (e.g., 30% depart per 30-min window)
    max_departure_fraction = 0.30
    row_sums = flow_matrix.sum(dim=1, keepdim=True)  # total trips per origin
    # Normalize so total outflow <= max_departure_fraction * population
    scale = torch.where(
        row_sums > 0,
        torch.clamp(max_departure_fraction * pop_safe.unsqueeze(1) / row_sums, max=1.0),
        torch.ones_like(row_sums),
    )
    scaled_flows = flow_matrix * scale  # absolute people moving

    # Build transition as fraction matrix (for differentiability)
    transition = scaled_flows / pop_safe.unsqueeze(1)  # fraction of each origin's pop

    # All trips complete within 1 timestep for now (in_flight scaffold)
    in_flight: List[dict] = []
    # Future: for multi-step travel, append to in_flight instead of transition

    return transition, in_flight


def apply_transition(
    population: torch.Tensor,
    transition: torch.Tensor,
    in_flight_in: List[dict],
    current_timestep: int,
) -> Tuple[torch.Tensor, List[dict]]:
    """
    Apply transition matrix to population vector.

    population @ transition gives arrivals at each hex.
    (population is row vector, transition[i,j] = fraction from i to j)

    Returns updated population tensor and new in_flight list.
    """
    # People departing each hex
    departures = (population.unsqueeze(0) @ transition).squeeze(0)  # shape (n_hexes,)

    # Outflow from each origin
    outflow = transition.sum(dim=1) * population  # shape (n_hexes,)

    # New population = old - outflow + arrivals
    new_population = population - outflow + departures

    # Deliver in-flight passengers arriving this timestep
    in_flight_remaining = []
    for parcel in in_flight_in:
        if parcel["arrival_timestep"] <= current_timestep:
            new_population[parcel["dest_idx"]] += parcel["amount"]
        else:
            in_flight_remaining.append(parcel)

    # Clamp to non-negative (numerical safety)
    new_population = torch.clamp(new_population, min=0.0)

    return new_population, in_flight_remaining
