"""
SimulationState: holds all mutable state for the simulation.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import torch

from llm_travel_sim.llm.trip_generator import Trip


@dataclass
class SimulationState:
    hex_ids: List[str]
    timestep_minutes: int
    # Either a per-hex population dict (from Census) or a flat fallback int
    initial_population: Dict[str, float] = field(default_factory=dict)
    base_population: int = 500  # fallback if initial_population not provided

    # Current timestep
    timestep_idx: int = 0
    wall_clock_hours: float = 6.0  # starts at 6am

    # Population tensor: shape (n_hexes,), differentiable
    population: torch.Tensor = field(init=False)

    # In-flight population: {destination_hex_idx: (population_fraction, arrival_timestep)}
    in_flight: List[dict] = field(default_factory=list)

    # Logs
    trip_log: List[Trip] = field(default_factory=list)
    journal_log: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    population_log: List[List[float]] = field(default_factory=list)

    def __post_init__(self):
        self._hex_index = {hid: i for i, hid in enumerate(self.hex_ids)}
        if self.initial_population:
            pops = [float(self.initial_population.get(hid, self.base_population))
                    for hid in self.hex_ids]
        else:
            pops = [float(self.base_population)] * len(self.hex_ids)
        self.population = torch.tensor(pops, dtype=torch.float32)

    def hex_idx(self, hex_id: str) -> int:
        return self._hex_index.get(hex_id, -1)

    def advance_timestep(self):
        self.timestep_idx += 1
        self.wall_clock_hours += self.timestep_minutes / 60.0

    def update_population(self, new_population: torch.Tensor):
        self.population = new_population.detach()

    def log_trips(self, trips: List[Trip]):
        self.trip_log.extend(trips)

    def log_journal(self, hex_id: str, timestep_str: str, entries: List[str]):
        if hex_id not in self.journal_log:
            self.journal_log[hex_id] = {}
        self.journal_log[hex_id][timestep_str] = entries

    def log_population_snapshot(self):
        self.population_log.append(self.population.tolist())

    def population_dict(self) -> Dict[str, float]:
        return {hid: float(self.population[i]) for i, hid in enumerate(self.hex_ids)}

    def save_partial(self, output_dir: str) -> None:
        """Write current state to output_dir. Safe to call after each timestep for intermediate inspection."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        trips_data = [
            {
                "origin": t.origin,
                "destination": t.destination,
                "mode": t.mode,
                "purpose": t.purpose,
                "departure_time": t.departure_time,
                "journal": t.journal,
                "weight": t.weight,
                "context_time": t.hex_context_time,
            }
            for t in self.trip_log
        ]
        (out / "trips.json").write_text(json.dumps(trips_data, indent=2))

        (out / "journals.json").write_text(json.dumps(self.journal_log, indent=2))

        pop_data = {
            "hex_ids": self.hex_ids,
            "snapshots": self.population_log,
        }
        (out / "population.json").write_text(json.dumps(pop_data, indent=2))

    def save(self, output_dir: str):
        """Write final state to output_dir (same as save_partial; logs to console)."""
        self.save_partial(output_dir)
        print(f"[state] Saved {len(self.trip_log)} trips, {len(self.journal_log)} hex journals to {output_dir}")
