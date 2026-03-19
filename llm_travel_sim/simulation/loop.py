"""
Main simulation loop.
"""
from __future__ import annotations
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from llm_travel_sim.config import Config
from llm_travel_sim.context.attention import build_context, AttentionWeights
from llm_travel_sim.data.h3_grid import HexCell
from llm_travel_sim.llm.cache import LLMCache
from llm_travel_sim.llm.prompt_builder import build_cluster_prompt
from llm_travel_sim.llm.trip_generator import Trip, TripGenerator
from llm_travel_sim.simulation.state import SimulationState
from llm_travel_sim.simulation.transition import build_transition_matrix, apply_transition


def _process_hex(
    hex_id: str,
    cells: Dict[str, HexCell],
    demographic_features: Dict[str, np.ndarray],
    attractor_features: Dict[str, np.ndarray],
    cluster_memberships: Dict[str, np.ndarray],
    cluster_labels: List[str],
    importance_weights: np.ndarray,
    state: SimulationState,
    trip_gen: TripGenerator,
    config: Config,
    attention_weights: Optional[AttentionWeights],
    place_context: Optional[Dict[str, dict]] = None,
) -> List[Trip]:
    """Process a single hex for one timestep: build context, prompt, generate trips."""
    cell = cells[hex_id]
    pop_dict = state.population_dict()

    ctx = build_context(
        hex_id=hex_id,
        neighbors=cell.neighbors,
        demographic_features=demographic_features,
        attractor_features=attractor_features,
        population=pop_dict,
        timestep_wall_clock=state.wall_clock_hours,
        attention_weights=attention_weights,
        lat=cell.lat,
        lon=cell.lon,
        cbd_center=config.cbd_center,
        place_context=place_context,
    )

    memberships = cluster_memberships[hex_id].tolist()
    timestep_str = ctx["time"]
    valid_hex_ids = set(cells.keys())
    iw = importance_weights.tolist()

    all_trips: List[Trip] = []

    # One LLM call per cluster — skip clusters with negligible presence
    for label, membership in zip(cluster_labels, memberships):
        if membership < 0.05:
            continue

        prompt = build_cluster_prompt(
            context=ctx,
            cluster_label=label,
            cluster_membership=membership,
            n_stories=config.n_stories_per_cluster,
            neighbor_hex_ids=cell.neighbors,
        )

        trips = trip_gen.generate(
            prompt=prompt,
            hex_id=hex_id,
            timestep_str=timestep_str,
            valid_hex_ids=valid_hex_ids,
            importance_weights=iw,
            cluster_memberships=memberships,
            context=ctx,
        )
        all_trips.extend(trips)

    journals = [t.journal for t in all_trips if t.journal]
    state.log_journal(hex_id, timestep_str, journals)

    return all_trips


def run_simulation(
    cells: Dict[str, HexCell],
    demographic_features: Dict[str, np.ndarray],
    attractor_features: Dict[str, np.ndarray],
    cluster_memberships: Dict[str, np.ndarray],
    cluster_labels: List[str],
    importance_weights: np.ndarray,
    config: Config,
    hex_populations: Optional[Dict[str, float]] = None,
    attention_weights: Optional[AttentionWeights] = None,
    cache_mode: Optional[str] = "fingerprint",
    place_context: Optional[Dict[str, dict]] = None,
) -> SimulationState:
    """Run the full day simulation."""

    hex_ids = list(cells.keys())
    state = SimulationState(
        hex_ids=hex_ids,
        timestep_minutes=config.timestep_minutes,
        initial_population=hex_populations or {},
        base_population=config.base_population,
    )
    state.wall_clock_hours = float(config.start_hour)

    cache = LLMCache(config.cache_dir, mode=cache_mode) if cache_mode else None

    trip_gen = TripGenerator(
        model=config.llm_model,
        max_tokens=config.llm_max_tokens,
        temperature=config.llm_temperature,
        max_retries=config.llm_max_retries,
        cache=cache,
    )

    n_timesteps = int((config.end_hour - config.start_hour) * 60 / config.timestep_minutes)
    print(f"\n[loop] Starting simulation: {n_timesteps} timesteps, {len(hex_ids)} hexes")
    print(f"[loop] Time range: {config.start_hour}:00 to {config.end_hour}:00")
    print(f"[loop] LLM model: {config.llm_model}")
    print(f"[loop] Concurrency: {config.llm_concurrency} parallel hex calls\n")

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "progress.jsonl").write_text("")  # clear so this run starts fresh

    in_flight = []
    sim_start_wall = time.time()

    for ts_idx in range(n_timesteps):
        hour = state.wall_clock_hours
        hour_int = int(hour)
        minute = int((hour - hour_int) * 60)
        time_str = f"{hour_int:02d}:{minute:02d}"
        print(f"\n{'='*60}")
        print(f"[loop] Timestep {ts_idx+1}/{n_timesteps} | Time: {time_str} | "
              f"Total population: {state.population.sum().item():,.0f}")
        print(f"{'='*60}")

        state.log_population_snapshot()

        # Parallel LLM calls across hexes, respecting concurrency limit
        all_trips: List[Trip] = []

        with ThreadPoolExecutor(max_workers=config.llm_concurrency) as executor:
            futures = {
                executor.submit(
                    _process_hex,
                    hex_id, cells, demographic_features, attractor_features,
                    cluster_memberships, cluster_labels, importance_weights,
                    state, trip_gen, config, attention_weights, place_context,
                ): hex_id
                for hex_id in hex_ids
            }

            completed = 0
            for future in as_completed(futures):
                hex_id = futures[future]
                try:
                    trips = future.result()
                    all_trips.extend(trips)
                    completed += 1
                    if completed % 10 == 0:
                        print(f"[loop]   {completed}/{len(hex_ids)} hexes processed, "
                              f"{len(all_trips)} trips so far...")
                except Exception as e:
                    print(f"[loop]   ERROR processing hex {hex_id}: {e}")

        # Aggregate trips and apply transition
        state.log_trips(all_trips)

        # Build and apply transition matrix
        transition, new_in_flight = build_transition_matrix(
            trips=all_trips,
            hex_index=state._hex_index,
            n_hexes=len(hex_ids),
            population=state.population,
        )
        in_flight.extend(new_in_flight)

        new_pop, in_flight = apply_transition(
            population=state.population,
            transition=transition,
            in_flight_in=in_flight,
            current_timestep=ts_idx,
        )
        state.update_population(new_pop)

        # Mode split summary
        mode_counts = {}
        for t in all_trips:
            mode_counts[t.mode] = mode_counts.get(t.mode, 0) + 1
        mode_str = " | ".join(f"{m}: {c}" for m, c in sorted(mode_counts.items()))
        print(f"[loop] Trips this timestep: {len(all_trips)} | Mode split: {mode_str}")
        print(f"[loop] Token usage: {trip_gen.total_input_tokens:,} in / "
              f"{trip_gen.total_output_tokens:,} out")
        if cache:
            cache.flush()
            print(f"[loop] {cache.stats}")

        # Write intermediate output so it can be inspected while run continues
        out_dir = Path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        state.save_partial(config.output_dir)
        with (out_dir / "progress.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestep": ts_idx + 1,
                "time": time_str,
                "trips_this_step": len(all_trips),
                "cumulative_trips": len(state.trip_log),
                "mode_split": mode_counts,
            }) + "\n")

        state.advance_timestep()

    elapsed = time.time() - sim_start_wall
    print(f"\n[loop] Simulation complete in {elapsed:.1f}s")
    print(f"[loop] Total trips generated: {len(state.trip_log)}")
    print(f"[loop] Total tokens used: {trip_gen.total_input_tokens:,} input / "
          f"{trip_gen.total_output_tokens:,} output")

    return state
