"""
llm_travel_sim - Main entry point.

Usage:
    python main.py
    python main.py --no-osm-api      # use synthetic OSM data
    python main.py --dry-run         # skip LLM, use placeholder trips
    python main.py --hexes 10        # limit to 10 hexes for testing
    python main.py inspect [hex_id]  # print profile for one hex
    python main.py export-caches [path]   # export cache/*.json to path (dir or .zip)
    python main.py load-caches <path>    # load caches from path (dir or .zip) into cache/

  During a run, output/ is updated after each timestep: trips.json, journals.json,
  population.json, and progress.jsonl (one line per timestep for tail -f).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from llm_travel_sim.cache_io import export_caches, load_caches
from llm_travel_sim.config import DEFAULT_CONFIG
from llm_travel_sim.data.h3_grid import build_grid
from llm_travel_sim.data.census import build_hex_features, build_hex_populations
from llm_travel_sim.data.osm import build_hex_attractors, build_hex_place_context
from llm_travel_sim.clustering.demographic import fit_clusters, cluster_cache_basename
from llm_travel_sim.simulation.loop import run_simulation
from llm_travel_sim.viz.flow_map import generate_flow_map, generate_mode_split_chart, print_summary_stats


def main():
    parser = argparse.ArgumentParser(description="LLM Travel Demand Simulator")
    parser.add_argument("--no-osm-api", action="store_true",
                        help="Skip Overpass API calls, use synthetic OSM data")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip LLM calls, generate placeholder trips")
    parser.add_argument("--hexes", type=int, default=None,
                        help="Limit number of hexes (for testing)")
    parser.add_argument("--rings", type=int, default=None,
                        help="Override ring radius")
    parser.add_argument("--timestep", type=int, default=None,
                        help="Override timestep minutes (15 or 30)")
    parser.add_argument("--clusters", type=int, default=None,
                        help="Override number of demographic clusters")
    parser.add_argument("--model", type=str, default=None,
                        help="Override LLM model string")
    parser.add_argument("--cache-mode", type=str, default="fingerprint",
                        choices=["fingerprint", "prompt", "none"],
                        help="LLM response cache mode (default: fingerprint)")
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    if args.rings:
        config.ring_radius = args.rings
    if args.timestep:
        config.timestep_minutes = args.timestep
    if args.clusters:
        config.n_clusters = args.clusters
    if args.model:
        config.llm_model = args.model

    # Create directories
    Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LLM TRAVEL DEMAND SIMULATOR")
    print("Denver Metropolitan Area")
    print("=" * 60)
    print(f"H3 resolution: {config.h3_resolution}")
    print(f"Ring radius: {config.ring_radius}")
    print(f"Timestep: {config.timestep_minutes} minutes")
    print(f"Clusters: {config.n_clusters}")
    print(f"LLM model: {config.llm_model}")
    print(f"Cache mode: {args.cache_mode}")
    print()

    # --- Step 1: Build H3 grid ---
    print("[main] Step 1/5: Building H3 grid...")
    cells = build_grid(
        center_lat=config.cbd_center[0],
        center_lon=config.cbd_center[1],
        resolution=config.h3_resolution,
        ring_radius=config.ring_radius,
        cache_dir=config.cache_dir,
    )
    if args.hexes:
        hex_ids = list(cells.keys())[:args.hexes]
        cells = {hid: cells[hid] for hid in hex_ids}
        print(f"[main] Limited to {len(cells)} hexes (--hexes flag)")
    print(f"[main] Grid: {len(cells)} hexes\n")

    # --- Step 2: Census demographic features + population ---
    print("[main] Step 2/5: Loading demographic features...")
    demographic_features = build_hex_features(
        cells=cells,
        api_key=config.census_api_key,
        year=config.census_year,
        cache_dir=config.cache_dir,
    )
    demographic_features = {hid: demographic_features[hid]
                             for hid in cells if hid in demographic_features}

    hex_populations = build_hex_populations(
        cells=cells,
        api_key=config.census_api_key,
        year=config.census_year,
        cache_dir=config.cache_dir,
    )
    hex_populations = {hid: hex_populations[hid] for hid in cells if hid in hex_populations}
    total_pop = sum(hex_populations.values())
    print(f"[main] Demographics: {len(demographic_features)} hex feature vectors")
    print(f"[main] Population:    {len(hex_populations)} hexes, {total_pop:,.0f} total residents\n")

    # --- Step 3: OSM attractor features ---
    print("[main] Step 3/5: Loading OSM attractor features...")
    attractor_features = build_hex_attractors(
        cells=cells,
        cache_dir=config.cache_dir,
        use_api=not args.no_osm_api,
    )
    attractor_features = {hid: attractor_features[hid]
                          for hid in cells if hid in attractor_features}
    print(f"[main] Attractors: {len(attractor_features)} hex feature vectors\n")

    # --- Step 3b: OSM place context (neighbourhoods, major roads) for prompts ---
    place_context = build_hex_place_context(
        cells=cells,
        cache_dir=config.cache_dir,
        use_api=not args.no_osm_api,
    )

    # --- Step 4: Fit demographic clusters ---
    print("[main] Step 4/5: Fitting demographic clusters...")
    cluster_memberships, cluster_labels, importance_weights = fit_clusters(
        hex_features=demographic_features,
        n_clusters=config.n_clusters,
        random_seed=config.random_seed,
        cache_dir=config.cache_dir,
    )
    print(f"[main] Clusters: {cluster_labels}\n")

    # --- Step 5: Run simulation ---
    if args.dry_run:
        print("[main] DRY RUN: skipping LLM simulation\n")
        from llm_travel_sim.simulation.state import SimulationState
        state = SimulationState(
            hex_ids=list(cells.keys()),
            timestep_minutes=config.timestep_minutes,
            initial_population=hex_populations,
            base_population=config.base_population,
        )
    else:
        print("[main] Step 5/5: Running simulation loop...")
        cache_mode = None if args.cache_mode == "none" else args.cache_mode
        state = run_simulation(
            cells=cells,
            demographic_features=demographic_features,
            attractor_features=attractor_features,
            cluster_memberships=cluster_memberships,
            cluster_labels=cluster_labels,
            importance_weights=importance_weights,
            config=config,
            hex_populations=hex_populations,
            cache_mode=cache_mode,
            place_context=place_context,
        )

    # --- Output ---
    print("\n[main] Saving outputs...")
    state.save(config.output_dir)

    # Visualization
    generate_flow_map(
        cells=cells,
        state=state,
        output_dir=config.output_dir,
        demographic_features=demographic_features,
        attractor_features=attractor_features,
        cluster_memberships=cluster_memberships,
        cluster_labels=cluster_labels,
        hex_populations=hex_populations,
    )
    generate_mode_split_chart(state=state, output_dir=config.output_dir)
    print_summary_stats(state)

    # Summary JSON
    summary = {
        "config": {
            "h3_resolution": config.h3_resolution,
            "ring_radius": config.ring_radius,
            "n_hexes": len(cells),
            "timestep_minutes": config.timestep_minutes,
            "n_clusters": config.n_clusters,
            "llm_model": config.llm_model,
        },
        "results": {
            "total_trips": len(state.trip_log),
            "n_timesteps": len(state.population_log),
        },
    }
    (Path(config.output_dir) / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[main] All outputs saved to ./{config.output_dir}/")
    print("[main] Done.")


def inspect_hex(hex_id: str = None):
    """Print a full profile for one hex cell. Defaults to the first hex."""
    config = DEFAULT_CONFIG
    FEATURE_NAMES = [
        "median_hh_income_norm", "pct_below_poverty", "pct_snap", "median_age_norm",
        "pct_under_18", "pct_family_households", "pct_renter", "pct_zero_vehicle",
        "pct_work_from_home", "mean_commute_norm", "pct_mgmt_professional", "pct_service",
    ]
    ATTRACTOR_NAMES = ["employment", "retail", "education", "recreation", "transit"]

    cache = Path(config.cache_dir)
    grid = json.loads((cache / "grid_r8_k5.json").read_text()) if (cache / "grid_r8_k5.json").exists() else \
           json.loads(next(cache.glob("grid_*.json")).read_text())

    if hex_id is None:
        hex_id = list(grid.keys())[0]
    if hex_id not in grid:
        # allow prefix match
        matches = [k for k in grid if k.startswith(hex_id)]
        if not matches:
            print(f"Hex '{hex_id}' not found. Available (first 5): {list(grid.keys())[:5]}")
            return
        hex_id = matches[0]

    def load(name):
        p = cache / name
        return json.loads(p.read_text()) if p.exists() else {}

    features  = load(f"hex_features_{config.census_year}.json")
    attractors = load("hex_attractors.json")
    pops      = load(f"hex_populations_{config.census_year}.json")
    cluster_basename = cluster_cache_basename(list(grid.keys()), config.n_clusters)
    clusters  = load(cluster_basename)

    cell = grid[hex_id]
    print(f"\n=== Hex: {hex_id} ===")
    print(f"Location:   {cell['lat']:.4f}, {cell['lon']:.4f}")
    print(f"Neighbors:  {len(cell['neighbors'])} hexes")
    print(f"Population: {pops.get(hex_id, 0):,.0f} residents")

    if hex_id in features:
        print("\nDemographics:")
        for name, val in zip(FEATURE_NAMES, features[hex_id]):
            bar = "#" * int(val * 20)
            print(f"  {name:<28} {val:.2f}  {bar}")

    if hex_id in attractors:
        print("\nAttractors:")
        for name, val in zip(ATTRACTOR_NAMES, attractors[hex_id]):
            bar = "#" * int(val * 20)
            print(f"  {name:<28} {val:.2f}  {bar}")

    if clusters and hex_id in clusters.get("memberships", {}):
        print("\nCluster memberships:")
        for label, m, w in zip(clusters["labels"],
                                clusters["memberships"][hex_id],
                                clusters["weights"]):
            bar = "#" * int(m * 20)
            print(f"  [{m:.2f}] {label:<45} weight={w:.2f}  {bar}")

    journals_path = Path(config.output_dir) / "journals.json"
    if journals_path.exists():
        journals = json.loads(journals_path.read_text())
        if hex_id in journals:
            print("\nSample journal entries:")
            for ts, entries in list(journals[hex_id].items())[:3]:
                for e in entries[:2]:
                    print(f"  [{ts}] {e}")
    print()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "inspect":
        hex_id = sys.argv[2] if len(sys.argv) > 2 else None
        inspect_hex(hex_id)
    elif len(sys.argv) > 1 and sys.argv[1] == "export-caches":
        cache_dir = DEFAULT_CONFIG.cache_dir
        output = sys.argv[2] if len(sys.argv) > 2 else "cache_export"
        export_caches(cache_dir, output)
    elif len(sys.argv) > 1 and sys.argv[1] == "load-caches":
        if len(sys.argv) < 3:
            print("Usage: python main.py load-caches <path>")
            print("  path: directory containing .json files, or a .zip from export-caches")
            sys.exit(1)
        cache_dir = DEFAULT_CONFIG.cache_dir
        load_caches(sys.argv[2], cache_dir)
    else:
        main()
