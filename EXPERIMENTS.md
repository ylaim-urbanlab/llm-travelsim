# Incremental experiment plan

Runs that add complexity step-by-step so you can validate the pipeline and control cost/latency.

---

## Pipeline summary

| Step | What it does | Caching / knobs |
|------|----------------|------------------|
| 1 | H3 grid (Denver CBD, res 8) | `grid_r8_k{rings}.json` |
| 2 | Census demographics + population | `acs_bg_*.json`, `bg_centroids_*.json`, `hex_features_*.json`, `hex_populations_*.json` |
| 3 | OSM attractors | `hex_attractors.json` (or synthetic if `--no-osm-api`) |
| 4 | GMM demographic clusters | `clusters_k{n}_{hex_set_hash}.json` |
| 5 | Simulation loop | LLM per (hex × timestep × cluster with membership > 0.05); `llm_cache_*.json` |

**Grid size (H3 ring radius):**  
`rings=1` → 7 hexes, `2` → 19, `3` → 37, `4` → 61, `5` → 91.

**LLM load per run:**  
`#timesteps × #hexes × (clusters with membership > 0.05)`.  
Default: 32 timesteps × 91 hexes × ~2–4 clusters ≈ 6k–12k calls (reduced by cache hits).

---

## Experiment 1: Smoke (no APIs, no LLM)

**Goal:** Confirm grid → census → OSM → clustering → state init → viz with zero external calls.

- Grid only (small), all data synthetic, simulation skipped.
- No `CENSUS_API_KEY` needed; use `--no-osm-api`.

```bash
python main.py --dry-run --rings 1 --no-osm-api
```

- **Optional:** Limit hexes for even faster run:  
  `python main.py --dry-run --rings 2 --hexes 5 --no-osm-api`

**Check:** `output/` has `summary.json`, `population_snapshots.json`, flow map HTML (flows will be empty). No trips, no LLM.

---

## Experiment 2: Tiny LLM run (synthetic data)

**Goal:** Full simulation loop + trip parsing + transition matrix with minimal hexes and no Census/Overpass.

- Same small grid, synthetic demographics and attractors, **real LLM**.
- Use cache so reruns are cheap.

```bash
python main.py --rings 1 --hexes 5 --no-osm-api --clusters 2
```

- Set `OPENAI_API_KEY` (or whatever your LLM client uses).
- If you have a Census key and want real demographics for these 5 hexes, drop `--no-osm-api` only when you’re ready to hit Overpass for that tiny area (or keep `--no-osm-api` and rely on Census-only for a “Census + synthetic OSM” variant).

**Check:** Non-zero trips in `output/`, mode split chart, flow map with arcs. Token usage printed in logs.

---

## Experiment 3: Small grid, real Census, synthetic OSM

**Goal:** Real demographics and populations, still no Overpass.

- 19 hexes (`rings 2`), Census API key set, OSM synthetic.

```bash
python main.py --rings 2 --no-osm-api --clusters 3
```

**Check:** Demographics and populations in cache (e.g. `hex_features_*.json`, `hex_populations_*.json`). Flow map reflects real-ish demographics; attractors are synthetic.

---

## Experiment 4: Medium grid, full data

**Goal:** Real Census + real OSM, more hexes.

- 37 hexes, full data, default clusters.

```bash
python main.py --rings 3 --clusters 4
```

- Requires `CENSUS_API_KEY` and Overpass (no `--no-osm-api`).

**Check:** `hex_attractors.json` from API; flow map and mode split look plausible.

---

## Experiment 5: Full default run

**Goal:** Production-sized run (91 hexes, default settings).

```bash
python main.py
```

- Equivalently: `--rings 5`, 30-min timestep, 6 clusters, real data.
- Use `--cache-mode fingerprint` (default) during development to avoid re-calling the LLM on same prompts.

**Check:** `summary.json` total trips and timesteps; flow map and journals for sanity.

---

## Experiment 6 (optional): Finer time resolution

**Goal:** Test 15-minute timesteps (2× timesteps, 2× LLM calls per hex).

```bash
python main.py --rings 2 --timestep 15 --clusters 3
```

- Best on a small grid first (e.g. `--rings 2`) to compare vs 30-min before scaling up.

---

## Quick reference

| Experiment | Rings | Hexes (cap) | Census | OSM | LLM | Clusters | Notes |
|------------|-------|-------------|--------|-----|-----|----------|-------|
| 1 Smoke | 1 | — | synthetic | synthetic | no (dry-run) | — | No APIs |
| 2 Tiny LLM | 1 | 5 | synthetic | synthetic | yes | 2 | Minimal LLM |
| 3 Small real | 2 | — | real | synthetic | yes | 3 | No Overpass |
| 4 Medium full | 3 | — | real | real | yes | 4 | Full data |
| 5 Full | 5 | — | real | real | yes | 6 | Default |
| 6 Fine time | 2 | — | real | optional | yes | 3 | 15-min steps |

**Cache:** Keep default `--cache-mode fingerprint` for repeat runs; use `--cache-mode none` only when you want to force fresh LLM responses.

**Inspect a hex:** After any run, `python main.py inspect [hex_id]` prints demographics, attractors, and cluster memberships for that hex (uses same grid/cache as last run; ensure cache matches your experiment’s grid).
