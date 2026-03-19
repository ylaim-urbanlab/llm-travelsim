# llm-travel-sim

LLM-driven synthetic travel demand simulation over a Denver-area H3 hex grid.

## What it does

1. Build an H3 hex grid around Denver CBD.
2. Pull Census demographics and estimate per-hex population (or use synthetic fallbacks).
3. Pull OSM-derived context per hex (POI “attractors”, plus place context like neighborhoods, major roads, and proximity to trails/Cherry Creek).
4. Fit demographic clusters (GMM) and label them with human-readable personas.
5. Run a timestep loop from `start_hour` to `end_hour`, calling the LLM once per hex per (non-negligible) cluster, then converting trip stories into a population transition.

## Quickstart

### Install

```powershell
pip install -r requirements.txt
```

### Dry run (no LLM; verifies the pipeline)

```powershell
python main.py --dry-run --rings 1 --no-osm-api
```

### Small real run (LLM on; synthetic OSM context)

```powershell
python main.py --rings 1 --hexes 5 --clusters 2 --no-osm-api --cache-mode prompt
```

### Run with real OSM context

```powershell
python main.py --rings 3 --clusters 4
```

## CLI

```powershell
python main.py [flags]
```

Flags:

- `--no-osm-api` : skip live OSM; use synthetic attractors (and empty place context)
- `--dry-run` : skip LLM simulation (outputs saved, but no trips)
- `--hexes N` : limit to first `N` hexes for testing
- `--rings R` : set H3 ring radius around CBD
- `--timestep M` : timestep minutes (15 or 30)
- `--clusters K` : number of demographic clusters
- `--model STR` : LLM model string
- `--cache-mode {fingerprint,prompt,none}` : LLM response cache mode

Inspect and cache utilities:

- `python main.py inspect [hex_id]`
- `python main.py export-caches [path]`
  - exports `cache/*.json` to a directory or `.zip`
- `python main.py load-caches <path>`
  - loads cache JSON files from a directory or `.zip` into `cache/`

## Outputs and intermediate inspection

During an actual run (not `--dry-run`), `output/` is updated after each timestep:

- `output/trips.json` (all trips so far)
- `output/journals.json` (journal lines so far)
- `output/population.json` (population snapshots so far)
- `output/progress.jsonl` (one JSON line per timestep, append-only)

`progress.jsonl` is useful for live monitoring (e.g., “tail” it in another window).

## Caching

The code caches:

- H3 grid: `cache/grid_r{res}_k{rings}.json`
- Census: `cache/acs_bg_{year}.json`, `cache/hex_features_{year}.json`, `cache/hex_populations_{year}.json`, etc.
- OSM attractors and place context:
  - `cache/hex_attractors.json`
  - `cache/hex_place_context.json`
- Clusters:
  - `cache/clusters_k{n_clusters}_{hex_set_hash}.json`
- LLM responses:
  - `cache/llm_cache_fingerprint.json` or `cache/llm_cache_prompt.json`

### Note on cache-mode and variety

- `fingerprint` reuses responses across “similar” contexts (often reducing variation, which you might see as repeated journals within the same day period/bucket).
- `prompt` uses an exact prompt hash (more variety, more cost).
- `none` disables LLM caching (maximum variety, slowest/most expensive).

## Troubleshooting

### OSM connectivity / Overpass errors

OSM fetching is rate-limited and retried, but public Overpass servers can still be overloaded or unreachable.

- If place-context queries fail, the simulation can continue with partial/empty place context.
- Re-run later or ensure network connectivity to `overpass-api.de`.

### Cycling is zero (mode split sanity)

If cycling is missing:

1. Check the LLM cache mode (you may have cached responses generated with older prompt instructions).
2. Check that the prompt explicitly encourages cycling when plausible.
3. Check trip JSON validation/mode normalization.

## Development docs

- `docs/OSMNX_CONTEXT.md` explains how OSMnx is used to build the per-hex LLM context.
- `docs/CONTEXT_FOR_LLM.md` lists what the LLM prompt currently includes and what is missing.

