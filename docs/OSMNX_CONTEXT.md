# How we pull in context using OSMnx

## Pipeline order (we’re not “in grid building” when OSM runs)

1. **Step 1 — H3 grid**  
   We build the **grid first**: a set of hex cells (e.g. 37 for rings=3) with `hex_id`, centroid `(lat, lon)`, and neighbors. That’s all geometry; no OSM yet.

2. **Step 2 — Census**  
   For each of those hexes we get demographics and population (Census API or synthetic). Still no OSM.

3. **Step 3 — OSM**  
   **Only after** we have the grid do we pull OSM context. We use **each hex’s centroid (lat, lon)** as the center of OSMnx queries. So: grid defines *where* we care about; OSM fills in *what’s there* at those locations.

4. **Step 4 — Clustering**  
   Demographics are clustered (GMM) for persona labels.

5. **Step 5 — Simulation**  
   Each timestep, for each hex, we build a prompt that includes that hex’s OSM-derived context and call the LLM.

So we’re not “still in the grid building stage” when OSM runs — we’re past it. The grid is the list of places we care about; OSM is “for each of those places, what’s nearby?”

---

## How OSMnx is used (point + radius)

We never query “the whole city.” We query **per hex**, using that hex’s **centroid (lat, lon)** and a **radius in meters**. OSMnx (via Overpass) returns features or a graph **within that radius**.

Conceptually:

```
for each hex in grid:
    (lat, lon) = hex.centroid
    # Then we run several OSMnx calls, all centered on (lat, lon):
    attractors = count POIs in (lat, lon, radius=500m) by category
    neighbourhoods = place=neighbourhood in (lat, lon, 150m)
    roads = drive network in (lat, lon, 1000m), get named roads
    trails = path/footway/cycleway in (lat, lon, 400m), check Cherry Creek
```

All of this is in `llm_travel_sim/data/osm.py`.

---

## What we call and what we get

| What we want | OSMnx call | Radius | Output we use |
|--------------|------------|--------|----------------|
| **Attractors** (POI counts) | `features_from_point((lat, lon), tags, dist)` × 5 (one per category) | 500 m | Count of features per category → normalized vector [employment, retail, education, recreation, transit] |
| **Neighbourhoods** | `features_from_point((lat, lon), {"place": "neighbourhood"}, dist)` | 150 m | List of `name` values (e.g. "LoDo", "Cap Hill") |
| **Major roads** | `graph_from_point((lat, lon), dist, network_type="drive")` then read edge attributes | 1000 m | Named motorway/trunk/primary/secondary/tertiary, deduped and sorted |
| **Trail proximity** | `features_from_point((lat, lon), {highway: path/footway/cycleway, leisure: track}, dist)` | 400 m | Boolean “near trail”, boolean “Cherry Creek” in name, list of trail names |

Each call is **one (lat, lon) + one radius**. So for 37 hexes we do many such calls (with delay and retries). Results are cached in `cache/hex_attractors.json` and `cache/hex_place_context.json`.

---

## How it becomes LLM context

- **Attractors** → In the prompt under “Zone land uses” as “employment: many”, “retail: some”, etc. (from the normalized vector).
- **Place context** (neighbourhoods, roads, near_trail, near_cherry_creek, trail_names) → In the prompt as “Neighborhoods: …”, “Major roads nearby: …”, “Trail/greenway: Cherry Creek trail nearby …”, etc.

So: **grid gives us the list of hexes and their centroids → OSMnx queries each centroid with a radius → we cache counts and names → when building the prompt for that hex we pull from cache and paste into the “Zone context” section.** The LLM never sees raw coordinates; it sees the derived text (neighbourhood names, road names, “near Cherry Creek”, etc.).
