# What We Feed the LLM (and What’s Missing)

## Currently in the prompt

- **Zone ID**: Raw H3 index (e.g. `88268cda85fffff`) — not human-readable.
- **Time**: Clock time and period (morning_peak, midday, etc.).
- **Population**: Own zone and neighbors’ average.
- **Land uses**: Employment, retail, education, recreation, transit as “many/some/few” + %.
- **Demographics**: Income, poverty, age, tenure, vehicles, WFH, commute, occupation.
- **Destinations**: List of neighbor zone IDs (H3 indices only).

**Now also in the prompt (when available):**

- **Location**: Distance and direction from downtown Denver, plus lat/lon.
- **Neighborhoods**: Names from OSM `place=neighbourhood` nodes within 150 m of the hex centroid (cached in `hex_place_context.json`).
- **Major roads**: Named motorway/trunk/primary/secondary/tertiary ways within 1 km, deduped and sorted by importance (same cache).

Still **not** given: named landmarks (Union Station, City Park), or per-destination place labels.

So the model has to stay vague (“Denver”, “my apartment”, “nearby retail”, “across town”) or hallucinate place names. Adding real geography would make journals more specific and OD/mode choices more plausible.

---

## Options (by effort)

| Addition | Source | Effort | Notes |
|----------|--------|--------|--------|
| **Distance + direction from CBD** | Config + hex centroid | Low | e.g. “~2.3 mi northeast of downtown Denver”. No new APIs. |
| **Lat/lon** | Grid (HexCell) | Low | Already available; just pass into context/prompt. |
| **Neighborhood name** | Reverse geocode (Nominatim) or Denver neighborhoods GeoJSON | Medium | Cache per hex. Nominatim is rate-limited. |
| **Nearby street/highway names** | OSM Overpass (ways with `name` in bbox) | Medium | One extra Overpass query per hex, cache with attractors. |
| **Anchor labels** | Heuristics or manual | Medium | e.g. “downtown”, “retail corridor” for a few hexes; reference in prompt. |

---

## Recommendation (remaining)

- **Optional**: Label a few anchor hexes (downtown, main retail) and mention them in the prompt for richer “where to” context.
