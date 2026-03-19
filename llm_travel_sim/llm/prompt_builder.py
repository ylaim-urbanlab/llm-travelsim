"""
Build structured prompts for LLM trip generation.
One prompt per cluster per hex — focused single-persona calls.
"""
from __future__ import annotations
from typing import List

from llm_travel_sim.data.census import FEATURE_NAMES
from llm_travel_sim.data.osm import ATTRACTOR_NAMES


_DEMO_DESCRIPTIONS = {
    "median_hh_income_norm":  ("household income",          lambda v: f"${int(v * 200000):,}/yr"),
    "pct_below_poverty":      ("poverty rate",              lambda v: f"{v*100:.0f}%"),
    "pct_snap":               ("SNAP/food stamp usage",     lambda v: f"{v*100:.0f}%"),
    "median_age_norm":        ("median age",                lambda v: f"{int(v * 80)} yrs"),
    "pct_under_18":           ("children under 18",         lambda v: f"{v*100:.0f}%"),
    "pct_family_households":  ("family households",         lambda v: f"{v*100:.0f}%"),
    "pct_renter":             ("renter-occupied",           lambda v: f"{v*100:.0f}%"),
    "pct_zero_vehicle":       ("zero-car households",       lambda v: f"{v*100:.0f}%"),
    "pct_work_from_home":     ("work-from-home rate",       lambda v: f"{v*100:.0f}%"),
    "mean_commute_norm":      ("avg commute",               lambda v: f"{int(v * 90)} min"),
    "pct_mgmt_professional":  ("management/professional",   lambda v: f"{v*100:.0f}%"),
    "pct_service":            ("service sector",            lambda v: f"{v*100:.0f}%"),
}


def _describe_demographics(demo_vec: List[float]) -> str:
    lines = []
    for i, name in enumerate(FEATURE_NAMES):
        if name in _DEMO_DESCRIPTIONS and i < len(demo_vec):
            label, fmt = _DEMO_DESCRIPTIONS[name]
            val = demo_vec[i]
            if val > 0.05:
                lines.append(f"  - {label}: {fmt(val)}")
    return "\n".join(lines) if lines else "  - mixed demographics"


def _describe_attractors(attr_vec: List[float]) -> str:
    lines = []
    for i, name in enumerate(ATTRACTOR_NAMES):
        if i < len(attr_vec):
            val = attr_vec[i]
            if val > 0.05:
                level = "many" if val > 0.5 else ("some" if val > 0.2 else "few")
                lines.append(f"  - {name}: {level} ({val*100:.0f}% of max)")
    return "\n".join(lines) if lines else "  - mostly residential, few attractors"


def build_cluster_prompt(
    context: dict,
    cluster_label: str,
    cluster_membership: float,
    n_stories: int,
    neighbor_hex_ids: List[str],
) -> str:
    """
    Build a focused prompt for ONE cluster persona within one hex.
    Called k times per hex per timestep (once per cluster).
    """
    hex_id      = context["hex_id"]
    time_str    = context["time"]
    period      = context["period"].replace("_", " ")
    own_pop     = int(context["own_population"])
    own_demo    = context["own_demographics"]
    own_attr    = context["own_attractors"]
    nb_attr     = context["neighbor_attractors_avg"]
    nb_pop      = int(context["neighbor_population_avg"])

    # Approximate how many people of this cluster type are in the zone
    cluster_pop = int(own_pop * cluster_membership)

    dest_options = neighbor_hex_ids[:6] if neighbor_hex_ids else [hex_id]
    dest_list    = ", ".join(dest_options)

    location_line = f"- Location: {context['location_description']}\n" if context.get("location_description") else ""
    neighborhoods = context.get("neighborhoods", [])
    roads = context.get("roads", [])
    near_trail = context.get("near_trail", False)
    near_cherry_creek = context.get("near_cherry_creek", False)
    trail_names = context.get("trail_names", [])
    place_lines = []
    if neighborhoods:
        place_lines.append(f"- Neighborhoods: {', '.join(neighborhoods)}")
    if roads:
        place_lines.append(f"- Major roads nearby: {', '.join(roads[:8])}")  # cap at 8
    if near_trail or near_cherry_creek:
        if near_cherry_creek:
            place_lines.append("- Trail/greenway: Cherry Creek trail nearby — good for walking and cycling.")
        elif trail_names:
            place_lines.append(f"- Trail/greenway: {', '.join(trail_names[:3])} nearby — good for walking and cycling.")
        else:
            place_lines.append("- Trail/greenway: path/trail access nearby — good for walking and cycling.")
    place_block = "\n".join(place_lines) + "\n" if place_lines else ""

    prompt = f"""You are a travel demand simulator generating synthetic trip stories for Denver, Colorado.

## Your persona
You are generating trips for people who are: **{cluster_label}**
This group represents ~{cluster_membership*100:.0f}% of this zone ({cluster_pop:,} people).

## Zone context
- Zone ID: {hex_id}
{location_line}{place_block}- Time: {time_str} ({period})
- Total zone population: ~{own_pop:,}

Zone land uses:
{_describe_attractors(own_attr)}

Zone demographics:
{_describe_demographics(own_demo)}

Surrounding area (neighbors avg {nb_pop:,} people):
{_describe_attractors(nb_attr)}

## Task
Generate exactly {n_stories} trip stories for **{cluster_label}** people departing this zone during this 30-minute window.

All {n_stories} trips must be plausible for this specific persona type. Vary the mode and purpose naturally — do not force uniformity. Mode choices should reflect what **{cluster_label}** people actually do: car-dependent personas drive more; car-free personas walk, cycle, or use transit more. Include cycling where plausible (e.g. short trips, recreation, Denver has bike lanes and bike share).

Available destination zone IDs: {dest_list}
(Use {hex_id} for local trips that stay in zone)

For each trip output a JSON object with exactly these fields:
- "origin": "{hex_id}"
- "destination": one of the zone IDs above
- "mode": one of ["driving", "transit", "cycling", "walking", "rideshare"]
- "purpose": one of ["work", "shopping", "school", "recreation", "dining", "medical", "social", "return_home", "other"]
- "departure_time": time string like "08:15"
- "journal": one sentence, first-person, specific to this persona and Denver

Output ONLY a valid JSON array of {n_stories} objects. No other text."""

    return prompt
