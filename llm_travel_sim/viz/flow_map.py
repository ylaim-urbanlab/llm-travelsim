"""
Visualization: animated flow map, OD arc map, mode split chart.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from llm_travel_sim.data.h3_grid import HexCell
from llm_travel_sim.data.census import FEATURE_NAMES
from llm_travel_sim.data.osm import ATTRACTOR_NAMES
from llm_travel_sim.llm.trip_generator import Trip
from llm_travel_sim.simulation.state import SimulationState


MODE_COLORS = {
    "driving": "#e74c3c",
    "transit": "#3498db",
    "cycling": "#2ecc71",
    "walking": "#f39c12",
    "rideshare": "#9b59b6",
}


def _build_hex_geojson(cells: Dict[str, HexCell], population_snapshot: List[float]) -> dict:
    """Build GeoJSON FeatureCollection for hexes with population values."""
    try:
        import h3
    except ImportError:
        return {"type": "FeatureCollection", "features": []}

    features = []
    for i, (hex_id, cell) in enumerate(cells.items()):
        boundary = h3.cell_to_boundary(hex_id)
        # h3 returns (lat, lon) pairs; GeoJSON needs (lon, lat)
        coords = [[lon, lat] for lat, lon in boundary]
        coords.append(coords[0])  # close polygon

        pop = population_snapshot[i] if i < len(population_snapshot) else 0

        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {
                "hex_id": hex_id,
                "lat": cell.lat,
                "lon": cell.lon,
                "population": round(pop, 1),
            }
        })
    return {"type": "FeatureCollection", "features": features}


def _bar(val: float, width: int = 10) -> str:
    filled = int(val * width)
    return "█" * filled + "░" * (width - filled)


def _hex_popup(
    hex_id: str,
    pop: float,
    demographic_features: Optional[Dict],
    attractor_features: Optional[Dict],
    cluster_memberships: Optional[Dict],
    cluster_labels: Optional[List[str]],
    hex_populations: Optional[Dict],
    journal_entries: List[str],
) -> str:
    rows = []

    # Header
    rows.append(f"<b style='font-size:13px'>{hex_id[:12]}…</b>")
    census_pop = hex_populations.get(hex_id, 0) if hex_populations else 0
    rows.append(f"<b>Residents (ACS):</b> {census_pop:,.0f} &nbsp; <b>Sim pop:</b> {pop:,.0f}")
    rows.append("<hr style='margin:4px 0'>")

    # Demographics
    if demographic_features and hex_id in demographic_features:
        rows.append("<b>Demographics</b>")
        rows.append("<table style='font-size:11px;width:100%'>")
        raw_d = demographic_features[hex_id]
        d_list = raw_d.tolist() if hasattr(raw_d, "tolist") else list(raw_d)
        for name, val in zip(FEATURE_NAMES, d_list):
            label = name.replace("_norm", "").replace("_", " ")
            rows.append(
                f"<tr><td style='padding-right:6px'>{label}</td>"
                f"<td style='font-family:monospace;color:#555'>{_bar(val)}</td>"
                f"<td style='text-align:right'>{val:.2f}</td></tr>"
            )
        rows.append("</table><hr style='margin:4px 0'>")

    # Attractors
    if attractor_features and hex_id in attractor_features:
        rows.append("<b>Attractors</b>")
        rows.append("<table style='font-size:11px;width:100%'>")
        raw_a = attractor_features[hex_id]
        a_list = raw_a.tolist() if hasattr(raw_a, "tolist") else list(raw_a)
        for name, val in zip(ATTRACTOR_NAMES, a_list):
            rows.append(
                f"<tr><td style='padding-right:6px'>{name}</td>"
                f"<td style='font-family:monospace;color:#555'>{_bar(val)}</td>"
                f"<td style='text-align:right'>{val:.2f}</td></tr>"
            )
        rows.append("</table><hr style='margin:4px 0'>")

    # Cluster memberships
    if cluster_memberships and cluster_labels and hex_id in cluster_memberships:
        rows.append("<b>Cluster profile</b>")
        rows.append("<table style='font-size:11px;width:100%'>")
        raw_m = cluster_memberships[hex_id]
        m_list = raw_m.tolist() if hasattr(raw_m, "tolist") else list(raw_m)
        pairs = sorted(zip(cluster_labels, m_list), key=lambda x: -x[1])
        for label, m in pairs:
            if m > 0.05:
                rows.append(
                    f"<tr><td style='padding-right:6px'>{label}</td>"
                    f"<td style='font-family:monospace;color:#555'>{_bar(m)}</td>"
                    f"<td style='text-align:right'>{m:.0%}</td></tr>"
                )
        rows.append("</table><hr style='margin:4px 0'>")

    # Journal entries
    if journal_entries:
        rows.append("<b>Sample trips</b>")
        for e in journal_entries[:4]:
            rows.append(f"<p style='font-size:11px;margin:2px 0'><i>{e}</i></p>")

    return "<div style='width:280px;max-height:420px;overflow-y:auto;font-family:sans-serif;font-size:12px;line-height:1.5'>" + "".join(rows) + "</div>"


def generate_flow_map(
    cells: Dict[str, HexCell],
    state: SimulationState,
    output_dir: str,
    demographic_features: Optional[Dict] = None,
    attractor_features: Optional[Dict] = None,
    cluster_memberships: Optional[Dict] = None,
    cluster_labels: Optional[List[str]] = None,
    hex_populations: Optional[Dict] = None,
    peak_start_hour: float = 7.0,
    peak_end_hour: float = 9.0,
) -> str:
    """Generate an HTML flow map using folium."""
    out_path = Path(output_dir) / "flow_map.html"

    try:
        import folium
        from folium.plugins import HeatMap
    except ImportError:
        print("[viz] folium not installed, skipping flow map generation")
        return str(out_path)

    try:
        import h3
    except ImportError:
        print("[viz] h3 not installed, skipping flow map generation")
        return str(out_path)

    print("[viz] Generating flow map...")

    # Base map centered on Denver CBD
    m = folium.Map(
        location=[39.7392, -104.9903],
        zoom_start=12,
        tiles="CartoDB positron",
    )

    # --- Hex population layer (final state) ---
    if state.population_log:
        final_pop = state.population_log[-1]
    else:
        final_pop = state.population.tolist()

    for i, (hex_id, cell) in enumerate(cells.items()):
        try:
            boundary = h3.cell_to_boundary(hex_id)
            coords_ll = [[lat, lon] for lat, lon in boundary]
            pop = final_pop[i] if i < len(final_pop) else 0
            max_pop = max(final_pop) if final_pop else 1
            intensity = min(1.0, pop / max_pop)
            color = f"#{int(255 * intensity):02x}{int(100 * (1 - intensity)):02x}50"

            journals = state.journal_log.get(hex_id, {})
            journal_entries = []
            for entries in journals.values():
                journal_entries.extend(entries)

            popup_html = _hex_popup(
                hex_id=hex_id,
                pop=pop,
                demographic_features=demographic_features,
                attractor_features=attractor_features,
                cluster_memberships=cluster_memberships,
                cluster_labels=cluster_labels,
                hex_populations=hex_populations,
                journal_entries=journal_entries,
            )

            folium.Polygon(
                locations=coords_ll,
                fill=True,
                fill_color=color,
                fill_opacity=0.5,
                color="#333",
                weight=0.5,
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"{hex_id[:8]}… | residents: {int(hex_populations.get(hex_id, pop) if hex_populations else pop):,}",
            ).add_to(m)
        except Exception:
            continue

    # --- OD arc map for peak hour trips ---
    peak_trips = [
        t for t in state.trip_log
        if _is_peak_hour(t.departure_time, peak_start_hour, peak_end_hour)
    ]
    print(f"[viz] Drawing {min(len(peak_trips), 200)} peak hour OD arcs "
          f"({len(peak_trips)} total peak trips)")

    hex_centroids = {hid: (cell.lat, cell.lon) for hid, cell in cells.items()}

    for trip in peak_trips[:200]:  # cap at 200 arcs for performance
        orig = hex_centroids.get(trip.origin)
        dest = hex_centroids.get(trip.destination)
        if not orig or not dest or orig == dest:
            continue
        color = MODE_COLORS.get(trip.mode, "#666")
        folium.PolyLine(
            locations=[list(orig), list(dest)],
            color=color,
            weight=1.5,
            opacity=0.5,
            tooltip=f"{trip.mode} | {trip.purpose} | {trip.departure_time}",
        ).add_to(m)

    # --- Legend ---
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background: white; padding: 10px; border-radius: 5px;
                border: 1px solid #ccc; font-family: sans-serif; font-size: 12px;">
    <b>Peak Hour Trip Modes</b><br>
    """ + "".join(
        f'<span style="color:{c}">&#9650;</span> {m}<br>'
        for m, c in MODE_COLORS.items()
    ) + """
    <br><b>Hex color = relative population</b><br>
    (darker = more people)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(str(out_path))
    print(f"[viz] Flow map saved to {out_path}")
    return str(out_path)


def _is_peak_hour(departure_time: str, start: float, end: float) -> bool:
    try:
        parts = departure_time.split(":")
        h = int(parts[0]) + int(parts[1]) / 60.0
        return start <= h < end
    except Exception:
        return False


def generate_mode_split_chart(state: SimulationState, output_dir: str) -> str:
    """Generate a simple HTML mode split chart using inline JS/Chart.js."""
    out_path = Path(output_dir) / "mode_split.html"

    # Bin by timestep
    timestep_modes: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for trip in state.trip_log:
        timestep_modes[trip.hex_context_time][trip.mode] += 1

    times = sorted(timestep_modes.keys())
    modes = list(MODE_COLORS.keys())

    datasets = []
    for mode in modes:
        data = [timestep_modes[t].get(mode, 0) for t in times]
        color = MODE_COLORS[mode]
        datasets.append({
            "label": mode,
            "data": data,
            "backgroundColor": color,
            "borderColor": color,
            "fill": False,
        })

    chart_data = {
        "labels": times,
        "datasets": datasets,
    }

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>Mode Split Over Time</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
<h2>Trip Mode Split by Timestep</h2>
<canvas id="chart" width="1200" height="400"></canvas>
<script>
const ctx = document.getElementById('chart').getContext('2d');
new Chart(ctx, {{
  type: 'line',
  data: {json.dumps(chart_data)},
  options: {{
    responsive: true,
    scales: {{
      x: {{ title: {{ display: true, text: 'Time' }} }},
      y: {{ title: {{ display: true, text: 'Number of Trips' }}, stacked: false }}
    }}
  }}
}});
</script>
</body>
</html>"""

    out_path.write_text(html)
    print(f"[viz] Mode split chart saved to {out_path}")
    return str(out_path)


def print_summary_stats(state: SimulationState):
    """Print summary statistics to stdout."""
    total_trips = len(state.trip_log)
    if total_trips == 0:
        print("[viz] No trips generated.")
        return

    mode_counts = defaultdict(int)
    purpose_counts = defaultdict(int)
    for t in state.trip_log:
        mode_counts[t.mode] += 1
        purpose_counts[t.purpose] += 1

    print("\n" + "=" * 50)
    print("SIMULATION SUMMARY")
    print("=" * 50)
    print(f"Total trips generated: {total_trips:,}")
    print(f"Total timesteps: {len(state.population_log)}")
    print(f"Unique origin hexes: {len(set(t.origin for t in state.trip_log))}")
    print(f"Unique destination hexes: {len(set(t.destination for t in state.trip_log))}")
    print("\nMode split:")
    for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
        pct = count / total_trips * 100
        print(f"  {mode:12s}: {count:5d} ({pct:.1f}%)")
    print("\nTrip purpose split:")
    for purpose, count in sorted(purpose_counts.items(), key=lambda x: -x[1]):
        pct = count / total_trips * 100
        print(f"  {purpose:12s}: {count:5d} ({pct:.1f}%)")

    print("\nSample journal entries:")
    journals_shown = 0
    for trip in state.trip_log:
        if trip.journal and journals_shown < 5:
            print(f"  [{trip.mode}/{trip.purpose} @ {trip.departure_time}] {trip.journal}")
            journals_shown += 1
    print("=" * 50)
