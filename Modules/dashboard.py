"""
dashboard.py — Generates an interactive standalone HTML dashboard.

Reads trend_data.json and drift_events.json and produces dashboard.html
with Plotly charts embedded in a dark-themed layout.
"""

import json
import os

import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

from config import (
    DASHBOARD_FILE, DRIFT_FILE, TOP_N_KEYWORDS, TREND_FILE,
)

PALETTE = [
    "#4CC9F0", "#4361EE", "#7209B7", "#F72585", "#3F37C9",
    "#B5E48C", "#90E0EF", "#F4A261", "#E76F51", "#2EC4B6",
    "#FFBA08", "#9B5DE5",
]
BG = "#0D1117"
PAPER_BG = "#161B22"
FONT_COLOR = "#E6EDF3"
GRID_COLOR = "#21262D"


def _layout_defaults(**kwargs) -> dict:
    base = dict(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=BG,
        font=dict(family="Inter, sans-serif", color=FONT_COLOR, size=12),
        margin=dict(l=50, r=30, t=50, b=50),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID_COLOR),
    )
    base.update(kwargs)
    return base


# Individual charts 

def fig_paper_volume(trend_data: dict) -> go.Figure:
    periods = trend_data["periods"]
    volume  = [trend_data["volume"].get(p, 0) for p in periods]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=periods, y=volume,
        marker=dict(color=PALETTE[0], opacity=0.8, line=dict(color=PALETTE[1], width=1)),
        name="Papers published",
        hovertemplate="<b>%{x}</b><br>Papers: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=periods, y=volume,
        mode="lines", line=dict(color=PALETTE[3], width=2),
        name="Trend line", hoverinfo="skip",
    ))
    fig.update_layout(**_layout_defaults(
        title="AI Research Paper Volume Over Time",
        xaxis=dict(title="Quarter", gridcolor=GRID_COLOR),
        yaxis=dict(title="# Papers", gridcolor=GRID_COLOR),
    ))
    return fig


def fig_keyword_trends(trend_data: dict) -> go.Figure:
    periods   = trend_data["periods"]
    kw_trends = trend_data.get("kw_trends", [])
    kw_freq   = trend_data["kw_freq"]

    # Pick top-N keywords by growth rate (emerging) + top-N by total volume
    emerging = [k for k in kw_trends if k["trend"] == "increasing"][:TOP_N_KEYWORDS // 2]
    by_vol   = sorted(kw_trends, key=lambda x: x["total_count"], reverse=True)
    seen     = {k["keyword"] for k in emerging}
    top_vol  = [k for k in by_vol if k["keyword"] not in seen][:TOP_N_KEYWORDS // 2]
    selected = emerging + top_vol

    fig = go.Figure()
    for i, kw_info in enumerate(selected):
        kw     = kw_info["keyword"]
        series = [kw_freq.get(kw, {}).get(p, 0) for p in periods]
        dash   = "solid" if kw_info["trend"] == "increasing" else "dot"
        fig.add_trace(go.Scatter(
            x=periods, y=series,
            mode="lines+markers",
            name=kw[:30],
            line=dict(color=PALETTE[i % len(PALETTE)], width=2, dash=dash),
            marker=dict(size=5),
            hovertemplate=f"<b>{kw}</b><br>%{{x}}: %{{y}} papers<extra></extra>",
        ))

    fig.update_layout(**_layout_defaults(
        title="Keyword Frequency Trends (solid = emerging)",
        xaxis=dict(title="Quarter", gridcolor=GRID_COLOR),
        yaxis=dict(title="Papers mentioning keyword", gridcolor=GRID_COLOR),
        hovermode="x unified",
    ))
    return fig


def fig_topic_heatmap(trend_data: dict) -> go.Figure:
    periods      = trend_data["periods"]
    topic_dist   = trend_data["topic_dist"]
    topic_labels = trend_data.get("topic_labels", [])

    z = []
    for label in topic_labels:
        row = [topic_dist.get(p, [0] * len(topic_labels))[topic_labels.index(label)]
               for p in periods]
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=periods,
        y=[lbl[:35] for lbl in topic_labels],
        colorscale="Viridis",
        hovertemplate="Period: %{x}<br>Topic: %{y}<br>Weight: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Weight", tickfont=dict(color=FONT_COLOR)),
    ))
    fig.update_layout(**_layout_defaults(
        title="Topic Distribution Over Time (Heatmap)",
        xaxis=dict(title="Quarter", gridcolor=GRID_COLOR),
        yaxis=dict(title="", automargin=True),
        height=420,
    ))
    return fig


def fig_emerging_declining(trend_data: dict) -> go.Figure:
    kw_trends = trend_data.get("kw_trends", [])
    emerging  = sorted(
        [k for k in kw_trends if k["trend"] == "increasing"],
        key=lambda x: x["growth_rate"], reverse=True
    )[:10]
    declining = sorted(
        [k for k in kw_trends if k["trend"] == "decreasing"],
        key=lambda x: x["growth_rate"]
    )[:10]

    labels  = [k["keyword"][:30] for k in emerging] + [k["keyword"][:30] for k in declining]
    values  = [k["growth_rate"] for k in emerging] + [k["growth_rate"] for k in declining]
    colors  = ["#4CC9F0"] * len(emerging) + ["#F72585"] * len(declining)

    fig = go.Figure(go.Bar(
        y=labels, x=values,
        orientation="h",
        marker=dict(color=colors),
        hovertemplate="<b>%{y}</b><br>Growth rate: %{x:.2f}<extra></extra>",
    ))
    fig.update_layout(**_layout_defaults(
        title="Emerging (blue) vs Declining (red) Research Topics",
        xaxis=dict(title="Growth Rate (second half vs first half)", gridcolor=GRID_COLOR,
                   zeroline=True, zerolinecolor="#555"),
        yaxis=dict(automargin=True),
        height=480,
    ))
    return fig


def fig_drift_timeline(drift_events: list, trend_data: dict) -> go.Figure:
    periods = trend_data["periods"]
    volume  = [trend_data["volume"].get(p, 0) for p in periods]

    drift_periods = {e["period"] for e in drift_events}
    drift_x = [p for p in periods if p in drift_periods]
    drift_y = [trend_data["volume"].get(p, 0) for p in drift_x]
    drift_labels = []
    for p in drift_x:
        for e in drift_events:
            if e["period"] == p:
                drift_labels.append(
                    f"JS={e['js_divergence']:.3f}<br>↑ {e['gaining_topic'][:25]}"
                )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=periods, y=volume,
        mode="lines", fill="tozeroy",
        line=dict(color=PALETTE[0], width=2),
        fillcolor="rgba(76, 201, 240, 0.1)",
        name="Paper volume",
    ))
    if drift_x:
        fig.add_trace(go.Scatter(
            x=drift_x, y=drift_y,
            mode="markers+text",
            marker=dict(color=PALETTE[3], size=14, symbol="diamond",
                        line=dict(color="white", width=1)),
            text=["⚡"] * len(drift_x),
            textposition="top center",
            customdata=drift_labels,
            hovertemplate="<b>Drift Event</b><br>%{x}<br>%{customdata}<extra></extra>",
            name="Drift events",
        ))
    fig.update_layout(**_layout_defaults(
        title="Topic Drift Events on Paper Volume Timeline",
        xaxis=dict(title="Quarter", gridcolor=GRID_COLOR),
        yaxis=dict(title="# Papers", gridcolor=GRID_COLOR),
    ))
    return fig


#  HTML template 

def _chart_div(fig: go.Figure) -> str:
    return pyo.plot(fig, output_type="div", include_plotlyjs=False, config={"responsive": True})


def generate_dashboard(trend_data: dict, drift_events: list,
                       output: str = DASHBOARD_FILE):
    n_papers   = sum(trend_data["volume"].values())
    n_keywords = len(trend_data["kw_freq"])
    n_topics   = len(trend_data.get("topic_labels", []))
    n_drift    = len(drift_events)
    n_emerging = sum(1 for k in trend_data.get("kw_trends", []) if k["trend"] == "increasing")

    html_volume    = _chart_div(fig_paper_volume(trend_data))
    html_kw        = _chart_div(fig_keyword_trends(trend_data))
    html_heatmap   = _chart_div(fig_topic_heatmap(trend_data))
    html_emerge    = _chart_div(fig_emerging_declining(trend_data))
    html_drift     = _chart_div(fig_drift_timeline(drift_events, trend_data))

    # Top-N topic summary for table
    topic_rows = ""
    for i, label in enumerate(trend_data.get("topic_labels", [])):
        words = ", ".join(trend_data.get("topic_top_words", [[]])[i]) if i < len(trend_data.get("topic_top_words", [])) else ""
        topic_rows += f"<tr><td>{label}</td><td>{words}</td></tr>"

    # Drift event table rows
    drift_rows = ""
    for e in drift_events:
        drift_rows += (
            f"<tr>"
            f"<td>{e['period']}</td>"
            f"<td>{e['js_divergence']:.4f}</td>"
            f"<td class='gain'>{e['gaining_topic']}</td>"
            f"<td class='loss'>{e['losing_topic']}</td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Emerging Research Trends in AI Literature | Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet"/>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg: #0D1117; --surface: #161B22; --border: #21262D;
    --text: #E6EDF3; --muted: #8B949E; --accent: #4CC9F0;
    --green: #3FB950; --red: #F85149; --purple: #BC8CFF;
  }}
  body {{ background: var(--bg); color: var(--text); font-family: "Inter", sans-serif;
          line-height: 1.6; min-height: 100vh; }}
  header {{
    background: linear-gradient(135deg, #0D1117 0%, #1a1f36 50%, #0D1117 100%);
    border-bottom: 1px solid var(--border);
    padding: 2.5rem 2rem 2rem;
    text-align: center;
    position: relative; overflow: hidden;
  }}
  header::before {{
    content: ""; position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(76,201,240,0.08) 0%, transparent 70%);
  }}
  header h1 {{
    font-size: clamp(1.3rem, 3vw, 2rem); font-weight: 700;
    background: linear-gradient(90deg, #4CC9F0, #7209B7, #F72585);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
  }}
  header p {{ color: var(--muted); margin-top: .5rem; font-size: .95rem; }}
  .badge {{
    display: inline-block; background: rgba(76,201,240,0.15);
    border: 1px solid rgba(76,201,240,0.3); border-radius: 999px;
    padding: .2rem .8rem; font-size: .75rem; color: var(--accent);
    margin-top: .8rem; margin-right: .3rem;
  }}
  .stats-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1rem; padding: 1.5rem 2rem;
  }}
  .stat-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.2rem 1rem; text-align: center;
    transition: transform .15s, border-color .15s;
  }}
  .stat-card:hover {{ transform: translateY(-3px); border-color: var(--accent); }}
  .stat-card .value {{ font-size: 2rem; font-weight: 700; color: var(--accent); }}
  .stat-card .label {{ font-size: .78rem; color: var(--muted); margin-top: .2rem; }}
  .tabs {{ display: flex; gap: .5rem; padding: 0 2rem 1rem; flex-wrap: wrap; }}
  .tab-btn {{
    background: var(--surface); border: 1px solid var(--border);
    color: var(--muted); border-radius: 8px; padding: .5rem 1.1rem;
    cursor: pointer; font-size: .85rem; transition: all .15s;
  }}
  .tab-btn:hover, .tab-btn.active {{
    background: rgba(76,201,240,0.15); border-color: var(--accent);
    color: var(--accent);
  }}
  .tab-content {{ display: none; padding: 0 2rem 2rem; }}
  .tab-content.active {{ display: block; }}
  .chart-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; padding: 1rem; margin-bottom: 1.5rem;
    overflow: hidden;
  }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
  @media (max-width: 800px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  table {{ width: 100%; border-collapse: collapse; font-size: .85rem; }}
  th, td {{ padding: .6rem .8rem; text-align: left;
             border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 600; font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; }}
  tr:hover td {{ background: rgba(255,255,255,.03); }}
  .gain {{ color: var(--green); }}
  .loss {{ color: var(--red); }}
  .section-title {{ font-size: 1rem; font-weight: 600; color: var(--text);
                    margin-bottom: 1rem; padding-bottom: .5rem;
                    border-bottom: 1px solid var(--border); }}
  footer {{ text-align: center; padding: 2rem; color: var(--muted); font-size: .8rem; }}
</style>
</head>
<body>

<header>
  <h1>Detecting Emerging Research Trends in AI Literature</h1>
  <p>Multi-Keyword Temporal Analysis · Topic Modeling · Drift Detection</p>
  <span class="badge">LDA Topic Modeling</span>
  <span class="badge">Mann-Kendall Test</span>
  <span class="badge">Jensen-Shannon Divergence</span>
  <span class="badge">arXiv ({min(trend_data["periods"])} – {max(trend_data["periods"])})</span>
</header>

<div class="stats-grid">
  <div class="stat-card"><div class="value">{n_papers:,}</div><div class="label">Papers Analyzed</div></div>
  <div class="stat-card"><div class="value">{n_keywords}</div><div class="label">Unique Keywords</div></div>
  <div class="stat-card"><div class="value">{n_topics}</div><div class="label">Latent Topics</div></div>
  <div class="stat-card"><div class="value">{n_emerging}</div><div class="label">Emerging Trends</div></div>
  <div class="stat-card"><div class="value">{n_drift}</div><div class="label">Drift Events</div></div>
  <div class="stat-card"><div class="value">{len(trend_data["periods"])}</div><div class="label">Time Windows</div></div>
</div>

<div class="tabs">
  <button class="tab-btn active" onclick="showTab('overview')">📊 Overview</button>
  <button class="tab-btn" onclick="showTab('keywords')">🔑 Keyword Trends</button>
  <button class="tab-btn" onclick="showTab('topics')">🧠 Topic Analysis</button>
  <button class="tab-btn" onclick="showTab('drift')">⚡ Drift Events</button>
</div>

<div id="overview" class="tab-content active">
  <div class="chart-card">{html_volume}</div>
  <div class="two-col">
    <div class="chart-card">{html_emerge}</div>
    <div class="chart-card">
      <div class="section-title">Discovered Research Topics</div>
      <table>
        <thead><tr><th>Topic</th><th>Top Keywords</th></tr></thead>
        <tbody>{topic_rows}</tbody>
      </table>
    </div>
  </div>
</div>

<div id="keywords" class="tab-content">
  <div class="chart-card">{html_kw}</div>
</div>

<div id="topics" class="tab-content">
  <div class="chart-card">{html_heatmap}</div>
</div>

<div id="drift" class="tab-content">
  <div class="chart-card">{html_drift}</div>
  <div class="chart-card">
    <div class="section-title">All Detected Drift Events</div>
    { '<p style="color:var(--muted);font-size:.9rem">No drift events detected at current threshold.</p>'
      if not drift_events else
      f'<table><thead><tr><th>Period</th><th>JS Divergence</th><th>Gaining Topic</th><th>Losing Topic</th></tr></thead><tbody>{drift_rows}</tbody></table>'
    }
  </div>
</div>

<footer>Built with arXiv API · scikit-learn LDA · SciPy · Plotly | MS Research Project</footer>

<script>
function showTab(id) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  event.target.classList.add('active');
}}
</script>
</body>
</html>"""

    with open(output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Dashboard {output}  (open in browser)")


# Standalone runner 

def run_dashboard():
    print("Loading analysis results …")
    with open(TREND_FILE) as f:
        trend_data = json.load(f)
    with open(DRIFT_FILE) as f:
        drift_events = json.load(f)
    generate_dashboard(trend_data, drift_events)
