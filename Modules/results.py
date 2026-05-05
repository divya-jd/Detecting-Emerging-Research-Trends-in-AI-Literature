"""
results.py — Report-quality graphs + narrative analysis directly in Python.
Run after the full pipeline completes.

Usage:
    python3 results.py
"""

import json, sys, os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Check data 
for path in ["data/trend_data.json", "data/drift_events.json"]:
    if not os.path.exists(path):
        print(f"✗  {path} not found. Run: python3 pipeline.py --analyze")
        sys.exit(1)

with open("data/trend_data.json") as f:  trend = json.load(f)
with open("data/drift_events.json") as f: drift = json.load(f)

periods      = trend["periods"]
kw_trends    = trend.get("kw_trends", [])
kw_freq      = trend.get("kw_freq", {})
volume       = trend.get("volume", {})
topic_labels = trend.get("topic_labels", [])
topic_words  = trend.get("topic_top_words", [])
topic_dist   = trend.get("topic_dist", {})

emerging  = sorted([k for k in kw_trends if k["trend"] == "increasing"],
                   key=lambda x: x["growth_rate"], reverse=True)
declining = sorted([k for k in kw_trends if k["trend"] == "decreasing"],
                   key=lambda x: x["growth_rate"])
stable    = [k for k in kw_trends if k["trend"] == "stable"]

# Check if we have enough data 
if len(periods) < 2:
    print("\n Only 1 time window found — not enough for trend analysis.")
    print("   Run: python3 pipeline.py --analyze  (papers already fetched)\n")
    sys.exit(0)

# Known AI milestones by quarter 
AI_MILESTONES = {
    "2020Q2": "GPT-3 released by OpenAI",
    "2021Q1": "CLIP and DALL-E introduced multimodal AI",
    "2021Q3": "GitHub Copilot launched; code generation surged",
    "2021Q4": "Large-scale vision-language models proliferated",
    "2022Q1": "Stable Diffusion and DALL-E 2 sparked diffusion model research",
    "2022Q2": "Chinchilla scaling laws published; LLM efficiency focus grew",
    "2022Q3": "Instruction-tuned LLMs (InstructGPT) dominated research",
    "2022Q4": "ChatGPT launched — massive shift to foundation models",
    "2023Q1": "GPT-4 and LLaMA released; open-source LLM explosion",
    "2023Q2": "AI alignment & hallucination concerns surged post-ChatGPT",
    "2023Q3": "Llama 2, Mistral, Code Llama — efficient LLMs proliferated",
    "2023Q4": "Mixture-of-Experts (MoE) and efficient inference became key",
    "2024Q1": "Sora (video generation) and multimodal models expanded",
    "2024Q2": "Agentic AI and RAG systems became dominant research themes",
    "2024Q3": "Small language models and on-device AI gained traction",
    "2024Q4": "AI reasoning models (o1-style) and test-time compute surged",
}

# Colour palette 
BG      = "#0D1117"
SURFACE = "#161B22"
BORDER  = "#21262D"
TEXT    = "#E6EDF3"
MUTED   = "#8B949E"
BLUE    = "#4CC9F0"
PINK    = "#F72585"
GREEN   = "#3FB950"
PURPLE  = "#9B5DE5"
ORANGE  = "#F4A261"
YELLOW  = "#FFBA08"
PALETTE = [BLUE, PINK, "#B5E48C", ORANGE, PURPLE,
           YELLOW, "#4361EE", "#2EC4B6", "#E76F51", "#7209B7",
           "#90E0EF", "#F85149"]

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    SURFACE,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   MUTED,
    "text.color":        TEXT,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "grid.color":        BORDER,
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "legend.facecolor":  SURFACE,
    "legend.edgecolor":  BORDER,
})

# TERMINAL OUTPUT

W = 72

def section(title):
    print(f"\n{'─'*W}")
    print(f"  {title}")
    print(f"{'─'*W}")

print("\n" + "═"*W)
print(f"  AI RESEARCH TREND ANALYSIS  ·  arXiv {periods[0]} → {periods[-1]}")
print("═"*W)
print(f"  Papers   : {sum(volume.values()):,}  |  Quarters: {len(periods)}"
      f"  |  Keywords: {len(kw_trends)}  |  Topics: {len(topic_labels)}")
#  Emerging 
section(f" TRENDING NOW — {len(emerging)} Emerging Keywords")
if emerging:
    print(f"  {'#':<4} {'Keyword':<40} {'Growth':>8}   Bar")
    print(f"  {'─'*4} {'─'*40} {'─'*8}   {'─'*18}")
    for i, k in enumerate(emerging[:15], 1):
        bar = "▓" * min(int(abs(k["growth_rate"]) * 4) + 1, 18)
        print(f"  {i:<4} {k['keyword']:<40} {k['growth_rate']:>+8.2f}   {bar}")
else:
    print("  (none detected — run full pipeline for multi-year data)")

# Declining
section(f"  LOSING TRACTION — {len(declining)} Declining Keywords")
if declining:
    print(f"  {'#':<4} {'Keyword':<40} {'Growth':>8}")
    print(f"  {'─'*4} {'─'*40} {'─'*8}")
    for i, k in enumerate(declining[:10], 1):
        print(f"  {i:<4} {k['keyword']:<40} {k['growth_rate']:>+8.2f}")
else:
    print("  (none detected)")

#  Stable 
section(f" STABLE : {len(stable)} Keywords holding steady")
if stable:
    names = [k["keyword"] for k in stable[:14]]
    # Print in 2 columns
    for j in range(0, len(names), 2):
        left  = names[j][:34]
        right = names[j+1][:34] if j+1 < len(names) else ""
        print(f"  • {left:<36}  • {right}")
    if len(stable) > 14:
        print(f"  ... and {len(stable)-14} more")

# Topics 
section("DISCOVERED RESEARCH TOPICS (LDA)")
print(f"  {'ID':<6} {'Topic Label':<30} Top Keywords")
print(f"  {'─'*6} {'─'*30} {'─'*32}")
for i, (lbl, wds) in enumerate(zip(topic_labels, topic_words), 1):
    print(f"  T{i:<5} {lbl:<30} {', '.join(wds[:5])}")

# Drift events 
section(f" TOPIC DRIFT EVENTS — {len(drift)} detected")
if drift:
    print(f"  {'Period':<10} {'JS²':>7}  {'Gaining Topic':<30} Losing Topic")
    print(f"  {'─'*10} {'─'*7}  {'─'*30} {'─'*28}")
    for e in drift:
        js2 = e['js_divergence']
        print(f"  {e['period']:<10} {js2:>7.4f}  "
              f"{e['gaining_topic'][:29]:<30} {e['losing_topic'][:28]}")
else:
    print("  (none at current threshold — will appear after full analysis)")

# Narrative analysis 
section("NARRATIVE ANALYSIS — What the data tells us")

# Phase detection based on periods
first_year = int(periods[0][:4])
last_year  = int(periods[-1][:4])

# Describe overall trajectory
if emerging:
    top3 = [k["keyword"] for k in emerging[:3]]
    print(f"\n WHAT'S RISING:")
    print(f"  The strongest emerging trends are: {', '.join(top3)}.")
    print(f"  These topics show consistent growth across the study period,")
    print(f"  indicating sustained and increasing community interest.")

if declining:
    bot3 = [k["keyword"] for k in declining[:3]]
    print(f"\n WHAT'S FADING:")
    print(f"  Research attention is declining for: {', '.join(bot3)}.")
    print(f"  These were established topics that the field is moving past,")
    print(f"  likely superseded by more capable or efficient alternatives.")

# Drift narrative
if drift:
    print(f"\n KEY PIVOTS (Topic Drift Events):")
    for e in sorted(drift, key=lambda x: x["js_divergence"], reverse=True):
        period  = e["period"]
        gained  = e["gaining_topic"]
        lost    = e["losing_topic"]
        js2     = e["js_divergence"]
        year    = period[:4]
        quarter = period[4:]
        milestone = AI_MILESTONES.get(period, "")

        intensity = "major" if js2 > 0.015 else "notable" if js2 > 0.008 else "mild"
        print(f"\n  In {quarter} {year}, a {intensity} shift was detected (JS²={js2:.4f}).")
        print(f"    The field moved TOWARD  -> {gained}")
        print(f"    The field moved AWAY FROM -> {lost}")
        if milestone:
            print(f"    Context: This coincides with — {milestone}")
else:
    # No drift detected, compute and display actual JS² values for diagnostics
    from scipy.spatial.distance import jensenshannon as _js
    _topic_dist = trend.get("topic_dist", {})
    _js_vals = []
    for _i in range(1, len(periods)):
        _prev = np.array(_topic_dist.get(periods[_i-1], [])) + 1e-10
        _curr = np.array(_topic_dist.get(periods[_i], [])) + 1e-10
        if len(_prev) > 1 and len(_curr) > 1:
            _prev /= _prev.sum()
            _curr /= _curr.sum()
            _js_vals.append(float(_js(_prev, _curr) ** 2))
    
    if _js_vals:
        _max_js = max(_js_vals)
        print(f"\n  No drift events exceeded the threshold (JS² > {0.005}).")
        print(f"  Actual max JS² observed: {_max_js:.6f}")
        if _max_js < 0.0001:
            print(f"  Extremely low divergence — this likely indicates")
            print(f"  the LDA model collapsed (one topic dominates all).")
            print(f"  Re-run: python3 pipeline.py --analyze")
        else:
            print(f"  The research landscape shifted gradually (no sharp pivots).")
    else:
        print(f"\n  No drift computed — insufficient topic distribution data.")

# Period summary
print(f"\n PERIOD SUMMARY ({first_year}–{last_year}):")
by_year = defaultdict(int)
for p, cnt in volume.items():
    by_year[p[:4]] += cnt
for yr in sorted(by_year):
    bar = "█" * min(by_year[yr] // 10, 30)
    print(f"  {yr}  {by_year[yr]:>5} papers  {bar}")

print("\n" + "═"*W + "\n")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE — Keyword frequency trends
# ════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(15, 6))
fig2.suptitle("Keyword Frequency Trends Over Time  "
              "(solid ↑ = emerging  ·  dashed ↓ = declining)",
              fontsize=12, fontweight="bold", color=TEXT)

top_show = emerging[:6] + declining[:3]
xpos2 = list(range(len(periods)))
for i, k in enumerate(top_show):
    kw     = k["keyword"]
    series = [kw_freq.get(kw, {}).get(p, 0) for p in periods]
    ls     = "-"  if k["trend"] == "increasing" else "--"
    mark   = "o"  if k["trend"] == "increasing" else "s"
    arrow  = "↑"  if k["trend"] == "increasing" else "↓"
    ax2.plot(xpos2, series, ls, color=PALETTE[i % len(PALETTE)],
             linewidth=2.2, marker=mark, markersize=4.5,
             label=f"{arrow} {kw[:38]}")
# Mark drift quarters on trend chart
for e in drift:
    if e["period"] in periods:
        xi = periods.index(e["period"])
        ax2.axvline(xi, color=YELLOW, linewidth=1.2, linestyle=":", alpha=0.6)

ax2.set_xticks(xpos2)
ax2.set_xticklabels(periods, rotation=45, ha="right", fontsize=8)
ax2.set_ylabel("Papers mentioning keyword", color=MUTED)
ax2.set_xlabel("Quarter", color=MUTED)
ax2.legend(fontsize=8.5, loc="upper left", ncol=2,
           facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)
ax2.grid(alpha=0.3)
ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))


plt.tight_layout(rect=[0, 0, 1, 0.96])
print(" Displaying graph — close window to exit.\n")
plt.show()