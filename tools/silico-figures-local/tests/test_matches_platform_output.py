"""Check the stand-in reproduces the platform's own figure geometry exactly.

Every ``want`` below is copied verbatim out of a tracked figure bundle that the
*real* silico-figures package produced, so this is a comparison against recorded
platform output rather than against an expectation invented here:

``figures/arm_scores``      an x guide line on a two-panel grid   (run 3)
``figures/step_stability``  a y guide line striped row/col="all"  (run 2)

Run it as a script; it exits non-zero on any mismatch.

    python tools/silico-figures-local/tests/test_matches_platform_output.py
"""
import json, plotly.graph_objects as go
from plotly.subplots import make_subplots
from silico_figures import add_reference_line, apply_theme

fails = []
def check(name, got, want):
    ok = got == want
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        print("   got :", json.dumps(got, sort_keys=True))
        print("   want:", json.dumps(want, sort_keys=True))
        fails.append(name)

# --- x-line, 2-panel grid (exemplar: run3 figures/arm_scores) ---------------
fig = make_subplots(rows=1, cols=2)
for c in (1, 2):
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1]), row=1, col=c)
apply_theme(fig, height=640)
add_reference_line(fig, x=0.028601970174156315, label="shuffled-label null", row=1, col=1)
d = json.loads(fig.to_json())["layout"]
check("x-line shape", d["shapes"][0], {
    "line": {"color": "#31362E", "dash": "dot", "width": 2}, "opacity": 1, "type": "line",
    "x0": 0.028601970174156315, "x1": 0.028601970174156315, "xref": "x",
    "y0": 0, "y1": 1, "yref": "y domain"})
check("x-line annotation", d["annotations"][0], {
    "bgcolor": "rgba(246, 245, 240, 0.85)", "borderpad": 2,
    "font": {"color": "#31362E", "size": 12}, "showarrow": False,
    "text": "shuffled-label null", "x": 0.028601970174156315, "xanchor": "left",
    "xref": "x", "xshift": 6, "y": 1, "yanchor": "top", "yref": "y domain"})

# --- y-line with row="all"/col="all" (exemplar: run2 figures/step_stability) -
fig2 = make_subplots(rows=2, cols=3)
for r in (1, 2):
    for c in (1, 2, 3):
        fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1]), row=r, col=c)
apply_theme(fig2, height=760)
add_reference_line(fig2, y=0.999, label="0.999 threshold", row="all", col="all")
d2 = json.loads(fig2.to_json())["layout"]
check("y-line shape", d2["shapes"][0], {
    "line": {"color": "#31362E", "dash": "dot", "width": 2}, "opacity": 1, "type": "line",
    "x0": 0, "x1": 1, "xref": "x domain", "y0": 0.999, "y1": 0.999, "yref": "y"})
check("y-line annotation", d2["annotations"][0], {
    "bgcolor": "rgba(246, 245, 240, 0.85)", "borderpad": 2,
    "font": {"color": "#31362E", "size": 12}, "showarrow": False,
    "text": "0.999 threshold", "x": 1, "xanchor": "right", "xref": "x domain",
    "y": 0.999, "yanchor": "bottom", "yref": "y", "yshift": 6})
check("row/col='all' striped all 6 panels", len(d2["shapes"]), 6)

# --- theme actually applied ------------------------------------------------
lay = json.loads(fig.to_json())["layout"]["template"]["layout"]
check("colorway", lay["colorway"], ["#C4650D","#4E728A","#2E6E4E","#988453","#B9605B","#7495AB","#84713A","#31362E"])
check("font", lay["font"], {"color":"#1D272A","family":"'Suisse Intl', -apple-system, BlinkMacSystemFont, Arial, sans-serif","size":13})
check("margin", lay["margin"], {"autoexpand":True,"b":24,"l":24,"r":24,"t":40})
check("height", json.loads(fig.to_json())["layout"]["height"], 640)

print()
raise SystemExit(1 if fails else 0)
