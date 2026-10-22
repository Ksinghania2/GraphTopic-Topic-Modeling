"""Architecture + flowchart diagrams for the report."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch

def box(ax, x, y, w, h, text, fc="#e7f0ff", ec="#1f6feb", fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", wrap=True)

def arrow(ax, p, q, text=None):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.3, color="#333"))
    if text: ax.text((p[0] + q[0]) / 2 + 0.05, (p[1] + q[1]) / 2 + 0.07, text, fontsize=7.5, color="#c92a2a")

# ---------------- architecture
fig, ax = plt.subplots(figsize=(11, 5.2)); ax.set_xlim(0, 11); ax.set_ylim(0, 5.2); ax.axis("off")
cols = ["#fff4e6", "#e7f0ff", "#ebfbee", "#f3f0ff", "#fff0f6"]; ecs = ["#e8590c", "#1f6feb", "#2f9e44", "#5f3dc4", "#c2255c"]
heads = ["INPUT", "PRE-PROCESSING", "GRAPH CONSTRUCTION", "TOPIC DISCOVERY", "OUTPUT / EVALUATION"]
items = [["CSV / TXT corpus", "20 Newsgroups", "Synthetic news corpus"],
         ["Tokenise + lowercase", "Stop-word removal", "Lemmatisation", "NPMI phrase detection", "Vocabulary pruning"],
         ["Sliding-window\nco-occurrence counts", "NPMI edge weights", "Top-k edge pruning", "Weighted graph G(V,E)"],
         ["Louvain community\ndetection", "Word scoring\nk_in · k_in/k", "Soft membership\nmatrix", "Document-topic\ninference"],
         ["Topics + top words", "Network visualisation", "Coherence / NMI / purity", "Flask web interface"]]
for i in range(5):
    x = 0.1 + i * 2.18
    ax.add_patch(FancyBboxPatch((x, 0.15), 2.0, 4.85, boxstyle="round,pad=0.02,rounding_size=0.1", fc=cols[i], ec=ecs[i], lw=1.6, alpha=0.55))
    ax.text(x + 1.0, 4.7, heads[i], ha="center", fontsize=7.6, weight="bold", color=ecs[i])
    n = len(items[i]); h = 0.62; gap = (4.2 - n * h) / (n + 1) if n else 0
    for j, t in enumerate(items[i]):
        y = 4.3 - (j + 1) * gap - (j + 1) * h + h * 0.0 - 0.05
        box(ax, x + 0.12, y, 1.76, h, t, fc="white", ec=ecs[i], fs=7.6)
    if i < 4: arrow(ax, (x + 2.0, 2.6), (x + 2.18, 2.6))
fig.savefig("results/figures/fig_architecture.png", bbox_inches="tight", dpi=170); plt.close(fig)

# ---------------- flowchart
fig, ax = plt.subplots(figsize=(6.2, 9.2)); ax.set_xlim(0, 6.4); ax.set_ylim(0, 9.2); ax.axis("off")
cx = 3.1; W, H = 4.3, 0.55
steps = ["Load corpus (CSV / folder / generator)", "Clean text: tokenise, remove stop-words, lemmatise", "Detect phrases (NPMI ≥ τ) and prune vocabulary",
         "Count word pairs inside sliding window w", "Weight edges with NPMI, keep top-k neighbours", "Run Louvain community detection (resolution γ)",
         "Drop tiny communities; score words per community"]
ys = [8.1 - i * 0.95 for i in range(len(steps))]
ax.text(cx, 9.0, "START", ha="center", va="center", fontsize=9, weight="bold", bbox=dict(boxstyle="round,pad=0.35", fc="#2f9e44", ec="#2f9e44", alpha=0.25))
arrow(ax, (cx, 8.85), (cx, ys[0] + H))
for i, (t, y) in enumerate(zip(steps, ys)):
    box(ax, cx - W / 2, y, W, H, t, fs=8)
    if i < len(steps) - 1: arrow(ax, (cx, y), (cx, ys[i + 1] + H))
yd = ys[-1] - 1.35
arrow(ax, (cx, ys[-1]), (cx, yd + 0.8))
ax.add_patch(Polygon([[cx, yd + 0.8], [cx + 1.5, yd + 0.4], [cx, yd], [cx - 1.5, yd + 0.4]], fc="#fff9db", ec="#f08c00", lw=1.4))
ax.text(cx, yd + 0.4, "Modularity high &\ntopics coherent?", ha="center", va="center", fontsize=8)
ax.annotate("", xy=(0.45, ys[5] + H / 2), xytext=(cx - 1.5, yd + 0.4), arrowprops=dict(arrowstyle="-|>", color="#c92a2a", lw=1.3, connectionstyle="angle,angleA=0,angleB=-90"))
ax.plot([cx - 1.5, 0.45], [yd + 0.4, yd + 0.4], color="#c92a2a", lw=1.3); ax.plot([0.45, 0.45], [yd + 0.4, ys[5] + H / 2], color="#c92a2a", lw=1.3)
ax.annotate("", xy=(cx - W / 2, ys[5] + H / 2), xytext=(0.45, ys[5] + H / 2), arrowprops=dict(arrowstyle="-|>", color="#c92a2a", lw=1.3))
ax.text(0.5, yd + 0.5, "No: tune γ, w", fontsize=7.5, color="#c92a2a", rotation=90, va="bottom")
ye = yd - 0.9
arrow(ax, (cx, yd), (cx, ye + H)); ax.text(cx + 0.1, yd - 0.3, "Yes", fontsize=7.5, color="#2f9e44")
box(ax, cx - W / 2, ye, W, H, "Infer document–topic distributions", fs=8)
box(ax, cx - W / 2, ye - 0.85, W, H, "Evaluate: NPMI, NMI, ARI, purity; visualise", fs=8)
arrow(ax, (cx, ye), (cx, ye - 0.3))
arrow(ax, (cx, ye - 0.85), (cx, ye - 1.15))
ax.text(cx, ye - 1.3, "END", ha="center", va="center", fontsize=9, weight="bold", bbox=dict(boxstyle="round,pad=0.35", fc="#c92a2a", ec="#c92a2a", alpha=0.25))
ax.set_ylim(ye - 1.7, 9.3)
fig.savefig("results/figures/fig_flowchart.png", bbox_inches="tight", dpi=170); plt.close(fig)
