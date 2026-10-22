"""Re-draws the result figures at report size from results/results.json (+ one model fit).
   python make_figures.py   (run after run_experiments.py)"""
import json, numpy as np
import matplotlib.pyplot as plt
from gtopic import GraphTopicModel
from gtopic.corpus import generate_corpus, LABELS
from gtopic.visualize import *

FIG = "results/figures"; R = json.load(open("results/results.json"))
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
def save(fig, name): fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight", dpi=200); plt.close(fig)

texts, y = generate_corpus(12000, seed=7); y = np.array(y)
m = GraphTopicModel().fit(texts); docs = m.pre.transform(texts); P = m.doc_topic_matrix(docs); pred = P.argmax(1)
names = [R["topics"][k]["name"] for k in range(len(R["topics"]))]
save(network_figure(m, nodes_per_topic=8, ncol=2, fontsize=6.6, width_in=6.8, r=1.0, sx=5.9, sy=2.9), "fig_network")
save(topic_bars(m, names=[f"T{k}: {n}" for k, n in enumerate(names)], ncols=3, figsize=(7.2, 7.0), fontsize=6.5), "fig_topic_words")
save(doc_topic_heatmap(P, y, LABELS, [f"T{k}" for k in range(len(names))], figsize=(6.4, 4.3), fontsize=7.5), "fig_heatmap")

comp, sweep, rob, sc = R["comparison"], R["resolution_sweep"], R["noise_robustness"], R["scalability"]
cols = {"Graph (Louvain)": PALETTE[0], "LDA": PALETTE[1], "NMF": PALETTE[2]}
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
for a, (key, ttl) in zip(ax, (("npmi_coherence", "NPMI coherence ↑"), ("nmi", "NMI vs. ground truth ↑"), ("time", "Fit time (s) ↓"))):
    a.bar(["Graph", "LDA", "NMF"], [comp[n][key][0] for n in comp], yerr=[comp[n][key][1] for n in comp], color=[cols[n] for n in comp], capsize=3)
    a.set_title(ttl); a.grid(axis="x", alpha=0)
fig.tight_layout(); save(fig, "fig_comparison")

fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5)); res = [s["res"] for s in sweep]
ax[0].plot(res, [s["k"] for s in sweep], "o-", color=PALETTE[0]); ax[0].axhline(10, ls="--", c="#888", lw=1); ax[0].set_title("Topics found")
ax[1].plot(res, [s["mod"] for s in sweep], "o-", color=PALETTE[1]); ax[1].set_title("Modularity Q")
ax[2].plot(res, [s["npmi"] for s in sweep], "o-", color=PALETTE[2], label="NPMI"); ax[2].plot(res, [s["nmi"] for s in sweep], "s-", color=PALETTE[3], label="NMI"); ax[2].legend(frameon=False, fontsize=7); ax[2].set_title("Quality")
for a in ax: a.set_xlabel("Louvain resolution γ")
fig.tight_layout(); save(fig, "fig_resolution")

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
for name, c in (("Graph", PALETTE[0]), ("LDA", PALETTE[1]), ("NMF", PALETTE[2])):
    ax[0].plot([s["n"] for s in sc], [s[name] for s in sc], "o-", label=name, color=c, ms=4)
    ax[1].plot([r["noise"] for r in rob], [r[name] for r in rob], "o-", label=name, color=c, ms=4)
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("Number of documents"); ax[0].set_ylabel("Seconds"); ax[0].legend(frameon=False); ax[0].set_title("Scalability")
ax[1].set_xlabel("Cross-topic noise probability"); ax[1].set_ylabel("NMI"); ax[1].set_title("Robustness to noise"); ax[1].legend(frameon=False)
fig.tight_layout(); save(fig, "fig_scal_noise")

degs = np.array([d for _, d in m.graph.degree()])
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
ax[0].hist(degs, bins=30, color=PALETTE[0]); ax[0].set_xlabel("Node degree"); ax[0].set_ylabel("Words"); ax[0].set_title("Degree distribution")
ax[1].bar([f"T{k}" for k in range(len(names))], [(pred == k).sum() for k in range(len(names))], color=PALETTE[:len(names)]); ax[1].set_title("Documents per topic"); ax[1].grid(axis="x", alpha=0)
fig.tight_layout(); save(fig, "fig_graphstats")
print("figures written")
