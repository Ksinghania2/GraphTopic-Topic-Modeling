"""Reproduces every number / figure used in the report.  python run_experiments.py"""
import json, sys, time, os
import numpy as np
import matplotlib.pyplot as plt
from gtopic import GraphTopicModel
from gtopic.corpus import generate_corpus, LABELS
from gtopic.evaluate import doc_term_binary, evaluate
from gtopic.baselines import run_lda, run_nmf
from gtopic.visualize import *

FIG = "results/figures"; os.makedirs(FIG, exist_ok=True)
R = {}

def prep(texts, **kw):
    m = GraphTopicModel(**kw); m.pre.fit(texts); docs = m.pre.transform(texts); m.id2word = m.pre.id2word
    return m, docs

def graph_eval(m, docs, y, B):
    ids = {w: i for i, w in enumerate(m.id2word)}
    P = m.doc_topic_matrix(docs); pred = P.argmax(1)
    return evaluate([[ids[w] for w in ws] for ws in m.top_words()], B, y, pred), P

# ------------------------------------------------------------------ dataset
texts, y = generate_corpus(12000, seed=7); y = np.array(y)
m, docs = prep(texts); V = len(m.id2word)
R["dataset"] = {"n_docs": len(texts), "n_tokens_raw": int(sum(len(t.split()) for t in texts)),
                "n_tokens_clean": int(sum(len(d) for d in docs)), "vocab": V, "n_classes": len(LABELS),
                "avg_len_clean": round(float(np.mean([len(d) for d in docs])), 1), "n_phrases": len(m.pre.phrases),
                "class_counts": {LABELS[c]: int((y == c).sum()) for c in range(len(LABELS))}}
B = doc_term_binary(docs, V)

# --------------------------------------------------------------- main model
m.fit_ids(docs)
ev, P = graph_eval(m, docs, y, B)
R["graph"] = {**m.summary(), **{k: v for k, v in ev.items() if k != "per_topic_npmi"}}
K = len(m.communities)
# name topics by majority ground-truth class
pred = P.argmax(1); names = []
for k in range(K):
    c = np.bincount(y[pred == k], minlength=len(LABELS)).argmax() if (pred == k).any() else 0
    names.append(LABELS[c])
R["topics"] = [{"id": k, "name": names[k], "size_words": m.topics_[k]["n_words"], "docs": int((pred == k).sum()),
                "npmi": round(ev["per_topic_npmi"][k], 3), "words": [w for w, _ in m.topics_[k]["words"]]} for k in range(K)]
R["bridge_words"] = [{"word": w, "participation": round(p, 3), "topics": [names[t] for t in ts]} for w, p, ts in m.bridge_words(10)]
poly = []
for w in ("model", "energy", "market", "season", "power", "policy", "study", "network", "record", "training", "growth", "cell", "launch"):
    if w in m.pre.vocab and m.pre.vocab[w] in m.graph:
        i = m.pre.vocab[w]; top = np.argsort(-m.membership[i])[:3]
        poly.append({"word": w, "participation": round(float(m.participation[i]), 3),
                     "topics": [f"{names[t]} ({m.membership[i, t]:.2f})" for t in top if m.membership[i, t] > 0.05]})
R["polysemous"] = sorted(poly, key=lambda r: -r["participation"])
degs = np.array([d for _, d in m.graph.degree()])
R["graph_stats"] = {"avg_degree": round(float(degs.mean()), 1), "max_degree": int(degs.max()),
                    "clustering": round(float(__import__("networkx").average_clustering(m.graph, weight="weight")), 3),
                    "components": int(__import__("networkx").number_connected_components(m.graph))}

# ----------------------------------------------- baselines (3 seeds, k = K)
rows = {"Graph (Louvain)": [], "LDA": [], "NMF": []}
for seed in (1, 2, 3):
    mg = GraphTopicModel(seed=seed); mg.pre = m.pre; mg.id2word = m.id2word; mg.fit_ids(docs)
    e, _ = graph_eval(mg, docs, y, B); e["time"] = mg.timing["total"] - mg.timing.get("preprocess", 0); e["k"] = len(mg.communities)
    rows["Graph (Louvain)"].append(e)
    for name, fn in (("LDA", run_lda), ("NMF", run_nmf)):
        r = fn(docs, V, K, seed=seed); e = evaluate(r["topics"], B, y, r["pred"]); e["time"] = r["time"]; e["k"] = K
        rows[name].append(e)
comp = {}
for name, lst in rows.items():
    comp[name] = {k: [round(float(np.mean([x[k] for x in lst])), 4), round(float(np.std([x[k] for x in lst])), 4)]
                  for k in ("npmi_coherence", "diversity", "nmi", "ari", "purity", "time")}
R["comparison"] = comp
print("comparison", json.dumps(comp, indent=1))

# ------------------------------------------- community detection ablation
abl = {}
for meth in ("louvain", "greedy_modularity", "label_propagation"):
    mm = GraphTopicModel(method=meth, seed=1); mm.pre = m.pre; mm.id2word = m.id2word; mm.fit_ids(docs)
    e, _ = graph_eval(mm, docs, y, B)
    abl[meth] = {"topics": len(mm.communities), "modularity": round(mm.modularity, 4), "npmi": round(e["npmi_coherence"], 4),
                 "nmi": round(e["nmi"], 4), "time": round(mm.timing["community"], 3)}
R["ablation_method"] = abl

# ----------------------------------------------------- resolution sweep
sweep = []
for res in (0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5):
    mm = GraphTopicModel(resolution=res, seed=1); mm.pre = m.pre; mm.id2word = m.id2word; mm.fit_ids(docs)
    e, _ = graph_eval(mm, docs, y, B)
    sweep.append({"res": res, "k": len(mm.communities), "mod": mm.modularity, "npmi": e["npmi_coherence"], "nmi": e["nmi"]})
R["resolution_sweep"] = sweep
# ----------------------------------------------------------- window sweep
ws = []
for w in (3, 5, 8, 12, 16, 24):
    mm = GraphTopicModel(window=w, seed=1); mm.pre = m.pre; mm.id2word = m.id2word; mm.fit_ids(docs)
    e, _ = graph_eval(mm, docs, y, B)
    ws.append({"window": w, "k": len(mm.communities), "mod": mm.modularity, "npmi": e["npmi_coherence"], "nmi": e["nmi"], "edges": mm.graph.number_of_edges()})
R["window_sweep"] = ws

# ------------------------------------------------------ noise robustness
rob = []
for noise in (0.05, 0.18, 0.35, 0.5):
    t2, y2 = generate_corpus(6000, seed=11, noise=noise); y2 = np.array(y2)
    m2, d2 = prep(t2); B2 = doc_term_binary(d2, len(m2.id2word)); m2.fit_ids(d2)
    e, _ = graph_eval(m2, d2, y2, B2); row = {"noise": noise, "Graph": e["nmi"], "k": len(m2.communities)}
    for name, fn in (("LDA", run_lda), ("NMF", run_nmf)):
        r = fn(d2, len(m2.id2word), len(LABELS)); row[name] = evaluate(r["topics"], B2, y2, r["pred"])["nmi"]
    # graph with k forced equal to truth is not possible for Louvain; report found k
    rob.append(row); print("noise", row)
R["noise_robustness"] = rob

# ------------------------------------------------------------ scalability
big_t, big_y = generate_corpus(40000, seed=21)
mb = GraphTopicModel(); mb.pre.fit(big_t[:12000]); big_docs = mb.pre.transform(big_t); mb.id2word = mb.pre.id2word
sc = []
for n in (1000, 2500, 5000, 10000, 20000, 40000):
    d = big_docs[:n]; row = {"n": n, "tokens": int(sum(len(x) for x in d))}
    g = GraphTopicModel(); g.pre = mb.pre; g.id2word = mb.id2word; g.fit_ids(d); row["Graph"] = g.timing["total"] - g.timing.get("preprocess", 0)
    row["LDA"] = run_lda(d, len(mb.id2word), 10)["time"]; row["NMF"] = run_nmf(d, len(mb.id2word), 10)["time"]
    sc.append(row); print("scale", row)
R["scalability"] = sc

json.dump(R, open("results/results.json", "w"), indent=1, default=float)

# ================================================================= FIGURES
def save(fig, name): fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight", dpi=170); plt.close(fig)
save(network_figure(m), "fig_network")
tn = [f"T{k}: {names[k]}" for k in range(K)]
save(topic_bars(m, names=tn), "fig_topic_words")
save(doc_topic_heatmap(P, y, LABELS, [f"T{k}" for k in range(K)]), "fig_heatmap")

fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
res = [s["res"] for s in sweep]
ax[0].plot(res, [s["k"] for s in sweep], "o-", color=PALETTE[0]); ax[0].axhline(10, ls="--", c="#888", lw=1); ax[0].set_title("Topics found"); 
ax[1].plot(res, [s["mod"] for s in sweep], "o-", color=PALETTE[1]); ax[1].set_title("Modularity Q")
ax[2].plot(res, [s["npmi"] for s in sweep], "o-", color=PALETTE[2], label="NPMI coherence"); ax[2].plot(res, [s["nmi"] for s in sweep], "s-", color=PALETTE[3], label="NMI"); ax[2].legend(frameon=False, fontsize=8); ax[2].set_title("Quality")
for a in ax: a.set_xlabel("Louvain resolution γ")
fig.tight_layout(); save(fig, "fig_resolution")

fig, ax = plt.subplots(1, 3, figsize=(12, 3.4)); cols = {"Graph (Louvain)": PALETTE[0], "LDA": PALETTE[1], "NMF": PALETTE[2]}
for a, (key, ttl) in zip(ax, (("npmi_coherence", "NPMI coherence ↑"), ("nmi", "NMI vs. ground truth ↑"), ("time", "Fit time (s) ↓"))):
    a.bar(list(comp), [comp[n][key][0] for n in comp], yerr=[comp[n][key][1] for n in comp], color=[cols[n] for n in comp], capsize=4)
    a.set_title(ttl); a.grid(axis="x", alpha=0)
fig.tight_layout(); save(fig, "fig_comparison")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
for name, c in (("Graph", PALETTE[0]), ("LDA", PALETTE[1]), ("NMF", PALETTE[2])):
    ax[0].plot([s["n"] for s in sc], [s[name] for s in sc], "o-", label=name, color=c)
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("Number of documents"); ax[0].set_ylabel("Seconds"); ax[0].legend(frameon=False); ax[0].set_title("Scalability")
for name, c in (("Graph", PALETTE[0]), ("LDA", PALETTE[1]), ("NMF", PALETTE[2])):
    ax[1].plot([r["noise"] for r in rob], [r[name] for r in rob], "o-", label=name, color=c)
ax[1].set_xlabel("Cross-topic noise probability"); ax[1].set_ylabel("NMI"); ax[1].set_title("Robustness to noise"); ax[1].legend(frameon=False)
fig.tight_layout(); save(fig, "fig_scal_noise")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
ax[0].hist(degs, bins=30, color=PALETTE[0]); ax[0].set_xlabel("Node degree"); ax[0].set_ylabel("Words"); ax[0].set_title("Degree distribution")
ax[1].bar([f"T{k}" for k in range(K)], [t["docs"] for t in R["topics"]], color=PALETTE[:K]); ax[1].set_title("Documents per topic"); ax[1].grid(axis="x", alpha=0)
fig.tight_layout(); save(fig, "fig_graphstats")
print(json.dumps({k: R[k] for k in ("dataset", "graph", "graph_stats", "ablation_method")}, indent=1, default=float))
