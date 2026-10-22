"""Matplotlib visualisations (used by the experiments script and the web app)."""
from __future__ import annotations
import io, base64
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#1f6feb", "#e8590c", "#2f9e44", "#ae3ec9", "#c92a2a", "#0c8599", "#f08c00", "#5f3dc4", "#e64980", "#495057",
           "#74b816", "#868e96"]
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})


def to_b64(fig) -> str:
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", dpi=110); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def network_figure(model, nodes_per_topic: int = 9, ncol: int = 5, fontsize: float = 6.6, width_in: float = 16.0,
                   r: float = 1.15, sx: float = 5.9, sy: float | None = None):
    """Each topic community is drawn as a ring of its highest-scoring words, rings are laid out
    on a grid (`ncol` columns). Edges are NPMI-weighted; the figure height follows from the grid."""
    G = model.graph; K = len(model.communities)
    nrows = int(np.ceil(K / ncol)); sy = sy or (4.6 if nrows <= 2 else 3.4)
    centre = lambda ci: ((ci % ncol - (ncol - 1) / 2) * sx, ((nrows - 1) / 2 - ci // ncol) * sy)
    W, H = (ncol - 1) * sx + 6.0, (nrows - 1) * sy + 3.2
    pos, node2c, lab = {}, {}, {}
    for ci in range(K):
        top = [int(i) for i in np.argsort(-model.word_score[:, ci])[:nodes_per_topic] if model.word_score[i, ci] > 0 and i in G]
        cx, cy = centre(ci)
        for j, n in enumerate(top):
            ang = 2 * np.pi * j / max(len(top), 1) + np.pi / 2
            pos[n] = (cx + r * np.cos(ang), cy + r * np.sin(ang)); node2c[n] = ci
            lab[n] = (cx + (r + 0.25) * np.cos(ang), cy + (r + 0.25) * np.sin(ang), ang)
    H_ = G.subgraph(list(pos)).copy()
    fig, ax = plt.subplots(figsize=(width_in, width_in * H / W)); ax.axis("off"); ax.grid(False)
    ws = np.array([d["weight"] for _, _, d in H_.edges(data=True)])
    for (u, v, d) in H_.edges(data=True):
        same = node2c[u] == node2c[v]
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=PALETTE[node2c[u] % len(PALETTE)] if same else "#999",
                lw=0.3 + 1.6 * (d["weight"] - ws.min()) / (np.ptp(ws) + 1e-9) if same else 0.25, alpha=0.30 if same else 0.18, zorder=1)
    for n, (x, y) in pos.items():
        c = node2c[n]; sc = model.word_score[n, c] / (model.word_score[:, c].max() + 1e-9)
        ax.scatter([x], [y], s=(60 + 190 * sc) * (width_in / 16) ** 2 * 1.6 if width_in < 16 else 60 + 190 * sc,
                   color=PALETTE[c % len(PALETTE)], edgecolor="white", linewidth=0.8, zorder=3)
        lx, ly, ang = lab[n]
        ha = "left" if np.cos(ang) > 0.25 else ("right" if np.cos(ang) < -0.25 else "center")
        va = "bottom" if np.sin(ang) > 0.5 else ("top" if np.sin(ang) < -0.5 else "center")
        ax.text(lx, ly, model.id2word[n].replace("_", " "), fontsize=fontsize, ha=ha, va=va, zorder=4)
    for ci in range(K):
        cx, cy = centre(ci)
        ax.text(cx, cy, f"T{ci}", ha="center", va="center", fontsize=fontsize + 4, weight="bold", color=PALETTE[ci % len(PALETTE)])
    ax.set_xlim(-W / 2, W / 2); ax.set_ylim(-H / 2, H / 2)
    return fig


def topic_bars(model, n_words: int = 8, ncols: int = 5, names: list[str] | None = None, figsize=None, fontsize: float = 7):
    K = len(model.topics_); nrows = int(np.ceil(K / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (3.0 * ncols, 2.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, t in enumerate(model.topics_):
        ws = t["words"][:n_words][::-1]
        axes[i].barh([w.replace("_", " ") for w, _ in ws], [s for _, s in ws], color=PALETTE[i % len(PALETTE)])
        axes[i].set_title(names[i] if names else f"Topic {i}", fontsize=fontsize + 1.2); axes[i].tick_params(labelsize=fontsize); axes[i].grid(False)
    for j in range(K, len(axes)):
        axes[j].axis("off")
    fig.tight_layout(); return fig


def doc_topic_heatmap(P: np.ndarray, y_true, labels: list[str], topic_names: list[str], figsize=(8.5, 5.2), fontsize: float = 8):
    """Mean topic distribution per ground-truth class."""
    y_true = np.asarray(y_true)
    M = np.vstack([P[y_true == c].mean(0) for c in range(len(labels))])
    fig, ax = plt.subplots(figsize=figsize); ax.grid(False)
    im = ax.imshow(M, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=fontsize)
    ax.set_xticks(range(len(topic_names))); ax.set_xticklabels(topic_names, rotation=0, ha="center", fontsize=fontsize)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] > 0.05:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=fontsize - 1.5, color="white" if M[i, j] > 0.5 else "#222")
    ax.set_xlabel("Detected graph topic"); ax.set_ylabel("Ground-truth class")
    fig.colorbar(im, ax=ax, fraction=0.03); fig.tight_layout(); return fig


def doc_distribution_bar(theta_row, names):
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    ax.barh(names[::-1], theta_row[::-1], color=[PALETTE[i % len(PALETTE)] for i in range(len(names))][::-1])
    ax.set_xlim(0, 1); ax.tick_params(labelsize=7); ax.grid(False); fig.tight_layout(); return fig
