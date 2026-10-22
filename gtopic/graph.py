"""Word co-occurrence network construction (sliding window + NPMI weighting)."""
from __future__ import annotations
import numpy as np
import networkx as nx
import scipy.sparse as sp


def cooccurrence_matrix(docs: list[list[int]], vocab_size: int, window: int = 8) -> sp.csr_matrix:
    """Symmetric co-occurrence counts using a sliding window of `window` tokens.
    Vectorised: all documents are concatenated and pairs straddling a document
    boundary are masked out."""
    lens = np.array([len(d) for d in docs])
    if lens.sum() == 0:
        return sp.csr_matrix((vocab_size, vocab_size))
    flat = np.concatenate([np.asarray(d, dtype=np.int64) for d in docs if len(d)])
    doc_id = np.repeat(np.arange(len(lens)), lens)
    rows, cols = [], []
    for off in range(1, window):
        if off >= len(flat):
            break
        ok = (doc_id[:-off] == doc_id[off:]) & (flat[:-off] != flat[off:])
        rows.append(flat[:-off][ok]); cols.append(flat[off:][ok])
    r, c = np.concatenate(rows), np.concatenate(cols)
    m = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(vocab_size, vocab_size)).tocsr()
    return (m + m.T).tocsr()


def build_graph(counts: sp.csr_matrix, id2word: list[str], min_count: int = 3,
                top_k: int = 25, min_weight: float = 0.05) -> nx.Graph:
    """Convert counts to a weighted undirected graph.

    weight(x,y) = NPMI(x,y) = log(p(x,y)/(p(x)p(y))) / -log p(x,y)
    Edges with count < min_count or weight <= min_weight are dropped, then each
    node keeps its `top_k` strongest edges (union over both end-points)."""
    C = counts.tocsr().copy()
    C.data[C.data < min_count] = 0
    C.eliminate_zeros()
    total = C.sum()
    if total == 0:
        G = nx.Graph()
        G.add_nodes_from((i, {"word": id2word[i]}) for i in range(len(id2word)))
        return G
    px = np.asarray(C.sum(axis=1)).ravel() / max(total / 2.0, 1e-12)
    coo = C.tocoo()
    m = coo.row < coo.col
    r, c, v = coo.row[m], coo.col[m], coo.data[m]
    pxy = v / (total / 2.0)
    w = np.log(pxy / (px[r] * px[c])) / np.maximum(-np.log(pxy), 1e-12)
    ok = w > min_weight
    r, c, w = r[ok], c[ok], w[ok]
    keep = np.zeros(len(w), dtype=bool)
    for side in (r, c):
        order = np.lexsort((-w, side))
        s_sorted = side[order]
        rank = np.arange(len(order)) - np.searchsorted(s_sorted, s_sorted, side="left")
        keep[order[rank < top_k]] = True
    G = nx.Graph()
    G.add_nodes_from((i, {"word": id2word[i]}) for i in range(len(id2word)))
    G.add_weighted_edges_from(zip(r[keep].tolist(), c[keep].tolist(), w[keep].tolist()))
    G.remove_nodes_from([n for n in list(G) if G.degree(n) == 0])
    return G
