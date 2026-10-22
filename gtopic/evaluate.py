"""Topic-quality metrics: NPMI coherence, topic diversity, NMI / ARI / purity."""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def doc_term_binary(docs: list[list[int]], V: int) -> sp.csc_matrix:
    sets = [set(d) for d in docs]
    rows = np.repeat(np.arange(len(docs)), [len(s) for s in sets])
    cols = np.concatenate([np.fromiter(s, dtype=np.int64) for s in sets if s]) if any(s for s in sets) else np.array([], dtype=np.int64)
    return sp.csc_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(docs), V))


def npmi_coherence(topics: list[list[int]], B: sp.csc_matrix, eps: float = 1e-12) -> list[float]:
    """C_NPMI per topic using document-level co-occurrence on the reference corpus."""
    n = B.shape[0]
    df = np.asarray(B.sum(0)).ravel()
    scores = []
    for words in topics:
        vals = []
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                a, b = words[i], words[j]
                co = B[:, a].multiply(B[:, b]).sum()
                if co == 0:
                    vals.append(-1.0); continue
                pxy, px, py = co / n, df[a] / n, df[b] / n
                vals.append(np.log(pxy / (px * py) + eps) / -np.log(pxy + eps))
        scores.append(float(np.mean(vals)) if vals else 0.0)
    return scores


def topic_diversity(topics: list[list]) -> float:
    allw = [w for t in topics for w in t]
    return len(set(allw)) / max(len(allw), 1)


def purity(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tot = 0
    for c in np.unique(y_pred):
        tot += np.bincount(y_true[y_pred == c]).max()
    return tot / len(y_true)


def evaluate(topics_ids: list[list[int]], B, y_true=None, y_pred=None) -> dict:
    coh = npmi_coherence(topics_ids, B)
    out = {"npmi_coherence": float(np.mean(coh)), "per_topic_npmi": coh, "diversity": topic_diversity(topics_ids)}
    if y_true is not None and y_pred is not None:
        out.update(nmi=float(normalized_mutual_info_score(y_true, y_pred)),
                   ari=float(adjusted_rand_score(y_true, y_pred)), purity=float(purity(y_true, y_pred)))
    return out
