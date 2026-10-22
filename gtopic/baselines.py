"""LDA and NMF baselines (scikit-learn) run on the same vocabulary as the graph model."""
from __future__ import annotations
import time
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfTransformer


def _counts(docs, V):
    lens = [len(d) for d in docs]
    rows = np.repeat(np.arange(len(docs)), lens)
    cols = np.concatenate([np.asarray(d) for d in docs if len(d)]) if any(len(d) for d in docs) else np.array([], dtype=np.int64)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(docs), V))


def run_lda(docs, V, k, topn=10, seed=42, max_iter=15):
    X = _counts(docs, V); t = time.time()
    m = LatentDirichletAllocation(n_components=k, learning_method="online", max_iter=max_iter,
                                  random_state=seed, batch_size=512).fit(X)
    theta = m.transform(X)
    return {"topics": [list(np.argsort(-r)[:topn]) for r in m.components_], "pred": theta.argmax(1),
            "time": time.time() - t}


def run_nmf(docs, V, k, topn=10, seed=42):
    X = TfidfTransformer().fit_transform(_counts(docs, V)); t = time.time()
    m = NMF(n_components=k, init="nndsvd", random_state=seed, max_iter=300).fit(X)
    W = m.transform(X)
    return {"topics": [list(np.argsort(-r)[:topn]) for r in m.components_], "pred": W.argmax(1),
            "time": time.time() - t}
