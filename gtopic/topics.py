"""Graph-based topic model: communities of the word co-occurrence network are topics."""
from __future__ import annotations
import time
import numpy as np
import networkx as nx
from networkx.algorithms import community as nxc

from .preprocess import Preprocessor
from .graph import cooccurrence_matrix, build_graph


class GraphTopicModel:
    def __init__(self, window: int = 8, resolution: float = 1.0, min_count: int = 3, top_k_edges: int = 25,
                 min_weight: float = 0.05, method: str = "louvain", min_topic_size: int = 10,
                 topn_words: int = 10, seed: int = 42, preprocessor: Preprocessor | None = None):
        self.window, self.resolution, self.min_count = window, resolution, min_count
        self.top_k_edges, self.min_weight, self.method = top_k_edges, min_weight, method
        self.min_topic_size, self.topn_words, self.seed = min_topic_size, topn_words, seed
        self.pre = preprocessor or Preprocessor()
        self.timing: dict[str, float] = {}

    def fit(self, texts: list[str], prefit: bool = False) -> "GraphTopicModel":
        t0 = time.time()
        if not prefit:
            self.pre.fit(texts)
        docs = self.pre.transform(texts)
        self.id2word = self.pre.id2word
        self.timing["preprocess"] = time.time() - t0
        return self.fit_ids(docs)

    def fit_ids(self, docs: list[list[int]]) -> "GraphTopicModel":
        t0 = time.time()
        self.counts = cooccurrence_matrix(docs, len(self.id2word), self.window)
        self.graph = build_graph(self.counts, self.id2word, self.min_count, self.top_k_edges, self.min_weight)
        self.timing["graph"] = time.time() - t0
        t0 = time.time()
        allc = self._detect(self.graph)
        self.modularity = nxc.modularity(self.graph, allc, weight="weight") if allc else 0.0   # full partition
        comms = [c for c in allc if len(c) >= self.min_topic_size]                              # drop tiny communities
        comms.sort(key=lambda c: -sum(self.graph.degree(n, weight="weight") for n in c))
        self.communities = comms
        self.timing["community"] = time.time() - t0
        self._score_words()
        self.timing["total"] = sum(v for k, v in self.timing.items() if k != "total")
        return self

    def _detect(self, G: nx.Graph):
        if self.method == "louvain":
            return [set(c) for c in nxc.louvain_communities(G, weight="weight", resolution=self.resolution, seed=self.seed)]
        if self.method == "label_propagation":
            return [set(c) for c in nxc.asyn_lpa_communities(G, weight="weight", seed=self.seed)]
        if self.method == "greedy_modularity":
            return [set(c) for c in nxc.greedy_modularity_communities(G, weight="weight", resolution=self.resolution)]
        raise ValueError(f"unknown method {self.method}")

    def _score_words(self):
        """k_in(w,c): weight of w's edges that stay inside community c.
        score(w,c) = k_in(w,c) * k_in(w,c)/k(w)  -> strong AND focused on c.
        membership(w,c) = k_in(w,c)/sum_c' k_in(w,c')  (soft, used for documents)."""
        G, K, V = self.graph, len(self.communities), len(self.id2word)
        kin = np.zeros((V, K))
        node2c = {n: ci for ci, comm in enumerate(self.communities) for n in comm}
        for u, v, d in G.edges(data=True):
            if v in node2c:
                kin[u, node2c[v]] += d["weight"]
            if u in node2c:
                kin[v, node2c[u]] += d["weight"]
        strength = np.zeros(V)
        for n in G:
            strength[n] = G.degree(n, weight="weight")
        self.word_score = kin * kin / np.maximum(strength, 1e-12)[:, None]
        self.membership = kin / np.maximum(kin.sum(1, keepdims=True), 1e-12)
        self.participation = 1.0 - (self.membership ** 2).sum(1)
        self.topics_ = []
        for ci, comm in enumerate(self.communities):
            idx = [i for i in np.argsort(-self.word_score[:, ci])[: self.topn_words] if self.word_score[i, ci] > 0]
            words = [(self.id2word[i], float(self.word_score[i, ci])) for i in idx]
            self.topics_.append({"id": ci, "label": " / ".join(w for w, _ in words[:3]),
                                 "words": words, "n_words": len(comm)})

    def doc_topic_matrix(self, docs: list[list[int]]) -> np.ndarray:
        M = np.zeros((len(docs), len(self.communities)))
        for i, d in enumerate(docs):
            if d:
                M[i] = self.membership[d].sum(0)
        return M / np.maximum(M.sum(1, keepdims=True), 1e-12)

    def transform(self, texts: list[str]) -> np.ndarray:
        return self.doc_topic_matrix(self.pre.transform(texts))

    def top_words(self, n: int | None = None) -> list[list[str]]:
        n = n or self.topn_words
        return [[w for w, _ in t["words"][:n]] for t in self.topics_]

    def bridge_words(self, n: int = 10):
        """Words whose edges are spread over several communities (cross-topic terms)."""
        out = []
        for i in np.argsort(-self.participation):
            if i in self.graph and self.graph.degree(i) >= 5 and self.membership[i].sum() > 0:
                top2 = [int(t) for t in np.argsort(-self.membership[i])[:2]]
                out.append((self.id2word[i], float(self.participation[i]), top2))
            if len(out) >= n:
                break
        return out

    def summary(self) -> dict:
        return {"n_nodes": self.graph.number_of_nodes(), "n_edges": self.graph.number_of_edges(),
                "n_topics": len(self.communities), "modularity": round(self.modularity, 4),
                "density": round(nx.density(self.graph), 5),
                "timing": {k: round(v, 3) for k, v in self.timing.items()}}
