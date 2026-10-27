import numpy as np
from types import SimpleNamespace
from gtopic import GraphTopicModel
from gtopic.corpus import generate_corpus, load_20newsgroups
from gtopic.evaluate import evaluate, purity, topic_diversity
from gtopic.graph import cooccurrence_matrix


def test_cooccurrence_symmetric_and_window():
    m = cooccurrence_matrix([[0, 1, 2], [2, 3]], 4, window=2)     # only adjacent pairs
    assert (m != m.T).nnz == 0 and m[0, 1] == 1 and m[0, 2] == 0 and m[2, 3] == 1


def test_end_to_end_recovers_topics():
    texts, y = generate_corpus(2500, seed=1)
    m = GraphTopicModel().fit(texts)
    assert 8 <= len(m.communities) <= 12 and m.modularity > 0.5
    pred = m.doc_topic_matrix(m.pre.transform(texts)).argmax(1)
    assert purity(np.array(y), pred) > 0.85
    assert topic_diversity(m.top_words()) > 0.9


def test_inference_distribution_sums_to_one():
    texts, _ = generate_corpus(1500, seed=2)
    m = GraphTopicModel().fit(texts)
    p = m.transform(["the telescope observed a distant galaxy near the black hole"])[0]
    assert abs(p.sum() - 1) < 1e-6


def test_evaluate_reports_topic_dominant_class_and_share():
    result = evaluate(
        [[0], [1]], np.zeros((5, 2)), np.array([1, 1, 2, 2, 3]),
        np.array([0, 0, 1, 1, 1]), ["zero", "one", "two", "three"],
    )
    assert result["topic_class_mapping"] == [
        {"topic": 0, "dominant_class": "one", "share": 1.0, "documents": 2},
        {"topic": 1, "dominant_class": "two", "share": 2 / 3, "documents": 3},
    ]


def test_20newsgroups_loader_removes_metadata_and_quotes(monkeypatch):
    from sklearn import datasets

    calls = []
    document = " ".join(["body"] * 21)

    def fake_fetch(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=[document], target=np.array([4]))

    monkeypatch.setattr(datasets, "fetch_20newsgroups", fake_fetch)
    texts, labels = load_20newsgroups()
    assert calls == [{"subset": "all", "remove": ("headers", "footers", "quotes")}]
    assert texts == [document]
    assert labels == [4]
