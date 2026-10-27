"""Tune GraphTopic resolution on validation data and report test-set results."""
from __future__ import annotations

import csv
import json
import os
import time

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

from gtopic import GraphTopicModel, Preprocessor
from gtopic.corpus import load_20newsgroups
from gtopic.evaluate import doc_term_binary, evaluate
from run_newsgroups_experiments import _hardware_and_versions

RESOLUTIONS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
SEED = 42


def _fit_graph(texts, resolution):
    preprocessor = Preprocessor(min_df=5, max_df_ratio=0.5, max_vocab=3000)
    preprocessor.fit(texts)
    docs = preprocessor.transform(texts)
    model = GraphTopicModel(resolution=resolution, seed=SEED, preprocessor=preprocessor)
    model.id2word = preprocessor.id2word
    model.fit_ids(docs)
    return model, docs


def _score(model, docs, labels, class_names):
    word_ids = {word: index for index, word in enumerate(model.id2word)}
    topics = [[word_ids[word] for word in words] for words in model.top_words()]
    predicted = model.doc_topic_matrix(docs).argmax(axis=1)
    binary = doc_term_binary(docs, len(model.id2word))
    return evaluate(topics, binary, labels, predicted, class_names)


def main():
    texts, labels = load_20newsgroups()
    labels = np.asarray(labels)
    class_names = list(fetch_20newsgroups(
        subset="all", remove=("headers", "footers", "quotes")
    ).target_names)
    all_indices = np.arange(len(texts))
    train_val_indices, test_indices = train_test_split(
        all_indices, test_size=0.2, random_state=SEED, stratify=labels
    )
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=0.25, random_state=SEED, stratify=labels[train_val_indices]
    )
    train_texts = [texts[index] for index in train_indices]
    val_texts = [texts[index] for index in val_indices]
    test_texts = [texts[index] for index in test_indices]

    validation = []
    for resolution in RESOLUTIONS:
        started = time.perf_counter()
        model, _ = _fit_graph(train_texts, resolution)
        val_docs = model.pre.transform(val_texts)
        scores = _score(model, val_docs, labels[val_indices], class_names)
        validation.append({
            "resolution": resolution,
            "n_topics": len(model.communities),
            "modularity": float(model.modularity),
            "fit_seconds": time.perf_counter() - started,
            "npmi_coherence": scores["npmi_coherence"],
            "nmi": scores["nmi"],
            "ari": scores["ari"],
            "purity": scores["purity"],
        })
        print(f"VALIDATION gamma={resolution:g} k={len(model.communities)} "
              f"NMI={scores['nmi']:.4f} ARI={scores['ari']:.4f} "
              f"purity={scores['purity']:.4f}", flush=True)

    selected = max(validation, key=lambda row: row["nmi"])
    combined_indices = np.concatenate((train_indices, val_indices))
    combined_texts = [texts[index] for index in combined_indices]
    test_scores = {}
    for name, resolution in (("default_gamma_1", 1.0), ("validation_selected", selected["resolution"])):
        model, _ = _fit_graph(combined_texts, resolution)
        test_docs = model.pre.transform(test_texts)
        scores = _score(model, test_docs, labels[test_indices], class_names)
        test_scores[name] = {
            "resolution": resolution,
            "n_topics": len(model.communities),
            "modularity": float(model.modularity),
            "npmi_coherence": scores["npmi_coherence"],
            "nmi": scores["nmi"],
            "ari": scores["ari"],
            "purity": scores["purity"],
            "topic_class_mapping": scores["topic_class_mapping"],
        }
        print(f"TEST {name} gamma={resolution:g} k={len(model.communities)} "
              f"NMI={scores['nmi']:.4f} ARI={scores['ari']:.4f} "
              f"purity={scores['purity']:.4f}", flush=True)

    result = {
        "dataset": "20 Newsgroups, headers/footers/quotes removed, documents >20 words",
        "split": {
            "train_documents": len(train_indices),
            "validation_documents": len(val_indices),
            "test_documents": len(test_indices),
            "seed": SEED,
            "stratified": True,
        },
        "preprocessing": {
            "min_df": 5,
            "max_df_ratio": 0.5,
            "max_vocab": 3000,
            "phrase_min_count": 15,
            "npmi_phrase_threshold": 0.5,
        },
        "selection_metric": "validation NMI",
        "resolutions": list(RESOLUTIONS),
        "selected_resolution": selected["resolution"],
        "validation_sweep": validation,
        "test_results": test_scores,
        "software_hardware": _hardware_and_versions(),
    }

    os.makedirs("results", exist_ok=True)
    with open("results/newsgroups_gamma_validation.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    with open("results/newsgroups_gamma_validation.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(validation[0]))
        writer.writeheader()
        writer.writerows(validation)
    print(f"Selected gamma={selected['resolution']:g} on validation; saved validation and test results.", flush=True)


if __name__ == "__main__":
    main()
