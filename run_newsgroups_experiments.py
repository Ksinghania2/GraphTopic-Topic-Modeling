"""Run fixed-configuration 20 Newsgroups experiments and save auditable results."""
from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
import networkx as nx
import numpy as np
import sklearn
from sklearn.datasets import fetch_20newsgroups

from gtopic import GraphTopicModel, Preprocessor
from gtopic.baselines import run_lda, run_nmf
from gtopic.corpus import load_20newsgroups
from gtopic.evaluate import doc_term_binary, evaluate

OUT = "results"
GRAPH_SEEDS = (1, 2, 3, 42, 99)
BASELINE_SEEDS = (1, 2, 3)
METRICS = ("npmi_coherence", "diversity", "nmi", "ari", "purity", "seconds")


def _hardware_and_versions() -> dict:
    import scipy

    memory = None
    processor = platform.processor()
    if sys.platform == "win32":
        import ctypes
        import winreg

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong),
                ("avail_phys", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("avail_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("avail_virtual", ctypes.c_ulonglong),
                ("avail_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            memory = int(status.total_phys)
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            processor = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return {
        "platform": platform.platform(),
        "processor": processor,
        "logical_cpu_count": os.cpu_count(),
        "ram_bytes": memory,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "networkx": nx.__version__,
        "scikit_learn": sklearn.__version__,
    }


def _score(model, docs, labels, class_names, binary):
    word_ids = {word: index for index, word in enumerate(model.id2word)}
    topics = [[word_ids[word] for word in words] for words in model.top_words()]
    predicted = model.doc_topic_matrix(docs).argmax(axis=1)
    return evaluate(topics, binary, labels, predicted, class_names)


def _summarize(rows):
    groups = {}
    for row in rows:
        setting = row.get("k_setting", "seed-dependent")
        key = (row["method"], setting, None if row["method"] == "GraphTopic" else row["k"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (method, setting, k), runs in groups.items():
        record = {"method": method, "k_setting": setting, "k": k, "runs": len(runs)}
        if method == "GraphTopic":
            found_topics = [run["k"] for run in runs]
            record["found_k_mean"] = float(np.mean(found_topics))
            record["found_k_std"] = float(np.std(found_topics))
            for metric in ("modularity", "n_nodes", "n_edges"):
                values = [float(run[metric]) for run in runs]
                record[f"{metric}_mean"] = float(np.mean(values))
                record[f"{metric}_std"] = float(np.std(values))
        for metric in METRICS:
            values = [float(run[metric]) for run in runs]
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_std"] = float(np.std(values))
        summary.append(record)
    return summary


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Loading 20 Newsgroups (headers, footers, and quotes removed)...", flush=True)
    texts, labels = load_20newsgroups()
    labels = np.asarray(labels)
    class_names = list(fetch_20newsgroups(
        subset="all", remove=("headers", "footers", "quotes")
    ).target_names)

    preprocessor = Preprocessor(min_df=5, max_df_ratio=0.5, max_vocab=3000)
    preprocessor.fit(texts)
    docs = preprocessor.transform(texts)
    vocabulary_size = len(preprocessor.id2word)
    _write_csv(
        os.path.join(OUT, "newsgroups_vocabulary.csv"),
        [{"word_id": index, "term": word} for index, word in enumerate(preprocessor.id2word)],
    )
    binary = doc_term_binary(docs, vocabulary_size)
    print(f"Documents={len(docs)} vocabulary={vocabulary_size} classes={len(class_names)}", flush=True)

    metadata = {
        "dataset": "20 Newsgroups, all categories; documents with <=20 whitespace-separated words excluded",
        "n_documents": len(texts),
        "class_counts": {class_names[index]: int(count) for index, count in enumerate(
            np.bincount(labels, minlength=len(class_names))
        )},
        "vocabulary_size": vocabulary_size,
        "phrase_count": len(preprocessor.phrases),
        "preprocessing": {
            "min_document_frequency": 5,
            "max_document_frequency_ratio": 0.5,
            "max_vocabulary": 3000,
            "phrase_min_count": preprocessor.phrase_min_count,
            "npmi_phrase_threshold": preprocessor.phrase_threshold,
            "window": 8,
            "resolution": 1.0,
            "tuning": "No 20 Newsgroups tuning performed; fixed defaults evaluated on the full corpus.",
        },
        "graph_seeds": list(GRAPH_SEEDS),
        "baseline_seeds": list(BASELINE_SEEDS),
        "software_hardware": _hardware_and_versions(),
    }

    graph_runs, baseline_runs, mapping_rows, topic_details = [], [], [], []
    graph_reference_k = None
    print("Warming up GraphTopic...", flush=True)
    warmup_model = GraphTopicModel(seed=0, preprocessor=preprocessor)
    warmup_model.id2word = preprocessor.id2word
    warmup_model.fit_ids(docs)
    for seed in GRAPH_SEEDS:
        model = GraphTopicModel(seed=seed, preprocessor=preprocessor)
        model.id2word = preprocessor.id2word
        start = time.perf_counter()
        model.fit_ids(docs)
        elapsed = time.perf_counter() - start
        scores = _score(model, docs, labels, class_names, binary)
        if seed == 42:
            graph_reference_k = len(model.communities)
        record = {
            "method": "GraphTopic",
            "seed": seed,
            "k": len(model.communities),
            "modularity": float(model.modularity),
            "n_nodes": model.graph.number_of_nodes(),
            "n_edges": model.graph.number_of_edges(),
            "seconds": elapsed,
            **{metric: scores[metric] for metric in METRICS[:-1]},
        }
        graph_runs.append(record)
        for mapped in scores["topic_class_mapping"]:
            mapping_rows.append({"method": "GraphTopic", "seed": seed, "k": len(model.communities), **mapped})
        for topic_index, (topic, npmi) in enumerate(zip(model.topics_, scores["per_topic_npmi"])):
            topic_details.append({
                "method": "GraphTopic", "seed": seed, "k": len(model.communities),
                "topic": topic_index, "npmi": npmi,
                "top_words": " | ".join(word for word, _ in topic["words"]),
            })
        print(f"GraphTopic seed={seed} k={record['k']} modularity={record['modularity']:.4f} "
              f"graph={record['n_nodes']} nodes/{record['n_edges']} edges NMI={record['nmi']:.4f} "
              f"ARI={record['ari']:.4f} purity={record['purity']:.4f} time={elapsed:.2f}s", flush=True)

    assert graph_reference_k is not None
    metadata["graph_reference_k_seed_42"] = graph_reference_k
    for k_setting, k in (("graph_discovered_k", graph_reference_k), ("true_20_classes", 20)):
        for method, runner in (("LDA", run_lda), ("NMF", run_nmf)):
            print(f"Warming up {method} k={k}...", flush=True)
            runner(docs, vocabulary_size, k, seed=0)
            for seed in BASELINE_SEEDS:
                result = runner(docs, vocabulary_size, k, seed=seed)
                run_scores = evaluate(
                    result["topics"], binary, labels, result["pred"], class_names
                )
                record = {
                    "method": method,
                    "k_setting": k_setting,
                    "seed": seed,
                    "k": k,
                    "seconds": result["time"],
                    **{metric: run_scores[metric] for metric in METRICS[:-1]},
                }
                baseline_runs.append(record)
                for mapped in run_scores["topic_class_mapping"]:
                    mapping_rows.append({"method": method, "seed": seed, "k_setting": k_setting,
                                         "k": k, **mapped})
                for topic_index, (word_ids, npmi) in enumerate(
                    zip(result["topics"], run_scores["per_topic_npmi"])
                ):
                    topic_details.append({
                        "method": method, "seed": seed, "k_setting": k_setting, "k": k,
                        "topic": topic_index, "npmi": npmi,
                        "top_words": " | ".join(preprocessor.id2word[word_id] for word_id in word_ids),
                    })
                print(f"{method} k={k} seed={seed} NMI={record['nmi']:.4f} "
                      f"ARI={record['ari']:.4f} purity={record['purity']:.4f} "
                      f"time={result['time']:.2f}s", flush=True)

    runs = graph_runs + baseline_runs
    summary = _summarize(runs)
    for item in summary:
        topic_count = (f"found_k={item['found_k_mean']:.2f}±{item['found_k_std']:.2f}"
                       if item["method"] == "GraphTopic"
                       else f"k={item['k']} setting={item['k_setting']}")
        print(f"SUMMARY {item['method']} {topic_count} "
              f"NMI={item['nmi_mean']:.4f}±{item['nmi_std']:.4f} "
              f"ARI={item['ari_mean']:.4f}±{item['ari_std']:.4f} "
              f"purity={item['purity_mean']:.4f}±{item['purity_std']:.4f} "
              f"time={item['seconds_mean']:.2f}±{item['seconds_std']:.2f}s", flush=True)

    payload = {
        "metadata": metadata,
        "graph_seed_stability": graph_runs,
        "baseline_runs": baseline_runs,
        "summary": summary,
        "topic_class_mapping": mapping_rows,
        "topic_details": topic_details,
    }
    with open(os.path.join(OUT, "newsgroups_results.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    _write_csv(os.path.join(OUT, "newsgroups_runs.csv"), runs)
    _write_csv(os.path.join(OUT, "newsgroups_summary.csv"), summary)
    _write_csv(os.path.join(OUT, "newsgroups_topic_class_mapping.csv"), mapping_rows)
    _write_csv(os.path.join(OUT, "newsgroups_topic_words.csv"), topic_details)
    print("Saved results/newsgroups_results.json and CSV reports.", flush=True)


if __name__ == "__main__":
    main()
