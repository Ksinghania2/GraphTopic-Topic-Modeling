#!/usr/bin/env python
"""Command-line interface.

Examples
  python run_pipeline.py --demo                         # built-in labelled news corpus
  python run_pipeline.py --input my_docs.csv --text-col body
  python run_pipeline.py --input ./folder_of_txt_files --resolution 1.2
  python run_pipeline.py --newsgroups                   # 20 Newsgroups (needs internet)
"""
import argparse, csv, json, os
import numpy as np
from gtopic import GraphTopicModel, Preprocessor
from gtopic.corpus import load_corpus, generate_corpus, load_20newsgroups, LABELS
from gtopic.evaluate import doc_term_binary, evaluate
from gtopic.visualize import network_figure, topic_bars


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input"); src.add_argument("--demo", action="store_true"); src.add_argument("--newsgroups", action="store_true")
    ap.add_argument("--text-col", default="text"); ap.add_argument("--label-col", default="label")
    ap.add_argument("--window", type=int, default=8); ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--method", default="louvain", choices=["louvain", "greedy_modularity", "label_propagation"])
    ap.add_argument("--topn", type=int, default=10); ap.add_argument("--max-vocab", type=int, default=3000)
    ap.add_argument("--min-df", type=int, default=5); ap.add_argument("--out", default="results/cli_output")
    a = ap.parse_args()

    if a.demo:
        texts, y = generate_corpus(12000); y = np.array(y)
    elif a.newsgroups:
        texts, y = load_20newsgroups(); y = np.array(y)
    else:
        texts, y = load_corpus(a.input, a.text_col, a.label_col); y = None if y is None else np.array(y)
    print(f"Loaded {len(texts)} documents")

    pre = Preprocessor(min_df=a.min_df, max_vocab=a.max_vocab)
    m = GraphTopicModel(window=a.window, resolution=a.resolution, method=a.method, topn_words=a.topn, preprocessor=pre).fit(texts)
    print("Graph:", m.summary())
    docs = m.pre.transform(texts); P = m.doc_topic_matrix(docs); pred = P.argmax(1)
    ids = {w: i for i, w in enumerate(m.id2word)}
    ev = evaluate([[ids[w] for w in ws] for ws in m.top_words()], doc_term_binary(docs, len(m.id2word)), y, pred if y is not None else None)
    print(f"NPMI coherence={ev['npmi_coherence']:.3f}  diversity={ev['diversity']:.2f}" +
          (f"  NMI={ev['nmi']:.3f}  purity={ev['purity']:.3f}" if y is not None else ""))
    for t in m.topics_:
        print(f"\nTopic {t['id']:>2}  ({(pred == t['id']).sum()} docs, {t['n_words']} words)\n   " + ", ".join(w.replace('_', ' ') for w, _ in t["words"]))
    os.makedirs(a.out, exist_ok=True)
    json.dump({"summary": m.summary(), "metrics": {k: v for k, v in ev.items() if k != "per_topic_npmi"}, "topics": m.topics_},
              open(f"{a.out}/topics.json", "w"), indent=1)
    with open(f"{a.out}/doc_topics.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["doc_id", "topic", "confidence"])
        for i, (k, p) in enumerate(zip(pred, P.max(1))): w.writerow([i, int(k), round(float(p), 4)])
    network_figure(m).savefig(f"{a.out}/network.png", bbox_inches="tight", dpi=150)
    topic_bars(m).savefig(f"{a.out}/topic_words.png", bbox_inches="tight", dpi=150)
    print(f"\nSaved topics.json, doc_topics.csv, network.png, topic_words.png to {a.out}/")


if __name__ == "__main__":
    main()
