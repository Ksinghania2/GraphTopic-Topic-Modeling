# GraphTopic – Graph-based Topic Extraction from Large Text Corpora

Mini Project · B.Tech Computer & Communication Engineering · Manipal University Jaipur

GraphTopic turns a text corpus into a **word co-occurrence network** (nodes = words, edges = NPMI of
sliding-window co-occurrence) and uses **Louvain community detection** to find groups of words that
belong together. Every community is a **topic**; documents get topic distributions from word memberships.

## Install
```bash
pip install -r requirements.txt          # numpy scipy networkx scikit-learn matplotlib Flask
```

## Run
| What | Command |
|---|---|
| Web app (front page, upload/paste, live inference) | `python app.py`  →  http://127.0.0.1:5000 |
| CLI on built-in labelled demo corpus | `python run_pipeline.py --demo` |
| CLI on your own CSV | `python run_pipeline.py --input docs.csv --text-col body` |
| CLI on a folder of `.txt` files | `python run_pipeline.py --input ./texts` |
| CLI on 20 Newsgroups (needs internet) | `python run_pipeline.py --newsgroups` |
| Reproduce every table/figure in the report | `python run_experiments.py && python make_figures.py && python make_diagrams.py` |
| Unit tests | `python -m pytest tests` |

Useful flags: `--window 8` (co-occurrence window), `--resolution 1.0` (higher → more, finer topics),
`--method louvain|greedy_modularity|label_propagation`, `--topn 10`, `--max-vocab 3000`.

## Python API
```python
from gtopic import GraphTopicModel
m = GraphTopicModel(window=8, resolution=1.0).fit(list_of_strings)
for t in m.topics_: print(t["id"], [w for w, _ in t["words"]])
m.transform(["new unseen document"])      # topic distribution
m.bridge_words()                          # words shared across topics
```

## Layout
```
gtopic/preprocess.py   tokenise, stop-words, lemmatise, NPMI phrase detection, vocabulary
gtopic/graph.py        vectorised sliding-window co-occurrence + NPMI graph with top-k pruning
gtopic/topics.py       GraphTopicModel (Louvain / LPA / greedy), word scoring, doc inference
gtopic/evaluate.py     NPMI coherence, diversity, NMI, ARI, purity
gtopic/baselines.py    LDA and NMF (scikit-learn) for comparison
gtopic/corpus.py       loaders + labelled synthetic news corpus generator
gtopic/visualize.py    network / bar / heat-map figures
app.py, templates/     Flask web interface           run_pipeline.py   command-line tool
run_experiments.py     all experiments + figures     data/news_corpus.csv   12,000-doc demo corpus
```

## Note on the demo data
`data/news_corpus.csv` is a **synthetically generated** news-style corpus (10 subjects, shared
polysemous words, cross-topic noise, mixed-topic documents). It exists so the system can be run and
scored against ground-truth labels offline. Results on it show the method works as designed; for
claims about real-world performance run `--newsgroups` or your own corpus.
