# GraphTopic – Graph-based Topic Extraction from Large Text Corpora

Mini Project · B.Tech Computer & Communication Engineering · Manipal University Jaipur

GraphTopic turns a text corpus into a **word co-occurrence network** (nodes = words, edges = NPMI of
sliding-window co-occurrence) and uses **Louvain community detection** to find groups of words that
belong together. Every community is a **topic**; documents get topic distributions from word memberships.

## Install
```bash
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
```

## Run
| What | Command |
|---|---|
| Web app (Waitress server, front page, upload/paste, live inference) | `python app.py`  →  http://127.0.0.1:5000 |
| CLI on built-in labelled demo corpus | `python run_pipeline.py --demo` |
| CLI on your own CSV | `python run_pipeline.py --input docs.csv --text-col body` |
| CLI on a folder of `.txt` files | `python run_pipeline.py --input ./texts` |
| CLI on 20 Newsgroups (needs internet) | `python run_pipeline.py --newsgroups` |
| Repeated 20 Newsgroups evaluation and LDA/NMF baselines | `python run_newsgroups_experiments.py` |
| Tune Louvain resolution on validation, score on test | `python run_newsgroups_gamma_validation.py` |
| Five-run 2,500 / 5,000 document timing check | `python run_experiments.py --timing-only` |
| Harder synthetic overlap/noise experiment only | `python run_experiments.py --hard-synthetic-only` |
| Reproduce every table/figure in the report | `python run_experiments.py && python make_figures.py && python make_diagrams.py` |
| Unit tests | `python -m pytest tests` |

Useful flags: `--window 8` (co-occurrence window), `--resolution 1.0` (higher → more, finer topics),
`--method louvain|greedy_modularity|label_propagation`, `--topn 10`, `--max-vocab 3000`.

## Web deployment notes
The app uses Waitress rather than Flask's development server. It listens on `127.0.0.1:5000`
by default. Configure `HOST` and `PORT` with environment variables if needed; for example, in
PowerShell:

```powershell
$env:HOST = "127.0.0.1"
$env:PORT = "8000"
python app.py
```

Uploaded and pasted input is limited to 25 MB and 12,000 documents per analysis. The web app
is still a single-instance research tool: fitted models and inference sessions are kept in
process memory, and analysis is CPU-intensive. Before exposing it to untrusted users, put it
behind an HTTPS reverse proxy with authentication and rate limiting, and monitor its memory
and CPU use. Do not bind to `0.0.0.0` on an untrusted network without those protections.

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
run_newsgroups_experiments.py  real-data evaluation and baseline runner
```

## Note on the demo data
`data/news_corpus.csv` is a **synthetically generated** news-style corpus (10 subjects, shared
polysemous words, cross-topic noise, mixed-topic documents). It exists so the system can be run and
scored against ground-truth labels offline. Results on it show the method works as designed; for
claims about real-world performance run `--newsgroups` or your own corpus.

The real-data runner uses all 20 Newsgroups categories after removing headers,
footers, and quoted text. It applies the fixed 3,000-word cap, 5-document minimum
frequency, and 50% maximum document frequency; it does not tune on the evaluation
corpus. It measures GraphTopic across five Louvain seeds and LDA/NMF across three
seeds, at both the seed-42 graph topic count and 20 topics. Results, per-topic
dominant-class shares, the fitted vocabulary, run summaries, and environment metadata are written to
`results/newsgroups_results.json` and `results/newsgroups_*.csv`. The timing-only
command does one warm-up and five measured runs for each method and corpus size.
The separate gamma-validation runner makes stratified train/validation/test
splits, selects resolution by validation NMI, then reports the selected and
default resolutions on the untouched test split.
`generate_corpus` accepts an explicit `seed` (default 7) for repeatable synthetic
corpora.

`requirements.txt` pins the direct runtime dependencies for the recorded
environment. `python -m pip install -r requirements.txt` installs those versions;
the editable install adds the project package and development extras.
