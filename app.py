"""Flask web interface.   python app.py   ->  http://127.0.0.1:5000"""
import csv
import io
import math
import os
import threading
import uuid

import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from gtopic import GraphTopicModel, Preprocessor
from gtopic.corpus import load_corpus
from gtopic.evaluate import doc_term_binary, evaluate
from gtopic.visualize import network_figure, topic_bars, to_b64

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
MIN_DOCUMENTS = 20
MAX_DOCUMENTS = 12_000
SAMPLE_DEMO_DOCUMENTS = 2_000
SAMPLE = os.path.join(os.path.dirname(__file__), "data", "news_corpus.csv")
MODELS: dict[str, GraphTopicModel] = {}          # token -> fitted model (in-memory cache)
SAMPLE_ANALYSES: dict[tuple[int, float, int], tuple[GraphTopicModel, dict, list[dict]]] = {}
SAMPLE_ANALYSIS_LOCK = threading.Lock()


def parse_parameters(values):
    def integer(name, default):
        value = values.get(name, default)
        if isinstance(value, bool):
            raise ValueError
        parsed = int(value)
        if isinstance(value, float) and value != parsed:
            raise ValueError
        return parsed

    def decimal(name, default):
        value = values.get(name, default)
        if isinstance(value, bool):
            raise ValueError
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError
        return parsed

    try:
        window = integer("window", 8)
        resolution = decimal("resolution", 1.0)
        topn = integer("topn", 10)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Parameters must be valid numbers.") from exc

    if not 2 <= window <= 40:
        raise ValueError("Window must be between 2 and 40.")
    if not 0.1 <= resolution <= 5:
        raise ValueError("Resolution must be between 0.1 and 5.")
    if not 5 <= topn <= 20:
        raise ValueError("Top words must be between 5 and 20.")
    return window, resolution, topn


def input_error(message):
    return render_template("index.html", sample_exists=os.path.exists(SAMPLE), error=message), 400


def parse_upload(f, text_col="text"):
    raw = f.read().decode("utf-8", errors="ignore")
    if f.filename.lower().endswith(".csv"):
        rows = list(csv.DictReader(io.StringIO(raw)))
        if text_col not in (rows[0] if rows else {}):
            text_col = next(iter(rows[0]), "text") if rows else "text"
        return [r[text_col] for r in rows if r.get(text_col)]
    return split_docs(raw)


def split_docs(raw):
    docs = [d.strip() for d in raw.split("\n\n") if len(d.split()) >= 5]
    if len(docs) < 20:                                   # fall back to one document per line
        docs = [d.strip() for d in raw.splitlines() if len(d.split()) >= 5]
    return docs


def fit_model(texts, window, resolution, topn):
    small = len(texts) < 500
    pre = Preprocessor(min_df=2 if small else 5, phrase_min_count=5 if small else 15)
    m = GraphTopicModel(window=window, resolution=resolution, topn_words=topn, preprocessor=pre,
                        min_count=2 if small else 3, min_topic_size=5 if small else 10).fit(texts)
    return m


def sample_demo_subset(texts, labels, limit=SAMPLE_DEMO_DOCUMENTS):
    if len(texts) <= limit:
        return texts, labels

    if labels is None:
        indices = np.linspace(0, len(texts) - 1, num=limit, dtype=int)
    else:
        label_array = np.asarray(labels)
        classes = np.unique(label_array)
        per_class = max(1, limit // len(classes))
        indices = np.sort(np.concatenate([
            np.flatnonzero(label_array == label)[:per_class]
            for label in classes
        ]))[:limit]

    selected_texts = [texts[int(index)] for index in indices]
    selected_labels = [labels[int(index)] for index in indices] if labels is not None else None
    return selected_texts, selected_labels


@app.route("/")
def index():
    return render_template("index.html", sample_exists=os.path.exists(SAMPLE))


@app.route("/analyze", methods=["POST"])
def analyze():
    f = request.form
    try:
        window, res, topn = parse_parameters(f)
    except ValueError as exc:
        return input_error(str(exc))

    labels = None
    is_sample = f.get("source") == "sample"
    sample_total = None
    if is_sample:
        texts, labels = load_corpus(SAMPLE)
        sample_total = len(texts)
        texts, labels = sample_demo_subset(texts, labels)
    elif request.files.get("file") and request.files["file"].filename:
        texts = parse_upload(request.files["file"], f.get("text_col", "text") or "text")
    else:
        texts = split_docs(f.get("text", ""))

    if len(texts) > MAX_DOCUMENTS:
        return input_error(f"Please provide no more than {MAX_DOCUMENTS:,} documents per analysis.")
    if len(texts) < MIN_DOCUMENTS:
        return input_error("Please provide at least 20 documents (separate documents with a blank line).")

    def compute_analysis():
        model = fit_model(texts, window, res, topn)
        if not model.communities:
            return None
        docs = model.pre.transform(texts)
        predictions = model.doc_topic_matrix(docs).argmax(1)
        ids = {word: index for index, word in enumerate(model.id2word)}
        metrics = evaluate(
            [[ids[word] for word in words] for words in model.top_words()],
            doc_term_binary(docs, len(model.id2word)),
            np.array(labels) if labels else None,
            predictions if labels else None,
        )
        topic_results = [
            {
                **topic,
                "docs": int((predictions == topic["id"]).sum()),
                "npmi": round(metrics["per_topic_npmi"][topic["id"]], 3),
            }
            for topic in model.topics_
        ]
        return model, metrics, topic_results

    if is_sample:
        cache_key = (window, res, topn)
        with SAMPLE_ANALYSIS_LOCK:
            analysis = SAMPLE_ANALYSES.get(cache_key)
            if analysis is None:
                analysis = compute_analysis()
                if analysis is not None:
                    SAMPLE_ANALYSES[cache_key] = analysis
    else:
        analysis = compute_analysis()

    if analysis is None:
        return input_error("No topics could be extracted - try a larger corpus or a lower resolution.")

    m, ev, topics = analysis
    tokens = uuid.uuid4().hex[:10]; MODELS[tokens] = m
    if len(MODELS) > 5: MODELS.pop(next(iter(MODELS)))
    return render_template("results.html", token=tokens, n_docs=len(texts), s=m.summary(), ev=ev, topics=topics,
                           has_labels=labels is not None, is_sample=is_sample, sample_total=sample_total,
                           bridge=m.bridge_words(8))


@app.route("/visualizations/<token>")
def visualizations(token):
    model = MODELS.get(token)
    if model is None:
        return jsonify(error="Session expired - re-run the analysis."), 404
    return jsonify(
        network=to_b64(network_figure(model, ncol=3, fontsize=8, width_in=12)),
        bars=to_b64(topic_bars(model, ncols=4)),
    )


@app.route("/infer/<token>", methods=["POST"])
def infer(token):
    m = MODELS.get(token)
    if m is None:
        return jsonify(error="Session expired - re-run the analysis."), 404
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get("text"), str):
        return jsonify(error="Send a JSON object with a text string."), 400
    text = body["text"]
    if len(m.pre.transform([text])[0]) == 0:
        return jsonify(error="No known vocabulary words in this text."), 400
    p = m.transform([text])[0]
    order = np.argsort(-p)[:5]
    return jsonify(topics=[{"id": int(i), "label": m.topics_[i]["label"].replace("_", " "), "p": round(float(p[i]), 3)} for i in order])


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error="Send a valid JSON object."), 400
    texts = body.get("documents")
    if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
        return jsonify(error="'documents' must be a list of strings."), 400
    if len(texts) > MAX_DOCUMENTS:
        return jsonify(error=f"Send no more than {MAX_DOCUMENTS:,} documents."), 400
    texts = [text.strip() for text in texts if text.strip()]
    if len(texts) < MIN_DOCUMENTS:
        return jsonify(error=f"Send at least {MIN_DOCUMENTS} non-empty documents."), 400
    try:
        window, resolution, topn = parse_parameters(body)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    m = fit_model(texts, window, resolution, topn)
    if not m.communities:
        return jsonify(error="No topics could be extracted from these documents."), 400
    return jsonify(summary=m.summary(), topics=[{"id": t["id"], "words": [w for w, _ in t["words"]]} for t in m.topics_])


@app.errorhandler(RequestEntityTooLarge)
def request_too_large(_error):
    message = "Request is too large (maximum 25 MB)."
    if request.path.startswith(("/api/", "/infer/")):
        return jsonify(error=message), 413
    return render_template("index.html", sample_exists=os.path.exists(SAMPLE), error=message), 413


if __name__ == "__main__":
    from waitress import serve

    serve(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")))
