"""Text pre-processing: tokenisation, stop-word removal, light lemmatisation,
statistical phrase (bigram) detection and vocabulary pruning."""
from __future__ import annotations
import re
from collections import Counter
from math import log

STOPWORDS = set("""
a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having
he her here hers him his how i if in into is it its itself just me more most my no nor not now of off on
once only or other our out over own same she should so some such than that the their theirs them then there
these they this those through to too under until up very was we were what when where which while who whom
why will with would you your yours said says say one two three four five new first last many much may might
must shall since yet still even per via according however among around across along within without upon
whether either neither another every often like get got make made makes making take took taken use used using
well back way ever never always already really quite rather less least year years day days time times
thing things people also today week month report reports
""".split())

_KEEP = {"series", "species", "news", "analysis", "diagnosis", "basis", "physics", "politics", "economics", "mathematics"}
_TOKEN = re.compile(r"[a-z][a-z'\-]{2,}")


def lemmatize(w: str) -> str:
    """Very small rule-based lemmatiser (plural forms)."""
    if len(w) <= 4 or w in _KEEP:
        return w
    if w.endswith("ies") and len(w) > 5:
        return w[:-3] + "y"
    if w.endswith(("sses", "shes", "ches", "xes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def tokenize(text: str) -> list[str | None]:
    """Lower-case tokens; stop-words are returned as None placeholders so that
    phrase detection never joins words across a removed stop-word."""
    out: list[str | None] = []
    for t in _TOKEN.findall(text.lower()):
        t = t.strip("-'")
        if t in STOPWORDS or len(t) < 3:
            out.append(None)
        else:
            out.append(lemmatize(t))
    return out


class Preprocessor:
    """Fits a vocabulary on a corpus and converts documents to id lists."""

    def __init__(self, min_df: int = 5, max_df_ratio: float = 0.5, max_vocab: int = 3000,
                 detect_phrases: bool = True, phrase_min_count: int = 15, phrase_threshold: float = 0.5):
        self.min_df, self.max_df_ratio, self.max_vocab = min_df, max_df_ratio, max_vocab
        self.detect_phrases = detect_phrases
        self.phrase_min_count, self.phrase_threshold = phrase_min_count, phrase_threshold
        self.phrases: set[tuple[str, str]] = set()
        self.vocab: dict[str, int] = {}

    def _learn_phrases(self, toks: list[list]):
        uni, bi, total = Counter(), Counter(), 0
        for d in toks:
            for i, w in enumerate(d):
                if w is None:
                    continue
                uni[w] += 1; total += 1
                if i + 1 < len(d) and d[i + 1] is not None:
                    bi[(w, d[i + 1])] += 1
        nb = sum(bi.values()) or 1
        for (a, b), c in bi.items():
            if c < self.phrase_min_count:
                continue
            pxy = c / nb
            pmi = log(pxy / ((uni[a] / total) * (uni[b] / total)))
            if pmi / -log(pxy) >= self.phrase_threshold:
                self.phrases.add((a, b))

    def _apply_phrases(self, d: list) -> list[str]:
        out, i = [], 0
        while i < len(d):
            w = d[i]
            if w is None:
                i += 1; continue
            if i + 1 < len(d) and d[i + 1] is not None and (w, d[i + 1]) in self.phrases:
                out.append(f"{w}_{d[i + 1]}"); i += 2
            else:
                out.append(w); i += 1
        return out

    def _tokens(self, text: str) -> list[str]:
        d = tokenize(text)
        return self._apply_phrases(d) if self.detect_phrases else [w for w in d if w]

    def fit(self, texts: list[str]) -> "Preprocessor":
        toks = [tokenize(t) for t in texts]
        if self.detect_phrases:
            self._learn_phrases(toks)
        docs = [self._apply_phrases(d) if self.detect_phrases else [w for w in d if w] for d in toks]
        df = Counter(w for d in docs for w in set(d))
        tf = Counter(w for d in docs for w in d)
        n = len(docs)
        keep = [w for w, c in df.items() if c >= self.min_df and c / n <= self.max_df_ratio]
        keep.sort(key=lambda w: (-tf[w], w))
        self.vocab = {w: i for i, w in enumerate(keep[: self.max_vocab])}
        return self

    def transform(self, texts: list[str]) -> list[list[int]]:
        return [[self.vocab[w] for w in self._tokens(t) if w in self.vocab] for t in texts]

    @property
    def id2word(self) -> list[str]:
        return [w for w, _ in sorted(self.vocab.items(), key=lambda x: x[1])]
