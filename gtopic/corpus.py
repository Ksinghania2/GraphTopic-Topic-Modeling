"""Corpus loading + a labelled synthetic news-style corpus generator.

The generator exists so that the whole pipeline can be run and *quantitatively*
evaluated (ground-truth topic labels) without downloading anything. Any real
corpus can be plugged in through `load_corpus` (CSV / folder of .txt files) or
`load_20newsgroups` (needs internet the first time)."""
from __future__ import annotations
import csv, os, random, re
from pathlib import Path
import numpy as np

# --------------------------------------------------------------------- topics
TOPICS = {
 "Space & Astronomy": dict(
  n="telescope galaxy orbit satellite rocket astronaut planet comet asteroid nebula mission spacecraft lunar mars probe launch cosmos telescope observatory eclipse meteor crater supernova quasar payload thruster module station universe radiation signal exoplanet gravity light power energy".split(),
  v="orbits observes detects launches explores maps discovers tracks photographs measures approaches".split(),
  a="distant lunar solar stellar orbital massive bright ancient cosmic interplanetary".split(),
  p=["black hole", "solar system", "space station", "dark matter", "milky way", "space agency", "mission control", "red planet"]),
 "Medicine & Health": dict(
  n="patient doctor hospital vaccine disease therapy clinical treatment symptom virus infection diagnosis surgery drug medicine immune cell tumor cancer heart nurse dose trial recovery antibody pathogen outbreak clinic cure study growth model".split(),
  v="treats diagnoses prevents cures infects heals prescribes screens reduces monitors".split(),
  a="clinical chronic acute viral medical severe cardiac infectious therapeutic genetic".split(),
  p=["blood pressure", "clinical trial", "immune system", "heart disease", "public health", "side effect", "health care", "cancer cell"]),
 "Sports": dict(
  n="match team player coach tournament league goal stadium season champion score trophy captain wicket batsman bowler referee fixture final striker defender medal athlete record training fans penalty innings squad power energy".split(),
  v="wins defeats scores chases leads dominates qualifies trains beats clinches".split(),
  a="decisive unbeaten professional national veteran dominant thrilling defensive aggressive youthful".split(),
  p=["world cup", "grand slam", "home ground", "man of the match", "premier league", "test series", "golden boot", "title race"]),
 "Finance & Markets": dict(
  n="stock market investor bank inflation interest rate shares profit revenue trading bond portfolio equity dividend economy fund loan credit currency earnings analyst recession asset growth record policy model".split(),
  v="invests trades borrows lends earns forecasts raises hedges rallies declines".split(),
  a="fiscal financial volatile bullish bearish monetary quarterly global corporate speculative".split(),
  p=["stock market", "interest rate", "central bank", "hedge fund", "balance sheet", "venture capital", "price index", "bond yield"]),
 "AI & Computing": dict(
  n="algorithm neural network software dataset model training processor cloud server chip robot language code startup platform developer cybersecurity encryption database compute automation sensor device launch cell data benchmark".split(),
  v="trains computes predicts deploys optimises learns encodes classifies generates processes".split(),
  a="digital autonomous scalable open-source intelligent distributed predictive secure efficient cloud-based".split(),
  p=["machine learning", "neural network", "deep learning", "data center", "open source", "large language", "computer vision", "source code"]),
 "Climate & Environment": dict(
  n="climate emission carbon forest ocean glacier biodiversity renewable solar wind pollution drought flood species ecosystem temperature warming coal habitat wildlife energy power policy study model season".split(),
  v="melts pollutes warms threatens restores conserves absorbs emits protects flooded".split(),
  a="renewable sustainable arctic coastal tropical ecological fossil global endangered toxic".split(),
  p=["global warming", "climate change", "carbon emission", "sea level", "fossil fuel", "rain forest", "greenhouse gas", "clean energy"]),
 "Food & Cooking": dict(
  n="recipe chef kitchen flavour spice ingredient dish restaurant bakery sauce vegetable dessert cuisine menu grill pastry soup noodle cheese herb dinner breakfast sugar oven roast season market".split(),
  v="bakes simmers roasts seasons serves stirs grills whisks tastes chops".split(),
  a="crispy savoury fresh spicy delicious tender traditional homemade creamy aromatic".split(),
  p=["olive oil", "street food", "fine dining", "comfort food", "soy sauce", "farm table", "tasting menu", "fresh herb"]),
 "Politics & Government": dict(
  n="election parliament minister government senate vote party campaign president law policy diplomat treaty coalition reform cabinet voter democracy constitution bill opposition mayor legislation sanction network power".split(),
  v="elects passes debates vetoes legislates campaigns appoints negotiates ratifies governs".split(),
  a="political federal electoral democratic constitutional bipartisan diplomatic national regional parliamentary".split(),
  p=["prime minister", "foreign policy", "supreme court", "general election", "national security", "vote bank", "opposition leader", "peace talks"]),
 "Education & Universities": dict(
  n="student university professor curriculum lecture degree campus scholarship exam school teacher classroom thesis research course tuition graduate faculty library syllabus semester admission training study network".split(),
  v="teaches graduates enrols studies lectures assesses mentors publishes examines learns".split(),
  a="academic undergraduate collegiate scholarly vocational compulsory rigorous interdisciplinary doctoral online".split(),
  p=["higher education", "entrance exam", "research paper", "teaching staff", "study abroad", "primary school", "online course", "student loan"]),
 "Travel & Tourism": dict(
  n="tourist hotel airline beach itinerary destination passport flight resort museum island cruise visa luggage hostel backpacker monument sightseeing airport guide journey landmark season market train".split(),
  v="visits explores books travels departs lands wanders hikes tours cruises".split(),
  a="scenic coastal historic popular remote luxurious bustling exotic cultural picturesque".split(),
  p=["world heritage", "boarding pass", "tourist season", "city centre", "travel guide", "package tour", "national park", "hot spring"]),
}
LABELS = list(TOPICS)

_GENERIC = ["officials", "analysts", "experts", "residents", "observers", "critics", "reporters", "workers", "members", "groups"]
_FILLER = ["on monday", "last week", "this month", "in recent months", "earlier this decade", "on thursday", "over the weekend"]
_TPL = [
 "The {a} {n1} {v} the {n2} near the {n3}.",
 "{G} said the {n1} and the {n2} could change how {n3} works.",
 "A {a} {p} {v2} attention {f}.",
 "In a {a} move, the {n1} {v} a {a2} {n2} {f}.",
 "Experts link the {p} to a rise in {n1} and {n2}.",
 "{F}, the {n1} {v} the {p} despite the {a} {n2}.",
 "The {n1} of the {p} is {a} and the {n2} remains {a2}.",
 "{G} expect the {n1} to {vb} the {n2} before the {n3}.",
 "Every {n1} depends on a {a} {n2}, and the {p} proves it.",
 "Not every {n1} is {a}, but the {n2} {v} a {p}.",
]


def _zipf(rng, items, k=1.0):
    w = 1.0 / (np.arange(1, len(items) + 1) ** k)
    return rng.choices(items, weights=w, k=1)[0]


def generate_corpus(n_docs: int = 12000, seed: int = 7, noise: float = 0.18, mix: float = 0.4):
    """Return (texts, labels). Each document has a primary topic (label), optionally
    a secondary topic (prob `mix`), generic filler sentences and `noise` probability
    that a content word is drawn from a random other topic. Polysemous words such
    as model/energy/market/season/power appear in several topics."""
    rng = random.Random(seed)
    texts, labels = [], []
    for _ in range(n_docs):
        primary = rng.randrange(len(LABELS))
        secondary = rng.randrange(len(LABELS)) if rng.random() < mix else None
        sents = []
        for _s in range(rng.randint(6, 11)):
            r = rng.random()
            t = primary if (secondary is None or r < 0.78) else secondary
            if rng.random() < 0.08:
                t = rng.randrange(len(LABELS))
            def pick(kind, base=t):
                tt = rng.randrange(len(LABELS)) if (kind == "n" and rng.random() < noise) else base
                return _zipf(rng, TOPICS[LABELS[tt]][kind])
            tpl = rng.choice(_TPL)
            s = tpl.format(a=pick("a"), a2=pick("a"), n1=pick("n"), n2=pick("n"), n3=pick("n"), v=pick("v"),
                           v2=pick("v").rstrip("s") + "s" if False else pick("v"), vb=pick("v").rstrip("s"),
                           p=_zipf(rng, TOPICS[LABELS[t]]["p"], 0.6), G=rng.choice(_GENERIC).capitalize(),
                           F=rng.choice(_FILLER).capitalize(), f=rng.choice(_FILLER))
            sents.append(re.sub(r"\b([Aa]) (?=[aeiouAEIOU])", r"\1n ", s))
        texts.append(" ".join(sents)); labels.append(primary)
    return texts, labels


# -------------------------------------------------------------------- loaders
def save_csv(path, texts, labels=None):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "label", "text"])
        for i, t in enumerate(texts):
            w.writerow([i, "" if labels is None else labels[i], t])


def load_corpus(path: str, text_col: str = "text", label_col: str = "label"):
    """Load a CSV (with a text column) or a directory / .txt file. Returns (texts, labels|None)."""
    p = Path(path)
    if p.is_dir():
        files = sorted(p.rglob("*.txt"))
        return [f.read_text(encoding="utf-8", errors="ignore") for f in files], None
    if p.suffix.lower() == ".csv":
        texts, labels = [], []
        with open(p, newline="", encoding="utf-8", errors="ignore") as f:
            for row in csv.DictReader(f):
                texts.append(row[text_col]); labels.append(row.get(label_col, ""))
        has = any(l != "" for l in labels)
        if has:
            uniq = {l: i for i, l in enumerate(sorted(set(labels)))}
            labels = [uniq[l] for l in labels]
        return texts, (labels if has else None)
    text = p.read_text(encoding="utf-8", errors="ignore")
    docs = [d.strip() for d in text.split("\n\n") if d.strip()]
    return docs, None


def load_20newsgroups(categories=None):
    """Optional real-world corpus (needs internet on first call)."""
    from sklearn.datasets import fetch_20newsgroups
    d = fetch_20newsgroups(subset="all", categories=categories, remove=("headers", "footers", "quotes"))
    keep = [i for i, t in enumerate(d.data) if len(t.split()) > 20]
    return [d.data[i] for i in keep], [int(d.target[i]) for i in keep]
