"""Text classifier that estimates whether a transcript reflects depression.

The classifier trains on two labelled corpora:

* ``data/depressed.txt``     – free-text posts written by people describing
  feelings of depression. Split into sentences, one document each.
* ``data/not_depressed.txt`` – happy-moment entries from the HappyDB dataset,
  stored as CSV. Only the cleaned moment text (the ``cleaned_hm`` column) is
  used; the surrounding metadata is discarded.

Each document is cleaned, stop-word filtered and Porter-stemmed, then vectorized
with TF-IDF or counts over unigrams/bigrams and classified with one of several
scikit-learn models. The two classes are balanced before training so the model
is not biased toward the larger class.

Example
-------
    python -m src.classifier --model svm --vectorizer tfidf --ngram unigram \
        --predict output.txt
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEPRESSED_LABEL = "depressed"
NOT_DEPRESSED_LABEL = "not_depressed"
RANDOM_SEED = 42

# Index of the cleaned happy-moment text in the HappyDB CSV rows.
HAPPYDB_TEXT_COLUMN = 4

MODELS = {
    "svm": lambda: SVC(kernel="linear"),
    "multinomial_nb": MultinomialNB,
    "gaussian_nb": GaussianNB,
    "random_forest": lambda: RandomForestClassifier(random_state=RANDOM_SEED),
    "mlp": lambda: MLPClassifier(
        solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 2),
        max_iter=1000, random_state=RANDOM_SEED,
    ),
}

# Minimal stop-word list used only if the NLTK corpus cannot be downloaded
# (e.g. offline / CI). The full NLTK list is preferred when available.
_FALLBACK_STOPWORDS = {
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
    "them", "a", "an", "the", "and", "or", "but", "if", "of", "at", "by",
    "for", "with", "to", "from", "in", "on", "is", "am", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "this",
    "that", "these", "those", "as", "so", "than", "too", "very", "can", "will",
    "just", "not", "no", "nor", "s", "t", "don", "now",
}

# Resolved once per process (which tokenizer to use + which stop-words).
_resources: Tuple[bool, set] | None = None


def _resolve_resources() -> Tuple[bool, set]:
    """Decide once whether NLTK data is usable; download it if possible."""
    use_nltk_tokenizer = False
    try:
        nltk.data.find("tokenizers/punkt")
        use_nltk_tokenizer = True
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
            nltk.data.find("tokenizers/punkt")
            use_nltk_tokenizer = True
        except Exception:
            use_nltk_tokenizer = False

    try:
        from nltk.corpus import stopwords
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)
        stop_words = set(stopwords.words("english"))
    except Exception:
        stop_words = set(_FALLBACK_STOPWORDS)

    return use_nltk_tokenizer, stop_words


def _get_resources() -> Tuple[bool, set]:
    global _resources
    if _resources is None:
        _resources = _resolve_resources()
    return _resources


def _tokenize(text: str, use_nltk: bool) -> List[str]:
    if use_nltk:
        return nltk.word_tokenize(text)
    return re.findall(r"[a-z]+", text)


@dataclass
class Preprocessor:
    """Cleans and normalizes raw text into a list of stemmed tokens."""

    stemmer: object = field(default=None)

    def __post_init__(self) -> None:
        from nltk.stem import PorterStemmer
        self.stemmer = self.stemmer or PorterStemmer()
        self._use_nltk, self._stop_words = _get_resources()

    def tokens(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[‘’“”]", "", text)          # smart quotes
        text = re.sub(r"[0-9]", "", text)            # digits
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        return [
            self.stemmer.stem(tok)
            for tok in _tokenize(text, self._use_nltk)
            if tok and tok not in self._stop_words
        ]

    def clean(self, text: str) -> str:
        """Return a normalized, space-joined string ready for vectorizing."""
        return " ".join(self.tokens(text))


def read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def split_sentences(text: str) -> List[str]:
    """Split prose into sentences (regex-based, no external data needed)."""
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def load_depressed_docs() -> List[str]:
    """Depressed corpus: free text -> one document per sentence."""
    return split_sentences(read_text(DATA_DIR / "depressed.txt"))


def load_not_depressed_docs() -> List[str]:
    """Not-depressed corpus: HappyDB CSV -> one document per moment text."""
    docs: List[str] = []
    with open(DATA_DIR / "not_depressed.txt", encoding="utf-8", errors="ignore") as fh:
        for row in csv.reader(fh):
            if len(row) > HAPPYDB_TEXT_COLUMN:
                moment = row[HAPPYDB_TEXT_COLUMN].strip()
                if moment:
                    docs.append(moment)
    return docs


def build_vectorizer(kind: str, ngram: str):
    ngram_range = (1, 1) if ngram == "unigram" else (1, 2)
    common = dict(ngram_range=ngram_range, min_df=2)
    if kind == "tfidf":
        return TfidfVectorizer(**common)
    if kind == "count":
        return CountVectorizer(**common)
    raise ValueError(f"Unknown vectorizer: {kind}")


def load_dataset(preprocess: Preprocessor, balance: bool = True):
    """Return (documents, labels), with classes balanced by down-sampling."""
    by_label: Dict[str, List[str]] = {DEPRESSED_LABEL: [], NOT_DEPRESSED_LABEL: []}
    for raw_docs, label in (
        (load_depressed_docs(), DEPRESSED_LABEL),
        (load_not_depressed_docs(), NOT_DEPRESSED_LABEL),
    ):
        by_label[label] = [c for c in (preprocess.clean(d) for d in raw_docs) if c]

    if balance:
        smallest = min(len(docs) for docs in by_label.values())
        rng = random.Random(RANDOM_SEED)
        by_label = {label: rng.sample(docs, smallest) for label, docs in by_label.items()}

    documents: List[str] = []
    labels: List[str] = []
    for label, docs in by_label.items():
        documents.extend(docs)
        labels.extend([label] * len(docs))
    return documents, labels


def train(model_name: str, vectorizer_kind: str, ngram: str, test_size: float = 0.2):
    preprocess = Preprocessor()
    documents, labels = load_dataset(preprocess)

    vectorizer = build_vectorizer(vectorizer_kind, ngram)
    features = vectorizer.fit_transform(documents)
    if model_name == "gaussian_nb":
        features = features.toarray()

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=RANDOM_SEED, stratify=labels
    )

    model = MODELS[model_name]()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions, zero_division=0))

    return model, vectorizer, preprocess


def predict_transcript(path: Path, model, vectorizer, preprocess) -> str:
    """Classify a transcript by majority vote over its sentences."""
    sentences = split_sentences(read_text(path)) or [read_text(path)]
    cleaned = [preprocess.clean(s) for s in sentences]
    cleaned = [c for c in cleaned if c]
    if not cleaned:
        return NOT_DEPRESSED_LABEL

    features = vectorizer.transform(cleaned)
    if isinstance(model, GaussianNB):
        features = features.toarray()

    predictions = list(model.predict(features))
    depressed = predictions.count(DEPRESSED_LABEL)
    not_depressed = predictions.count(NOT_DEPRESSED_LABEL)
    # On a tie, prefer the less alarming label.
    return DEPRESSED_LABEL if depressed > not_depressed else NOT_DEPRESSED_LABEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="svm")
    parser.add_argument("--vectorizer", choices=("tfidf", "count"), default="tfidf")
    parser.add_argument("--ngram", choices=("unigram", "bigram"), default="unigram")
    parser.add_argument(
        "--predict",
        type=Path,
        help="Path to a transcript (e.g. output.txt) to classify after training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, vectorizer, preprocess = train(args.model, args.vectorizer, args.ngram)

    if args.predict:
        label = predict_transcript(args.predict, model, vectorizer, preprocess)
        verdict = (
            "shows signs of depression"
            if label == DEPRESSED_LABEL
            else "does not show signs of depression"
        )
        print(f"\nThe transcript {verdict}.")


if __name__ == "__main__":
    main()
