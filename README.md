# Depression Detector

A two-part natural language processing project that holds a short, empathetic
conversation with a user and then classifies the resulting transcript as
**depressed** or **not depressed** using classical machine-learning models.

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Built with scikit-learn](https://img.shields.io/badge/built%20with-scikit--learn-orange.svg)](https://scikit-learn.org/)

> ⚠️ **Disclaimer:** This is an academic NLP project, **not** a medical device.
> It cannot diagnose depression or any other condition and must not be used as a
> substitute for professional care. If you or someone you know is struggling,
> please reach out to a qualified mental-health professional or a local helpline.

---

## How it works

The system runs in two stages:

1. **Reflective chatbot** (`src/chatbot.py`) — An ELIZA-style bot, inspired by
   Weizenbaum's original, that uses regular-expression pattern matching to ask
   open-ended, empathetic questions and reflect the user's statements back to
   them. Once the user has shared enough (≥ 31 words) and ends the session, the
   full transcript is written to `output.txt`.

2. **Text classifier** (`src/classifier.py`) — The transcript is cleaned,
   tokenized, stop-word filtered, and Porter-stemmed, then vectorized
   (TF-IDF or counts, over unigrams or bigrams) and scored by one of five
   scikit-learn classifiers. A majority vote over the predicted tokens yields
   the final label.

```
 ┌──────────────┐     output.txt      ┌────────────────────┐
 │   chatbot.py │ ──────────────────► │   classifier.py    │
 │  (gather)    │   transcript        │  (preprocess →     │
 └──────────────┘                     │   vectorize →      │
                                       │   classify → vote) │
                                       └────────────────────┘
```

## Project structure

```
depression-detector/
├── src/
│   ├── chatbot.py       # ELIZA-style reflective conversation agent
│   ├── classifier.py    # preprocessing + ML classification pipeline
│   └── __init__.py
├── data/
│   ├── depressed.txt        # labelled corpus (depressed)
│   └── not_depressed.txt    # labelled corpus (not depressed)
├── requirements.txt
├── LICENSE
└── README.md
```

## Getting started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

The first run downloads two small NLTK resources (`punkt`, `stopwords`)
automatically.

### 2. Have a conversation

```bash
python -m src.chatbot --output output.txt
```

Chat with the bot until you've shared enough, then type `good-bye` to save the
transcript.

### 3. Classify the transcript

```bash
python -m src.classifier --model svm --vectorizer tfidf --ngram unigram --predict output.txt
```

## Command-line options

`classifier.py`:

| Option         | Choices                                                          | Default   |
| -------------- | ---------------------------------------------------------------- | --------- |
| `--model`      | `svm`, `multinomial_nb`, `gaussian_nb`, `random_forest`, `mlp`   | `svm`     |
| `--vectorizer` | `tfidf`, `count`                                                 | `tfidf`   |
| `--ngram`      | `unigram`, `bigram`                                              | `unigram` |
| `--predict`    | Path to a transcript to classify after training                  | *(none)*  |

`chatbot.py`:

| Option      | Description                              | Default      |
| ----------- | ---------------------------------------- | ------------ |
| `--output`  | Where to save the conversation transcript | `output.txt` |

## Results

The depressed and not-depressed corpora come in different formats (free text
vs. HappyDB CSV), so the loader parses each correctly — extracting only the
moment text from the CSV — and the two classes are **balanced** before training
to avoid bias toward the larger class. Trained on the balanced set with an
80/20 stratified split:

| Model           | Accuracy |
| --------------- | -------- |
| Multinomial NB  | ~0.87    |
| MLP (neural net)| ~0.82    |
| Linear SVM      | ~0.80    |
| Random Forest   | ~0.80    |

Exact numbers vary with the random split and on whether the full NLTK
tokenizer or the lightweight offline fallback is used. The dataset is small, so
these figures should be read as a comparison of approaches rather than a
benchmark.

## Tech stack

Python · NLTK · scikit-learn · NumPy

## License

Released under the [MIT License](LICENSE).
