# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NLP/Data Visualisation TP on US presidential inaugural speeches. The goal is to analyze themes, tonality, and sentiments across speeches using Python.

**Dataset**: [Kaggle – Presidential Address](https://www.kaggle.com/datasets/adhok93/presidentialaddress)  
Place the downloaded CSV in `data/`.

## Project Steps

1. **Data Acquisition** – load the Kaggle CSV into a Pandas DataFrame
2. **Preprocessing** – clean text, tokenize, apply lemmatization/stemming (NLTK or spaCy)
3. **Sentiment & Tonality** – use TextBlob or VADER to score each speech
4. **Word Frequency & TF-IDF** – compute term frequencies and TF-IDF per speech
5. **Visualisation** – sentiment charts, word clouds, TF-IDF bar charts
6. **Interpretation** – write insights about recurring themes and tonality shifts over time

## Stack

- Python 3.x (scripts `.py`, no Jupyter)
- `pandas`, `nltk`, `textblob` or `vaderSentiment`
- `scikit-learn` (TF-IDF via `TfidfVectorizer`)
- `matplotlib`, `seaborn`, `wordcloud`

## Install

```bash
pip install pandas nltk textblob vaderSentiment scikit-learn matplotlib seaborn wordcloud
```

## Run

```bash
python main.py
```

## File Layout

```
TPNote/
├── data/        # CSV from Kaggle goes here
├── consigne.md  # Project brief
└── main.py      # Entry point (to be created)
```
