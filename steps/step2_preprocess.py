import re
import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

for resource in ["punkt", "punkt_tab", "stopwords", "wordnet",
                 "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def _get_wordnet_pos(treebank_tag: str) -> str:
    """Convertit le tag POS de NLTK (ex: 'VBG') en format WordNet pour la lemmatisation."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN  # par défaut : nom


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_and_lemmatize(text: str) -> list[str]:
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)  

    return [
        LEMMATIZER.lemmatize(word, _get_wordnet_pos(tag))
        for word, tag in tagged
        if word not in STOP_WORDS and len(word) > 2
    ]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_clean"] = df["text"].apply(clean_text)
    df["tokens"] = df["text_clean"].apply(tokenize_and_lemmatize)
    # Version "texte" des tokens pour TF-IDF (scikit-learn attend des strings)
    df["tokens_str"] = df["tokens"].apply(lambda t: " ".join(t))
    return df
