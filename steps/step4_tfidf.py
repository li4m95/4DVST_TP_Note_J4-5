import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


def compute_word_freq(df: pd.DataFrame) -> pd.DataFrame:

    rows = []
    for _, row in df.iterrows():
        counter = Counter(row["tokens"])
        for word, count in counter.items():
            rows.append({
                "president": row["president"],
                "year": row["year"],
                "word": word,
                "count": count,
            })
    return pd.DataFrame(rows)


def compute_tfidf(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    vectorizer = TfidfVectorizer(max_features=5000)
    matrix = vectorizer.fit_transform(df["tokens_str"])  
    vocab = vectorizer.get_feature_names_out()          

    # Convertit la matrice sparse en DataFrame lisible
    tfidf_df = pd.DataFrame(matrix.toarray(), columns=vocab)
    tfidf_df.index = df.index
    print(f"\nDataFrame TF-IDF (extrait) :\n{tfidf_df.iloc[0, :10]}")

    rows = []
    for idx, row_series in tfidf_df.iterrows():
        president = df.loc[idx, "president"]
        year      = df.loc[idx, "year"]

        # top_n mots avec le score TF-IDF le plus élevé pour ce discours
        top_words = row_series.nlargest(top_n)
        for word, score in top_words.items():
            rows.append({
                "president": president,
                "year":      year,
                "word":      word,
                "tfidf":     round(score, 4),
            })

    return pd.DataFrame(rows)


def print_tfidf_summary(tfidf_df: pd.DataFrame, presidents: list[str]) -> None:
    """Affiche les top mots TF-IDF pour une liste de présidents donnée."""
    for president in presidents:
        subset = tfidf_df[tfidf_df["president"].str.contains(president)]
        if subset.empty:
            continue
        print(f"\n{president} :")
        for _, r in subset.iterrows():
            print(f"  {r['word']:<20} {r['tfidf']}")
