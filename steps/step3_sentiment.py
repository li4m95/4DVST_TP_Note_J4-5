import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ANALYZER = SentimentIntensityAnalyzer()


def _classify(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    scores = df["text_clean"].apply(lambda t: ANALYZER.polarity_scores(t))
    print(f"\nExemple de scores VADER :\n{scores.iloc[0]}")

    df["vader_neg"]      = scores.apply(lambda s: s["neg"])
    df["vader_neu"]      = scores.apply(lambda s: s["neu"])
    df["vader_pos"]      = scores.apply(lambda s: s["pos"])
    df["vader_compound"] = scores.apply(lambda s: s["compound"])
    df["sentiment"]      = df["vader_compound"].apply(_classify)

    return df


def print_sentiment_summary(df: pd.DataFrame) -> None:
    cols = ["president", "year", "vader_compound", "sentiment"]
    summary = df[cols].sort_values("vader_compound", ascending=False)

    print("\nTop 5 discours les plus POSITIFS :")
    print(summary.head(5).to_string(index=False))

    print("\nTop 5 discours les plus NEGATIFS :")
    print(summary.tail(5).to_string(index=False))

    print("\nRepartition globale des sentiments :")
    print(df["sentiment"].value_counts().to_string())
