from steps.step1_load import load_data, describe_data
from steps.step2_preprocess import preprocess
from steps.step3_sentiment import analyze_sentiment, print_sentiment_summary
from steps.step4_tfidf import compute_word_freq, compute_tfidf, print_tfidf_summary
from steps.step5_viz import plot_wordcloud, plot_sentiment_timeline, plot_tfidf_comparison

if __name__ == "__main__":
    df = load_data()
    describe_data(df)

    print("\n--- Etape 2 : Pretraitement ---")
    df = preprocess(df)

    print("\n--- Etape 3 : Analyse de sentiment ---")
    df = analyze_sentiment(df)
    print_sentiment_summary(df)

    print("\n--- Etape 4 : Frequences + TF-IDF ---")
    freq_df  = compute_word_freq(df)
    tfidf_df = compute_tfidf(df, top_n=10)
    print_tfidf_summary(tfidf_df, ["Lincoln", "Obama", "Trump"])

    print("\n--- Etape 5 : Visualisations ---")
    plot_wordcloud(freq_df, "Lincoln")
    plot_sentiment_timeline(df)
    plot_tfidf_comparison(tfidf_df, ["Lincoln", "Obama", "Trump"])
