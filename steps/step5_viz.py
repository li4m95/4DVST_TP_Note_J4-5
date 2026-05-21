import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from wordcloud import WordCloud

sns.set_theme(style="whitegrid", palette="muted")




def plot_wordcloud(freq_df: pd.DataFrame, president: str) -> None:

    subset = freq_df[freq_df["president"].str.contains(president)]
    if subset.empty:
        print(f"President '{president}' introuvable.")
        return

    word_counts = subset.groupby("word")["count"].sum().to_dict()

    wc = WordCloud(
        width=900, height=500,
        background_color="white",
        max_words=80,
        colormap="Blues",
    ).generate_from_frequencies(word_counts)

    plt.figure(figsize=(11, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Mots les plus frequents — {president}", fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(f"wordcloud_{president.split()[-1].lower()}.png", dpi=150)
    plt.show()



def plot_sentiment_timeline(df: pd.DataFrame) -> None:
    data = df.dropna(subset=["year"]).sort_values("year")

    # On n'annote que les vrais outliers : bottom 3 + top 1 par compound
    n_outliers = 3
    to_annotate = pd.concat([
        data.nsmallest(n_outliers, "vader_compound"),
        data.nlargest(1, "vader_compound"),
    ]).drop_duplicates(subset=["president", "year"])

    plt.figure(figsize=(14, 5))
    ax = sns.lineplot(data=data, x="year", y="vader_compound", marker="o", linewidth=1.5)
    ax.axhline(0, color="red", linewidth=0.8, linestyle="--", label="neutre (0)")

    for _, row in to_annotate.iterrows():
        name = row["president"].split()[-1]
        # Décale vers le haut si positif, vers le bas si négatif → évite le chevauchement
        offset_y = 12 if row["vader_compound"] >= 0 else -18
        ax.annotate(
            f"{name} ({int(row['year'])})",
            xy=(row["year"], row["vader_compound"]),
            xytext=(0, offset_y), textcoords="offset points",
            fontsize=8, ha="center",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
        )

    ax.set_title("Evolution du sentiment des discours d'investiture (1789-2017)", fontsize=13)
    ax.set_xlabel("Annee")
    ax.set_ylabel("Score VADER compound")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.legend()
    plt.tight_layout()
    plt.savefig("sentiment_timeline.png", dpi=150)
    plt.show()



def plot_tfidf_comparison(tfidf_df: pd.DataFrame, presidents: list[str]) -> None:
    subsets = []
    for president in presidents:
        s = tfidf_df[tfidf_df["president"].str.contains(president)]
        if not s.empty:
            latest_year = s["year"].max()
            subsets.append(s[s["year"] == latest_year].nlargest(10, "tfidf"))

    if not subsets:
        return

    n = len(subsets)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    colors = sns.color_palette("muted", n)

    for ax, subset, color in zip(axes, subsets, colors):
        subset = subset.sort_values("tfidf")
        president = subset["president"].iloc[0]
        year = int(subset["year"].iloc[0]) if not pd.isna(subset["year"].iloc[0]) else ""

        sns.barplot(data=subset, y="word", x="tfidf", ax=ax, color=color)
        ax.set_title(f"{president}\n({year})", fontsize=11)
        ax.set_xlabel("Score TF-IDF")
        ax.set_ylabel("")

    fig.suptitle("Mots les plus distinctifs par president (TF-IDF)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("tfidf_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
