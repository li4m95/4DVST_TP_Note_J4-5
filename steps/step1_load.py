import pandas as pd


def load_data(path: str = "data/inaug_speeches.csv") -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, encoding="latin1")
    df.columns = ["president", "address", "date", "text"]

    # Convertir la date en datetime (le format varie, errors='coerce' gère les exceptions)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year

    # Supprimer les lignes sans texte
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].str.strip()
    df = df[df["text"] != ""]

    return df.reset_index(drop=True)


def describe_data(df: pd.DataFrame) -> None:
    print(f"Nombre de discours : {len(df)}")
    print(f"Présidents uniques : {df['president'].nunique()}")
    print(f"Periode couverte   : {df['year'].min()} - {df['year'].max()}")
    print(f"\nAperçu :")
    print(df[["president", "address", "year"]].to_string(index=False))
