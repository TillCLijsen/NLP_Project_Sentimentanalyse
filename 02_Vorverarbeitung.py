"""
Führt die 3 CSVs zusammen, bereinigt den Text (lowercase, Stopwords
raus, lemmatisieren) und splittet in Train/Test (80/20).
"""


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import os

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

KATEGORIEN = ["Video_Games", "Automotive", "Pet_Supplies"]
RAW_ORDNER = "data/raw"
OUTPUT_ORDNER = "data/processed"
os.makedirs(OUTPUT_ORDNER, exist_ok=True)

stoppwörter = set(stopwords.words("english"))
lemmatisierer = WordNetLemmatizer()


def text_normalisieren(text: str, stoppwörter=stoppwörter, lemmatisierer=lemmatisierer) -> str:
    """
        Macht den Text sauber: lowercase, Sonderzeichen weg, tokenisieren,
        Stopwords raus, lemmatisieren.
        """
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [
        lemmatisierer.lemmatize(token)
        for token in tokens
        if token not in stoppwörter and len(token) > 1
    ]
    return " ".join(tokens)


# Zusammenführung der Kategorien
teil_dataframes = []
for kategorie in KATEGORIEN:
    pfad = os.path.join(RAW_ORDNER, f"{kategorie}.csv")
    teil_dataframes.append(pd.read_csv(pfad))

df = pd.concat(teil_dataframes, ignore_index=True)
df = df.dropna(subset=["rating"])
df["rating"] = df["rating"].astype(int)

# Verkettung von Titel und Beschreibungstext als Eingabetext
df["titel"] = df["titel"].fillna("")
df["text"] = df["text"].fillna("")
df["rezensionstext"] = df["titel"] + " " + df["text"]

print("Normalisierung des Rezensionstexts läuft...")
df["text_normalisiert"] = df["rezensionstext"].apply(text_normalisieren)

# Entfernung von Rezensionen, die nach der Normalisierung leer sind
df = df[df["text_normalisiert"].str.strip().astype(bool)]

# Stratifizierte Trainings-/Test-Aufteilung
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["rating"],
)

train_df.to_csv(os.path.join(OUTPUT_ORDNER, "train.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_ORDNER, "test.csv"), index=False)

print(f"\nGesamtdatensatz: {len(df)} Rezensionen")
print(f"Trainingsdatensatz: {len(train_df)} Rezensionen")
print(f"Testdatensatz: {len(test_df)} Rezensionen")

print("\nBewertungsverteilung Trainingsdatensatz:")
print(train_df["rating"].value_counts().sort_index())

print("\nBewertungsverteilung Testdatensatz:")
print(test_df["rating"].value_counts().sort_index())

print("\nBewertungsverteilung Trainingsdatensatz nach Kategorie:")
print(train_df.groupby(["kategorie", "rating"]).size().unstack(fill_value=0))
