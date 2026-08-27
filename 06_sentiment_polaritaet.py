"""
Rating in negativ/neutral/positiv umgewandelt und nochmal mit
Logistic Regression trainiert.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import os

DATEN_ORDNER = "data/processed"
ERGEBNIS_ORDNER = "results"
os.makedirs(ERGEBNIS_ORDNER, exist_ok=True)

train_df = pd.read_csv(os.path.join(DATEN_ORDNER, "train.csv"))
test_df = pd.read_csv(os.path.join(DATEN_ORDNER, "test.csv"))

train_df["text_normalisiert"] = train_df["text_normalisiert"].fillna("")
test_df["text_normalisiert"] = test_df["text_normalisiert"].fillna("")


def rating_zu_polarität(rating: int) -> str:
    """Bildet eine fünfstufige Bewertung auf eine dreistufige Polarität ab."""
    if rating in (1, 2):
        return "negativ"
    if rating == 3:
        return "neutral"
    return "positiv"


train_df["polarität"] = train_df["rating"].apply(rating_zu_polarität)
test_df["polarität"] = test_df["rating"].apply(rating_zu_polarität)

klassenreihenfolge = ["negativ", "neutral", "positiv"]

vektorisierer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
)
X_train = vektorisierer.fit_transform(train_df["text_normalisiert"])
X_test = vektorisierer.transform(test_df["text_normalisiert"])

y_train = train_df["polarität"]
y_test = test_df["polarität"]

modell = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)
modell.fit(X_train, y_train)

vorhersagen = modell.predict(X_test)

accuracy = accuracy_score(y_test, vorhersagen)
konfusionsmatrix = confusion_matrix(y_test, vorhersagen, labels=klassenreihenfolge)
bericht = classification_report(y_test, vorhersagen, labels=klassenreihenfolge)

print(f"Accuracy Sentimentpolarität (3 Klassen): {accuracy:.4f}")
print("\nKlassifikationsbericht:")
print(bericht)
print("Konfusionsmatrix (Zeilen: tatsächliche Klasse, Spalten: Vorhersage):")
print(konfusionsmatrix)

with open(os.path.join(ERGEBNIS_ORDNER, "klassifikationsbericht_polaritaet.txt"), "w") as datei:
    datei.write(f"Accuracy: {accuracy:.4f}\n\n")
    datei.write(bericht)

anzeige = ConfusionMatrixDisplay(
    confusion_matrix=konfusionsmatrix,
    display_labels=klassenreihenfolge,
)
fig, ax = plt.subplots(figsize=(6, 3))
anzeige.plot(ax=ax, cmap="Purples", colorbar=False)
ax.set_aspect("auto")
ax.set_xlabel("Vorhergesagte Polaritaet")
ax.set_ylabel("Tatsaechliche Polaritaet")
ax.set_title("Konfusionsmatrix: Sentimentpolaritaet")
plt.tight_layout()
plt.savefig(os.path.join(ERGEBNIS_ORDNER, "konfusionsmatrix_polaritaet.png"), dpi=200)
print(f"\nKonfusionsmatrix gespeichert unter {ERGEBNIS_ORDNER}/konfusionsmatrix_polaritaet.png")
