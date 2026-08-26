"""
Naive Bayes mit TF-IDF trainieren und auf dem Testset auswerten
(Accuracy + Konfusionsmatrix).
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
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

# Fehlende normalisierte Texte (z. B. durch CSV-Rundreise entstandene NaN-
# Werte bei leeren Strings) werden durch einen leeren String ersetzt, um
# Fehler bei der Vektorisierung zu vermeiden.
train_df["text_normalisiert"] = train_df["text_normalisiert"].fillna("")
test_df["text_normalisiert"] = test_df["text_normalisiert"].fillna("")

# TF-IDF-Vektorisierung: Die Merkmalsmatrix wird ausschliesslich auf Basis
# des Trainingsdatensatzes angepasst (fit), um Data Leakage aus dem
# Testdatensatz zu vermeiden. Der Testdatensatz wird lediglich transformiert.
vektorisierer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
)
X_train = vektorisierer.fit_transform(train_df["text_normalisiert"])
X_test = vektorisierer.transform(test_df["text_normalisiert"])

y_train = train_df["rating"]
y_test = test_df["rating"]

modell = MultinomialNB()
modell.fit(X_train, y_train)

vorhersagen = modell.predict(X_test)

accuracy = accuracy_score(y_test, vorhersagen)
konfusionsmatrix = confusion_matrix(y_test, vorhersagen, labels=[1, 2, 3, 4, 5])
bericht = classification_report(y_test, vorhersagen, labels=[1, 2, 3, 4, 5])

print(f"Accuracy auf Testdatensatz: {accuracy:.4f}")
print("\nKlassifikationsbericht:")
print(bericht)
print("Konfusionsmatrix (Zeilen: tatsaechliche Klasse, Spalten: Vorhersage):")
print(konfusionsmatrix)

# Speicherung des Klassifikationsberichts als Textdatei
with open(os.path.join(ERGEBNIS_ORDNER, "klassifikationsbericht.txt"), "w") as datei:
    datei.write(f"Accuracy: {accuracy:.4f}\n\n")
    datei.write(bericht)

# Speicherung der Konfusionsmatrix als Abbildung
anzeige = ConfusionMatrixDisplay(
    confusion_matrix=konfusionsmatrix,
    display_labels=[1, 2, 3, 4, 5],
)
fig, ax = plt.subplots(figsize=(6, 6))
anzeige.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_xlabel("Vorhergesagte Bewertung")
ax.set_ylabel("Tatsächliche Bewertung")
ax.set_title("Konfusionsmatrix: Multinomial Naive Bayes")
plt.tight_layout()
plt.savefig(os.path.join(ERGEBNIS_ORDNER, "konfusionsmatrix.png"), dpi=200)
print(f"\nKonfusionsmatrix gespeichert unter {ERGEBNIS_ORDNER}/konfusionsmatrix.png")
