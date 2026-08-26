"""
Fehleranalyse: sucht konkrete Beispielrezensionen aus dem Testdatensatz,
die vom Naive-Bayes- bzw. Logistic-Regression-Modell falsch klassifiziert
wurden. Dient dazu, die im Bericht diskutierten Hypothesen (Mehrheitsklassen-
Verzerrung, Ambivalenz neutraler Rezensionen) mit Beispielen zu belegen.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import os

DATEN_ORDNER = "data/processed"

train_df = pd.read_csv(os.path.join(DATEN_ORDNER, "train.csv"))
test_df = pd.read_csv(os.path.join(DATEN_ORDNER, "test.csv"))

train_df["text_normalisiert"] = train_df["text_normalisiert"].fillna("")
test_df["text_normalisiert"] = test_df["text_normalisiert"].fillna("")

vektorisierer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
X_train = vektorisierer.fit_transform(train_df["text_normalisiert"])
X_test = vektorisierer.transform(test_df["text_normalisiert"])

y_train = train_df["rating"]
y_test = test_df["rating"]

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)
pred_nb = naive_bayes.predict(X_test)

logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
logreg.fit(X_train, y_train)
pred_lr = logreg.predict(X_test)

test_df = test_df.reset_index(drop=True)
test_df["pred_nb"] = pred_nb
test_df["pred_lr"] = pred_lr


def zeige_beispiel(zeile):
    print(f"Kategorie: {zeile['kategorie']}")
    print(f"Tatsächliche Bewertung: {zeile['rating']} Sterne")
    print(f"Titel: {zeile['titel']}")
    text = str(zeile["text"])
    gekuerzt = text[:400] + ("..." if len(text) > 400 else "")
    print(f"Text: {gekuerzt}")
    print()


print("=== Beispiel 1: 1-Stern-Rezension, von Naive Bayes als 5 Sterne klassifiziert ===")
kandidaten = test_df[(test_df["rating"] == 1) & (test_df["pred_nb"] == 5)]
if len(kandidaten) > 0:
    zeige_beispiel(kandidaten.iloc[0])
else:
    print("Kein Beispiel gefunden.\n")

print("=== Beispiel 2: 3-Sterne-Rezension (neutral), von Logistic Regression als 5 Sterne eingestuft ===")
kandidaten = test_df[(test_df["rating"] == 3) & (test_df["pred_lr"] == 5)]
if len(kandidaten) > 0:
    zeige_beispiel(kandidaten.iloc[0])
else:
    print("Kein Beispiel gefunden.\n")

print("=== Beispiel 3: 3-Sterne-Rezension (neutral), von Logistic Regression als 1 Stern eingestuft ===")
kandidaten = test_df[(test_df["rating"] == 3) & (test_df["pred_lr"] == 1)]
if len(kandidaten) > 0:
    zeige_beispiel(kandidaten.iloc[0])
else:
    print("Kein Beispiel gefunden.\n")
