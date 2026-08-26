"""
Schaut sich die Accuracy von beiden Modellen einzeln pro Kategorie an.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

DATEN_ORDNER = "data/processed"
ERGEBNIS_ORDNER = "results"
os.makedirs(ERGEBNIS_ORDNER, exist_ok=True)

train_df = pd.read_csv(os.path.join(DATEN_ORDNER, "train.csv"))
test_df = pd.read_csv(os.path.join(DATEN_ORDNER, "test.csv"))

train_df["text_normalisiert"] = train_df["text_normalisiert"].fillna("")
test_df["text_normalisiert"] = test_df["text_normalisiert"].fillna("")

vektorisierer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
)
X_train = vektorisierer.fit_transform(train_df["text_normalisiert"])
X_test = vektorisierer.transform(test_df["text_normalisiert"])

y_train = train_df["rating"]
y_test = test_df["rating"]

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)
vorhersagen_nb = naive_bayes.predict(X_test)

logistic_regression = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)
logistic_regression.fit(X_train, y_train)
vorhersagen_lr = logistic_regression.predict(X_test)

test_df = test_df.reset_index(drop=True)
ergebnis_df = pd.DataFrame({
    "kategorie": test_df["kategorie"],
    "tatsächlich": y_test.reset_index(drop=True),
    "vorhersage_nb": vorhersagen_nb,
    "vorhersage_lr": vorhersagen_lr,
})

zeilen = []
for kategorie, gruppe in ergebnis_df.groupby("kategorie"):
    accuracy_nb = accuracy_score(gruppe["tatsächlich"], gruppe["vorhersage_nb"])
    accuracy_lr = accuracy_score(gruppe["tatsächlich"], gruppe["vorhersage_lr"])
    zeilen.append({
        "kategorie": kategorie,
        "anzahl_testfälle": len(gruppe),
        "accuracy_naive_bayes": round(accuracy_nb, 4),
        "accuracy_logistic_regression": round(accuracy_lr, 4),
    })

kategorie_vergleich = pd.DataFrame(zeilen)
print("Accuracy je Produktkategorie:")
print(kategorie_vergleich.to_string(index=False))

kategorie_vergleich.to_csv(
    os.path.join(ERGEBNIS_ORDNER, "accuracy_je_kategorie.csv"), index=False
)
print(f"\nTabelle gespeichert unter {ERGEBNIS_ORDNER}/accuracy_je_kategorie.csv")
