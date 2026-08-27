# Sentimentanalyse von Produktrezensionen

Code zum Projektbericht "Sentimentanalyse von Produktrezensionen: Entwicklung und
Evaluation eines Textklassifizierungssystems für Amazon-Produktbewertungen"
(IU Internationale Hochschule, Angewandte Künstliche Intelligenz B.Sc.).

## Datensatz

McAuley Lab (2023). *Amazon Reviews 2023* [Datensatz]. Hugging Face.
https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

Verwendete Kategorien: Video Games, Automotive, Pet Supplies (je 5.000 Rezensionen).

## Setup

```
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install "datasets<4.0.0" pandas scikit-learn nltk matplotlib
```

Hinweis: `datasets` muss unter Version 4.0 bleiben, da der Datensatz ein
Loading Script verwendet, das ab Version 4.0 nicht mehr unterstützt wird.

## Skripte (in dieser Reihenfolge ausführen)

| Skript | Beschreibung |
|---|---|
| `01_datenerhebung.py` | Lädt 5.000 Reviews pro Kategorie und speichert sie als CSV in `data/raw/` |
| `02_vorverarbeitung.py` | Führt die CSVs zusammen, normalisiert den Text und splittet in Train/Test (80/20) |
| `03_train_naive_bayes.py` | Trainiert Multinomial Naive Bayes mit TF-IDF, wertet auf dem Testset aus |
| `04_train_logistic_regression.py` | Trainiert Logistic Regression (balanced) zum Vergleich |
| `05_kategorienvergleich.py` | Vergleicht die Accuracy beider Modelle je Produktkategorie |
| `06_sentiment_polaritaet.py` | Reduziert die Bewertung auf 3 Klassen (negativ/neutral/positiv) und klassifiziert erneut |
| `07_fehleranalyse.py` | Sucht konkrete Beispielrezensionen, die falsch klassifiziert wurden (Basis für Kapitel 4.6) |
| `08_baseline_check.py` | Prüft die tatsächlichen Mehrheitsklassen-Baselines je Kategorie im Testdatensatz (Basis für Tabelle 7) |

Ausgaben (CSV-Dateien, Konfusionsmatrizen, Klassifikationsberichte) landen automatisch
in `data/processed/` bzw. `results/`.

## Ergebnisse (Kurzüberblick)

| Modell | Accuracy | Macro F1 |
|---|---|---|
| Naive Bayes | 0,673 | 0,25 |
| Logistic Regression (balanced) | 0,657 | 0,48 |
| Logistic Regression, 3-Klassen-Polarität | 0,819 | 0,64 |

Ausführliche Diskussion siehe Projektbericht.
