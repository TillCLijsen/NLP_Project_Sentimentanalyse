"""
Prüft, ob die in Tabelle 7 verwendeten Baseline-Werte (5-Sterne-Anteil je
Kategorie) mit den tatsächlichen Anteilen im Testdatensatz übereinstimmen,
oder ob sie nur zufällig den Rohdaten-Anteilen aus Tabelle 1 entsprechen.
"""

import pandas as pd
import os

DATEN_ORDNER = "data/processed"

test_df = pd.read_csv(os.path.join(DATEN_ORDNER, "test.csv"))

for kategorie, gruppe in test_df.groupby("kategorie"):
    anteil_5_sterne = (gruppe["rating"] == 5).mean()
    print(f"{kategorie}: {len(gruppe)} Testfälle, Baseline (Anteil 5 Sterne) = {anteil_5_sterne:.4f}")