"""
Lädt 5000 Reviews pro Kategorie vom Amazon-Reviews-2023 Datensatz
(Video Games, Automotive, Pet Supplies) und speichert sie als CSV.
"""



from datasets import load_dataset
import pandas as pd
import os

# Untersuchte Produktkategorien und Stichprobenumfang je Kategorie.
# Der Stichprobenumfang wurde gewählt, um Rechenzeit und Speicherbedarf
# im Rahmen des Projekts zu begrenzen. Die Originalkategorien umfassen
# teilweise mehrere Millionen Rezensionen.

KATEGORIEN = ["Video_Games", "Automotive", "Pet_Supplies"]
STICHPROBE_PRO_KATEGORIE = 5000

OUTPUT_ORDNER = "data/raw"
os.makedirs(OUTPUT_ORDNER, exist_ok=True)

for kategorie in KATEGORIEN:
    print(f"Lade Kategorie: {kategorie} ...")

    # Streaming-Modus: Der Datensatz wird nicht vollständig heruntergeladen,
    # sondern sequenziell abgerufen, bis der definierte Stichprobenumfang
    # erreicht ist. Dies reduziert Download-Volumen und Ladezeit.
    ds_stream = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_review_{kategorie}",
        trust_remote_code=True,
        split="full",
        streaming=True,
    )

    beispiele = []
    for i, beispiel in enumerate(ds_stream):
        if i >= STICHPROBE_PRO_KATEGORIE:
            break
        beispiele.append({
            "kategorie": kategorie,
            "titel": beispiel.get("title", ""),
            "text": beispiel.get("text", ""),
            "rating": beispiel.get("rating", None),
        })

    df = pd.DataFrame(beispiele)
    ausgabe_pfad = os.path.join(OUTPUT_ORDNER, f"{kategorie}.csv")
    df.to_csv(ausgabe_pfad, index=False)

    print(f"  -> {len(df)} Rezensionen gespeichert in {ausgabe_pfad}")
    print(f"  -> Bewertungsverteilung:\n{df['rating'].value_counts().sort_index()}\n")

print("Datenerhebung abgeschlossen für alle Kategorien.")
