from datasets import load_from_disk
import csv
import json

# Spostarsi nella directory dove è presenta il dataset salvato e caricarlo
ds = load_from_disk("./samsum_distilled_tinyllama")

with open("dataset2.csv", "w", newline="", encoding="utf-8") as f:
    writer = None
    for row in ds:
        clean = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                 for k, v in row.items()}
        if writer is None:
            # Imposta il delimitatore su punto e virgola e forza i doppi apici sui testi
            writer = csv.DictWriter(f, fieldnames=clean.keys(), delimiter=';', quoting=csv.QUOTE_ALL)
            writer.writeheader()
        writer.writerow(clean)