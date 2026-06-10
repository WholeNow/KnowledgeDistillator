import argparse
from datasets import load_from_disk
import csv
import json
import os

def dataset_to_csv(dataset_path, output_csv="dataset.csv"):
    """
    Salva il dataset in formato CSV

    :param dataset_path: Percorso del dataset da esportare
    :param output_csv: Percorso del file CSV di output
    """
    if not os.path.exists(dataset_path):
        print(f"Errore: Il percorso del dataset '{dataset_path}' non esiste.")
        return

    print(f"Caricamento del dataset da: {dataset_path} ...")
    try:
        ds = load_from_disk(dataset_path)
    except Exception as e:
        print(f"Errore durante il caricamento del dataset: {e}")
        return

    print(f"Esportazione in corso verso '{output_csv}' ...")
    try:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = None
            for row in ds:
                clean = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                        for k, v in row.items()}
                if writer is None:
                    # Imposta il delimitatore su punto e virgola e forza i doppi apici sui testi
                    writer = csv.DictWriter(f, fieldnames=clean.keys(), delimiter=';', quoting=csv.QUOTE_ALL)
                    writer.writeheader()
                writer.writerow(clean)
        print(f"Dataset esportato con successo in '{output_csv}'!")
    except Exception as e:
        print(f"Errore durante la scrittura del CSV: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esporta un dataset HuggingFace (salvato su disco) in formato CSV.")
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        required=True, 
        help="Il percorso locale del dataset HuggingFace da esportare (es. ./dataset_distilled_summarization_teacher)"
    )
    parser.add_argument(
        "--output_csv", 
        type=str, 
        default="dataset.csv", 
        help="Il nome o percorso del file CSV di output (default: dataset.csv)"
    )
    
    args = parser.parse_args()
    dataset_to_csv(args.dataset_path, args.output_csv)
