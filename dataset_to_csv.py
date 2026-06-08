from datasets import load_from_disk
import csv
import json

def dataset_to_csv(dataset_path):
    """
        Salva il dataset in formato CSV, assicurandosi di gestire correttamente i campi che contengono liste o dizionari
        Utilizza un delimitatore che non confligga con i dati (in questo caso, il punto e virgola). 
        Inoltre, forza l'uso dei doppi apici per i testi per evitare problemi con i delimitatori all'interno dei campi.
    """
    ds = load_from_disk(dataset_path)

    with open("dataset.csv", "w", newline="", encoding="utf-8") as f:
        writer = None
        for row in ds:
            clean = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                    for k, v in row.items()}
            if writer is None:
                # Imposta il delimitatore su punto e virgola e forza i doppi apici sui testi
                writer = csv.DictWriter(f, fieldnames=clean.keys(), delimiter=';', quoting=csv.QUOTE_ALL)
                writer.writeheader()
            writer.writerow(clean)


def normalize_teacher_summaries(dataset_path):
    """
        Normalizza i campi "teacher_summary" del dataset, estraendo solo la parte rilevante dopo il tag "<|assistant|>\n"
         e rimuovendo eventuali spazi bianchi superflui. Salva poi il dataset aggiornato su disco.
    """
    ds = load_from_disk(dataset_path)

    def clean_output(row):
        full_output = row['teacher_summary']
        row['teacher_summary'] = full_output.split("<|assistant|>\n")[-1].strip()
        return row

    # .map() crea e restituisce una nuova istanza del dataset con i dati modificati
    ds_normalized = ds.map(clean_output)

    ds_normalized.save_to_disk(dataset_path + "_normalized")

    print("Nuovo dataset: " + dataset_path + "_normalized")

    



# Spostarsi nella directory dove è presenta il dataset salvato e caricarlo
ds = "./dataset_distilled_summarization_teacher"

choice = input("Scegli l'operazione da eseguire:\n1. Converti dataset in CSV\n2. Normalizza teacher summaries\nInserisci 1 o 2: ")

if choice == "1":
    try:
        dataset_to_csv(ds)
        print("Dataset salvato in formato CSV come 'dataset.csv'")
    except Exception as e:
        print(f"Errore durante la conversione del dataset in CSV: {e}")

elif choice == "2":
    try:
        normalize_teacher_summaries(ds)
        print("Teacher summaries normalizzati e dataset salvato su disco:")
    except Exception as e:
        print(f"Errore durante la normalizzazione: {e}")

else:
    print("Scelta non valida. Inserisci 1 o 2.")
