import argparse
from datasets import load_from_disk
import os

def normalize_teacher_summaries(dataset_path, output_path=None):
    """
    Normalizza i campi "teacher_summary" o "teacher_answer" del dataset, estraendo solo la parte rilevante 
    dopo il tag "<|assistant|>\n" e rimuovendo eventuali spazi bianchi superflui. 
    Salva poi il dataset aggiornato su disco.
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

    def clean_output(row):
        for field in ['teacher_summary', 'teacher_answer']:
            if field in row and row[field] is not None:
                row[field] = row[field].split("<|assistant|>\n")[-1].strip()
        return row

    print("Normalizzazione dei campi teacher_summary/teacher_answer in corso...")
    # .map() crea e restituisce una nuova istanza del dataset con i dati modificati
    ds_normalized = ds.map(clean_output)

    if output_path is None:
        output_path = dataset_path + "_normalized"

    print(f"Salvataggio del nuovo dataset in: {output_path} ...")
    try:
        ds_normalized.save_to_disk(output_path)
        print("Salvataggio completato con successo.")
    except Exception as e:
        print(f"Errore durante il salvataggio: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalizza i campi di output del Teacher rimuovendo il tag '<|assistant|>'.")
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        required=True, 
        help="Il percorso locale del dataset HuggingFace da normalizzare (es. ./dataset_distilled_qa_teacher)"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default=None, 
        help="Il percorso dove salvare il nuovo dataset normalizzato. (default: <dataset_path>_normalized)"
    )
    
    args = parser.parse_args()
    normalize_teacher_summaries(args.dataset_path, args.output_path)
