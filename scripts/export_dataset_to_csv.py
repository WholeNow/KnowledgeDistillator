import argparse
from datasets import load_from_disk
import csv
import json
import os

def dataset_to_csv(dataset_path, output_csv="dataset.csv"):
    """
    Exports a HuggingFace dataset saved on disk to a CSV format.

    Args:
        dataset_path (str): Local path to the dataset to export.
        output_csv (str): Path for the output CSV file. Defaults to "dataset.csv".
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        return

    print(f"Loading dataset from: {dataset_path} ...")
    try:
        ds = load_from_disk(dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Exporting to '{output_csv}' ...")
    try:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = None
            for row in ds:
                clean = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                        for k, v in row.items()}
                if writer is None:
                    # Set semicolon as delimiter and force double quotes for string fields
                    writer = csv.DictWriter(f, fieldnames=clean.keys(), delimiter=';', quoting=csv.QUOTE_ALL)
                    writer.writeheader()
                writer.writerow(clean)
        print(f"Dataset successfully exported to '{output_csv}'!")
    except Exception as e:
        print(f"Error writing to CSV: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a HuggingFace dataset (saved on disk) to CSV format.")
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        required=True, 
        help="Local path to the HuggingFace dataset to export (e.g., ./dataset_distilled_summarization_teacher)."
    )
    parser.add_argument(
        "--output_csv", 
        type=str, 
        default="dataset.csv", 
        help="Name or path of the output CSV file (default: dataset.csv)."
    )
    
    args = parser.parse_args()
    dataset_to_csv(args.dataset_path, args.output_csv)
