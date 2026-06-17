import argparse
from datasets import load_from_disk
import os

def normalize_teacher_summaries(dataset_path, output_path=None):
    """
    Normalizes the 'teacher_summary' or 'teacher_answer' fields in the dataset.
    It extracts only the relevant text following the '<|assistant|>\n' tag
    and removes any trailing or leading whitespace.
    The updated dataset is then saved to disk.
    
    Args:
        dataset_path (str): Local path to the dataset to normalize.
        output_path (str, optional): Path to save the normalized dataset. 
                                     Defaults to dataset_path + "_normalized".
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

    def clean_output(row):
        for field in ['teacher_summary', 'teacher_answer']:
            if field in row and row[field] is not None:
                row[field] = row[field].split("<|assistant|>\n")[-1].strip()
        return row

    print("Normalizing 'teacher_summary' and 'teacher_answer' fields...")
    # .map() creates and returns a new dataset instance with the applied modifications
    ds_normalized = ds.map(clean_output)

    if output_path is None:
        output_path = dataset_path + "_normalized"

    print(f"Saving the normalized dataset to: {output_path} ...")
    try:
        ds_normalized.save_to_disk(output_path)
        print("Dataset successfully saved.")
    except Exception as e:
        print(f"Error saving dataset: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalizes teacher output fields by extracting content after the '<|assistant|>' tag.")
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        required=True, 
        help="Local path to the HuggingFace dataset to normalize (e.g., ./dataset_distilled_qa_teacher)."
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default=None, 
        help="Path to save the new normalized dataset (default: <dataset_path>_normalized)."
    )
    
    args = parser.parse_args()
    normalize_teacher_summaries(args.dataset_path, args.output_path)
