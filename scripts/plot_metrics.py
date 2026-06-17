import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def clean_value(x, col):
    """
    Cleans and converts table values to their appropriate data types.
    Handles anomalies such as '1.355.613' by converting them correctly to float or int.
    
    Args:
        x: The value to clean.
        col (str): The column name to determine the expected data type.
        
    Returns:
        The cleaned value as an int, float, or None.
    """
    if pd.isna(x):
        return x
    
    x = str(x).strip()
    if x == '':
        return None
        
    if col in ['Step', 'Num Tokens']:
        # Remove thousands separators for integers
        x = x.replace('.', '').replace(',', '')
        try:
            return int(x)
        except ValueError:
            return None
    else:
        # For Loss, Entropy, and Accuracy
        # Transforms anomalous formats like '1.355.613' to '1.355613'
        parts = x.split('.')
        if len(parts) > 2:
            x = parts[0] + '.' + ''.join(parts[1:])
        x = x.replace(',', '.') # Handles comma as decimal separator if present
        try:
            return float(x)
        except ValueError:
            return None

def main():
    parser = argparse.ArgumentParser(description='Generates plots from training metrics.')
    parser.add_argument('input_file', type=str, help='Path to the input file (e.g., data.tsv). Data must be Tab-Separated (TAB).')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory where the generated plots will be saved (default: current directory).')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: The file {args.input_file} does not exist.")
        return

    print(f"Reading data from {args.input_file}...")
    try:
        # Read the file assuming it's tab-separated
        df = pd.read_csv(args.input_file, sep='\t')
        
        # If there's only one column, it might not be tab-separated
        if len(df.columns) == 1:
            print("Warning: The file does not appear to be tab-separated (TAB). Trying to use spaces as separators (may fail if column names contain spaces).")
            # Try with multiple space separators (e.g., 2+ spaces)
            df = pd.read_csv(args.input_file, sep=r'\s{2,}', engine='python')
            
    except Exception as e:
        print(f"Error reading the file: {e}")
        return

    # Clean column names to remove any extra whitespaces
    df.columns = [c.strip() for c in df.columns]
    print(f"Found columns: {', '.join(df.columns)}")

    # Apply data cleaning
    for col in df.columns:
        if col in ['Step', 'Num Tokens', 'Training Loss', 'Validation Loss', 'Entropy', 'Mean Token Accuracy']:
            df[col] = df[col].apply(lambda x: clean_value(x, col))

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Plot for Loss (Training and Validation)
    plt.figure(figsize=(10, 6))
    if 'Training Loss' in df.columns:
        plt.plot(df['Step'], df['Training Loss'], label='Training Loss', marker='o', linestyle='-')
    if 'Validation Loss' in df.columns:
        plt.plot(df['Step'], df['Validation Loss'], label='Validation Loss', marker='o', linestyle='-')
    plt.title('Loss over Steps')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    loss_path = os.path.join(args.output_dir, 'loss_plot.png')
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {loss_path}")
    plt.close()

    # 2. Plot for Entropy
    if 'Entropy' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df['Step'], df['Entropy'], label='Entropy', marker='o', color='green', linestyle='-')
        plt.title('Entropy over Steps')
        plt.xlabel('Step')
        plt.ylabel('Entropy')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        entropy_path = os.path.join(args.output_dir, 'entropy_plot.png')
        plt.savefig(entropy_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {entropy_path}")
        plt.close()

    # 3. Plot for Mean Token Accuracy
    if 'Mean Token Accuracy' in df.columns:
        plt.figure(figsize=(10, 6))
        # Remove rows with NaN (e.g., the first step which might lack this data)
        valid_acc = df.dropna(subset=['Mean Token Accuracy'])
        plt.plot(valid_acc['Step'], valid_acc['Mean Token Accuracy'], label='Mean Token Accuracy', marker='o', color='purple', linestyle='-')
        plt.title('Mean Token Accuracy over Steps')
        plt.xlabel('Step')
        plt.ylabel('Accuracy')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        acc_path = os.path.join(args.output_dir, 'accuracy_plot.png')
        plt.savefig(acc_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {acc_path}")
        plt.close()

    print("Plot generation completed!")

if __name__ == "__main__":
    main()
