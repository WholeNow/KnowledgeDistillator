import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def clean_value(x, col):
    """
    Pulisce e converte i valori della tabella.
    Gestisce casi come '1.355.613' convertendoli correttamente a float o int.
    """
    if pd.isna(x):
        return x
    
    x = str(x).strip()
    if x == '':
        return None
        
    if col in ['Step', 'Num Tokens']:
        # Rimuove i punti delle migliaia per i numeri interi
        x = x.replace('.', '').replace(',', '')
        try:
            return int(x)
        except ValueError:
            return None
    else:
        # Per Loss, Entropy e Accuracy
        # Trasforma formati anomali come '1.355.613' in '1.355613'
        parts = x.split('.')
        if len(parts) > 2:
            x = parts[0] + '.' + ''.join(parts[1:])
        x = x.replace(',', '.') # Gestisce la virgola come separatore decimale se presente
        try:
            return float(x)
        except ValueError:
            return None

def main():
    parser = argparse.ArgumentParser(description='Genera grafici dalle metriche di addestramento.')
    parser.add_argument('input_file', type=str, help='Percorso del file di input (es. dati.tsv). I dati devono essere separati da Tabulazione (TAB).')
    parser.add_argument('--output_dir', type=str, default='.', help='Cartella in cui salvare i grafici generati (default: cartella corrente).')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Errore: il file {args.input_file} non esiste.")
        return

    print(f"Leggendo i dati da {args.input_file}...")
    try:
        # Legge il file assumendo che sia separato da tab
        df = pd.read_csv(args.input_file, sep='\t')
        
        # Se c'è una sola colonna, forse non era separato da tab
        if len(df.columns) == 1:
            print("Attenzione: sembra che il file non sia separato da tab (TAB). Provo a usare gli spazi (potrebbe fallire se i nomi delle colonne hanno spazi).")
            # Prova con separatori di spazio multipli (es. 2+ spazi)
            df = pd.read_csv(args.input_file, sep=r'\s{2,}', engine='python')
            
    except Exception as e:
        print(f"Errore durante la lettura del file: {e}")
        return

    # Pulisce i nomi delle colonne per evitare spazi extra
    df.columns = [c.strip() for c in df.columns]
    print(f"Colonne trovate: {', '.join(df.columns)}")

    # Applica la pulizia dei dati
    for col in df.columns:
        if col in ['Step', 'Num Tokens', 'Training Loss', 'Validation Loss', 'Entropy', 'Mean Token Accuracy']:
            df[col] = df[col].apply(lambda x: clean_value(x, col))

    # Assicurati che l'output dir esista
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Grafico per Loss (Training e Validation)
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
    print(f"Salvato: {loss_path}")
    plt.close()

    # 2. Grafico per Entropy
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
        print(f"Salvato: {entropy_path}")
        plt.close()

    # 3. Grafico per Mean Token Accuracy
    if 'Mean Token Accuracy' in df.columns:
        plt.figure(figsize=(10, 6))
        # Rimuove le righe con NaN (es. il primo step che non ha questo dato)
        valid_acc = df.dropna(subset=['Mean Token Accuracy'])
        plt.plot(valid_acc['Step'], valid_acc['Mean Token Accuracy'], label='Mean Token Accuracy', marker='o', color='purple', linestyle='-')
        plt.title('Mean Token Accuracy over Steps')
        plt.xlabel('Step')
        plt.ylabel('Accuracy')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        acc_path = os.path.join(args.output_dir, 'accuracy_plot.png')
        plt.savefig(acc_path, dpi=300, bbox_inches='tight')
        print(f"Salvato: {acc_path}")
        plt.close()

    print("Generazione dei grafici completata!")

if __name__ == "__main__":
    main()
