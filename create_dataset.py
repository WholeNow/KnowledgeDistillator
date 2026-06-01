import torch
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer

# 1. Inizializzazione Teacher ottimizzata per T4
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# La pipeline gestisce il batching in modo efficiente
generator = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float16, # Obbligatorio per T4 (no bfloat16)
    device_map="auto",
    batch_size=16 # Ottimizzato per 16GB VRAM
)

# Funzione per formattare il prompt secondo il template di TinyLlama
def format_prompt(dialogue):
    messages = [
        {"role": "system", "content": "You are a highly accurate summarization assistant. Provide a concise summary of the following conversation."},
        {"role": "user", "content": f"Summarize this dialogue:\n\n{dialogue}"}
    ]
    # Applica il ChatML template nativo del modello
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

dataset_name = input("Inserisci il nome del dataset da utilizzare (es. samsum): ")

# 2. Caricamento e preparazione dataset
dataset = load_dataset(dataset_name, split="train") # Per test usa split="train[:500]"

# Creiamo una lista di prompt formattati
prompts = [format_prompt(dialogue) for dialogue in dataset['dialogue']]

# 3. Generazione batch
print("Inizio generazione pseudo-labels...")
outputs = generator(
    prompts,
    max_new_tokens=128,
    temperature=0.1,    # Bassa temperatura per KD (riduce varianza)
    do_sample=False,    # Greedy decoding per stabilità
    return_full_text=False # Restituisce solo la risposta generata
)

# Estrazione dei testi
teacher_summaries = [out[0]['generated_text'].strip() for out in outputs]

# 4. Aggiunta delle pseudo-label al dataset e salvataggio
distilled_dataset = dataset.add_column("teacher_summary", teacher_summaries)
distilled_dataset.save_to_disk("./samsum_distilled_tinyllama")
print("Dataset salvato con successo.")