# %% [markdown]
# # 🤖 Interactive Chat - Knowledge Distillator
# 
# Questo notebook ti permette di caricare il modello **Student** distillato (es. SmolLM) e testarlo in tempo reale come se fosse una vera e propria chat interattiva. L'input sarà bloccante e aspetterà la tua domanda, generando poi la risposta dal modello.

# %%
!pip install transformers torch accelerate

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------
# INSERISCI QUI IL PERCORSO DEL TUO MODELLO DISTILLATO
# Esempio: "./student_distilled_qa_final"
# Oppure un modello base come: "HuggingFaceTB/SmolLM-135M"
# ---------------------------------------------------------
MODEL_PATH = "./student_distilled_qa_final"

print(f"Caricamento del modello da: {MODEL_PATH} ...")

# Caricamento Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Applichiamo lo stesso template (ChatML) usato in addestramento
if tokenizer.chat_template is None:
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "<|im_start|>{{ message['role'] }}\n"
        "{{ message['content'] }}<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "<|im_start|>assistant\n"
        "{% endif %}"
    )

# Caricamento Modello
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",  # Usa la GPU se disponibile
    torch_dtype=torch.float16
)

print("Modello caricato con successo! Pronto per la chat.")

# %%
def generate_response(prompt, max_new_tokens=128):
    # Utilizziamo lo stesso identical system prompt dell'addestramento (Question Answering)
    messages = [
        {
            "role": "system", 
            "content": (
                "You are an expert assistant strictly dedicated to question answering. "
                "You must answer the user's question accurately. If a context is provided, base your answer on it. "
                "RULES: "
                "1) Provide a clear and concise answer. "
                "2) Do NOT add unnecessary conversational filler."
            )
        },
        {
            "role": "user", 
            "content": (
                f"Question:\n{prompt}\n\n"
                "Task: Answer the question."
            )
        }
    ]
    
    # Applichiamo il template per ottenere la stringa formattata
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    
    # Generazione
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,  # Greedy decoding come in fase di valutazione
    )
    
    # Decodifica isolando esclusivamente i token generati dal modello
    gen_tokens = outputs[0][prompt_len:]
    reply = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        
    return reply

print("=================================================")
print("             CHAT INTERATTIVA AVVIATA            ")
print("        Scrivi 'exit' o 'quit' per terminare     ")
print("=================================================\n")

while True:
    user_input = input("👤 Tu: ")
    
    if user_input.lower() in ['exit', 'quit']:
        print("\n👋 Chat terminata. Alla prossima!")
        break
        
    if not user_input.strip():
        continue
        
    # Generiamo la risposta
    response = generate_response(user_input)
    
    print(f"\n🤖 Modello: {response}\n")
    print("-" * 50)


