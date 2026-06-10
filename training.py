# %% [markdown]
# # Knowledge Distillation - Training
# 
# <a target="_blank" href="https://colab.research.google.com/github/WholeNow/KnowledgeDistillator/blob/main/Summarization.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# %% [markdown]
# ## Install dependencies and import libraries

# %%
!pip install -q transformers datasets trl evaluate rouge_score bert_score matplotlib accelerate bitsandbytes

# %%
# All imports
from datasets import load_dataset, Dataset, load_from_disk
from transformers import pipeline, AutoTokenizer, GenerationConfig, AutoModelForCausalLM
from tqdm.auto import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from trl import SFTTrainer, SFTConfig
import evaluate
import math
import random
import os
import gc
import torch
import time

# %% [markdown]
# ## Configurazione
# 
# | Parametro | Descrizione | Valori Ammessi |
# | :--- | :--- | :--- |
# | **TASK** | Il task NLP da eseguire (determina dataset e prompt). | `"summarization"`, `"question_answering"` |
# | **TEACHER_MODEL_ID** | Modello "grande" da cui distillare le etichette. | ID HuggingFace (`"TinyLlama/TinyLlama-1.1B-Chat-v1.0"`) |
# | **STUDENT_MODEL_ID** | Modello "piccolo" da addestrare tramite SFT. | ID HuggingFace (`"HuggingFaceTB/SmolLM-135M"`) |
# | **MAX_TRAIN_SAMPLES** | Esempi massimi per la generazione del Teacher. | `"all"`, o intero testuale (es. `"10000"`) |
# | **MAX_TEST_SAMPLES** | Esempi massimi per la valutazione finale. | `"all"`, o intero testuale (es. `"150"`) |
# | **TEACHER_PIPELINE_BATCH_SIZE**| Dimensione batch per l'inferenza del Teacher. | Intero positivo (es. `16`) |
# | **STUDENT_BATCH_SIZE** | Batch size per device (training Student). | Intero positivo (es. `4`) |
# | **GRAD_ACCUMULATION** | Step di accumulo del gradiente per lo Student. | Intero positivo (es. `4`) |
# | **EPOCHS** | Numero di epoche di addestramento. | Intero positivo (es. `3`) |
# | **LEARNING_RATE** | Learning rate massimo (scheduler Cosine). | Float (es. `1e-5`) |
# | **MAX_SEQ_LENGTH** | Lunghezza massima in token per lo Student. | Intero positivo (es. `1024`) |

# %%
# ============================================================
#                     CONFIGURAZIONE
# ============================================================

# TASK:
# "summarization" (knkarthick/samsum)
# "question_answering" (databricks/databricks-dolly-15k)
TASK = "summarization"

# TEACHER: scegli uno dei due
# "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# "Qwen/Qwen2.5-1.5B-Instruct"
TEACHER_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# STUDENT:
# "HuggingFaceTB/SmolLM-135M"
STUDENT_MODEL_ID = "HuggingFaceTB/SmolLM-135M"

# Numero di sample da usare per il training del Teacher (generazione etichette)
# Usa 'all' per l'intero dataset, oppure un intero es. 500
MAX_TRAIN_SAMPLES = '10000'

# Numero di sample per la valutazione finale
# Usa 'all' oppure un intero es. 100
MAX_TEST_SAMPLES = '150'

# Batch size per la pipeline del Teacher
TEACHER_PIPELINE_BATCH_SIZE = 16

# Iperparametri training Student
STUDENT_BATCH_SIZE = 4
GRAD_ACCUMULATION = 4
EPOCHS = 2
LEARNING_RATE = 1e-5
MAX_SEQ_LENGTH = 1024

# Cartelle di output
BASE_OUTPUT_DIR = f"./results"
TEACHER_DATASET_DIR = os.path.join(BASE_OUTPUT_DIR, f"dataset_distilled_{TASK}_teacher")
STUDENT_OUTPUT_DIR  = os.path.join(BASE_OUTPUT_DIR, f"student_distilled_{TASK}")
STUDENT_FINAL_DIR   = os.path.join(BASE_OUTPUT_DIR, f"student_distilled_{TASK}_final")

# %%
def clear_memory():
    for var in ['generator', 'student_model', 'model', 'trainer']:
        if var in globals():
            del globals()[var]
    gc.collect()
    torch.cuda.empty_cache()

# %% [markdown]
# ## Generazione Dataset Distillato (Teacher)
# `Pipeline` di HuggingFace con batching nativo per generare le pseudo-label del Teacher in modo efficiente e salvarle su disco.
# Se il file è già presente, viene caricato direttamente.
# 
# aggiungere controllo se il file esiste già, per evitare di rigenerare tutto da capo ogni volta

# %% [markdown]
# 

# %%
# ── Formattazione prompt secondo template TinyLlama ────────────────────
def build_prompt(tokenizer, example):
    """
    Costruisce il prompt per un esempio del dataset in base al TASK specificato.

    param tokenizer: il tokenizer del modello Teacher, usato per applicare il template di chat
    param example: un esempio del dataset, con campi diversi a seconda del TASK
    return: un dizionario con chiavi 'prompt', 'target' e 'input_text' (testo originale usato per valutazione)
    """

    if TASK == "summarization":
        messages = [
        {
            "role": "system", 
            "content": (
                "You are an expert assistant strictly dedicated to abstractive summarization. "
                "You must extract the core event, problem, or decision from the conversation. "
                "RULES: "
                "1) Do NOT copy, repeat, or quote the dialogue. "
                "2) Do NOT use dialogue format (e.g., 'Name:'). "
                "3) Write exactly one or two sentences in the third person."
            )
        },
        {
            "role": "user", 
            "content": (
                f"Dialogue:\n{example['dialogue']}\n\n"
                "Task: Write a brief, third-person narrative summary describing what the people are doing or talking about."
            )
        }
        ]
        return {
            "prompt": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            "target": example["summary"],
            "input_text": example["dialogue"]
        }
    elif TASK == "question_answering":
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
                (f"Context:\n{example.get('context', '')}\n\n" if example.get('context') else "") +
                f"Question:\n{example['instruction']}\n\n"
                "Task: Answer the question."
            )
        }
        ]
        return {
            "prompt": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            "target": example["response"],
            "input_text": messages[1]["content"]
        }

# %%
# ── Cleanup VRAM/RAM ──
clear_memory()

# ── Configurazione prompt ──────────────────
if TASK == "summarization":
    raw_dataset = load_dataset("knkarthick/samsum", split="train")
elif TASK == "question_answering":
    full_dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    raw_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)["train"]

# ── Inizializzazione tokenizer Teacher ────────────────────
tokenizer_teacher = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID)
if tokenizer_teacher.pad_token is None:
    tokenizer_teacher.pad_token = tokenizer_teacher.eos_token
    
# ── Limitazione sample ─────────────────────────────────────
if MAX_TRAIN_SAMPLES != 'all':
    raw_dataset = raw_dataset.select(range(min(int(MAX_TRAIN_SAMPLES), len(raw_dataset))))

print(f"Task: {TASK} | Teacher: {TEACHER_MODEL_ID} | Train samples: {len(raw_dataset)}")
print("\nColonne del dataset:", raw_dataset.column_names)


# FIX: filtra i prompt che superano il context length del Teacher (2048 token per TinyLlama).
# Prompt troppo lunghi causano position embedding fuori range -> output degradati -> pseudo-label rumorose.
# Il margine di 128 token riserva spazio alla risposta (max_new_tokens=128).
prompts_raw = [build_prompt(tokenizer_teacher, example) for example in raw_dataset]

TEACHER_MAX_TOKENS = tokenizer_teacher.model_max_length or 2048
TOKEN_MARGIN = 128  # uguale a max_new_tokens in gen_config
 
valid_indices = []
skipped = 0
for i, p in enumerate(prompts_raw):
    n_tokens = len(tokenizer_teacher(p["prompt"], add_special_tokens=False)["input_ids"])
    if n_tokens <= TEACHER_MAX_TOKENS - TOKEN_MARGIN:
        valid_indices.append(i)
    else:
        skipped += 1
 
if skipped:
    print(f"[WARN] {skipped} sample scartati perche il prompt supera {TEACHER_MAX_TOKENS - TOKEN_MARGIN} token.")
 
prompts = [prompts_raw[i] for i in valid_indices]
raw_dataset = raw_dataset.select(valid_indices)
prompt_dataset = Dataset.from_dict({"prompt": [p["prompt"] for p in prompts]})

# vecchio
#prompts = [build_prompt(tokenizer_teacher, example) for example in raw_dataset]
#prompt_dataset = Dataset.from_dict({"prompt": [p["prompt"] for p in prompts]})

print("\nInizio generazione pseudo-labels...")
teacher_outputs = []

# 1. Configurazione generazione (massimo 128 token, no sampling, solo testo generato)
gen_config = GenerationConfig(
    max_new_tokens=128,
    do_sample=False, # greedy decoding per coerenza e riproducibilità
    return_full_text=False,
    max_length=None,
    num_beams=1,
)

# 2. Inizializzazione pipeline del Teacher
generator = pipeline(
    "text-generation",
    model=TEACHER_MODEL_ID,
    dtype=torch.float16, 
    device_map="auto",
    batch_size=TEACHER_PIPELINE_BATCH_SIZE
)
generator.model.generation_config.max_length = None  # Annulla il limite nativo per evitare conflitti

# 3. Generazione pseudo-labels con progress bar
for out in tqdm(generator(KeyDataset(prompt_dataset, "prompt"), generation_config=gen_config),total=len(prompts),desc=f"Distillazione per {TASK}"):
    full_output = out[0]['generated_text']
    clean_summary = full_output.split("<|assistant|>\n")[-1].strip()
    teacher_outputs.append(clean_summary)

# 4. Aggiunta delle pseudo-label al dataset e salvataggio
match TASK:
    case "question_answering":
        column_name = "teacher_answer"
    case "summarization":
        column_name = "teacher_summary"
    case _:
        raise ValueError(f"Task {TASK} non supportato.")

distilled_dataset = raw_dataset.add_column(column_name, teacher_outputs)
distilled_dataset.save_to_disk(TEACHER_DATASET_DIR)
print("Dataset salvato con successo.")

# %% [markdown]
# ## Training dello Student
# Fine-tuning dello Student sulle pseudo-label generate dal Teacher tramite `SFTTrainer`.
# 
# I passaggi eseguiti in questa sezione sono:
# 1. Mappatura del dataset nel formato messaggi (system/user/assistant).
# 2. Iniezione del chat template con token standard per fermare correttamente la generazione.
# 3. Avvio dell'addestramento supervisionato (SFT).

# %%
# Mappatura nel formato nativo Prompt-Completion
def format_example(example):
    if TASK == "summarization":
        system_msg = (
                "You are an expert assistant strictly dedicated to abstractive summarization. "
                "You must extract the core event, problem, or decision from the conversation. "
                "RULES: "
                "1) Do NOT copy, repeat, or quote the dialogue. "
                "2) Do NOT use dialogue format (e.g., 'Name:'). "
                "3) Write exactly one or two sentences in the third person."
            )
        user_msg = (
                f"Dialogue:\n{example['dialogue']}\n\n"
                "Task: Write a brief, third-person narrative summary describing what the people are doing or talking about."
            )
        assistant_msg = example['teacher_summary']

    elif TASK == "question_answering":
        system_msg = (
            "You are an expert assistant strictly dedicated to question answering. "
            "You must answer the user's question accurately. If a context is provided, base your answer on it. "
            "RULES: "
            "1) Provide a clear and concise answer. "
            "2) Do NOT add unnecessary conversational filler."
        )
        context_str = f"Context:\n{example.get('context', '')}\n\n" if example.get('context') else ""
        user_msg = (
                f"{context_str}Question:\n{example['instruction']}\n\n"
                "Task: Answer the question."
            )
        assistant_msg = example['teacher_answer']

    return {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }

# %%
# ── Cleanup VRAM/RAM per evitare OOM su riavvio/interruzione ──
clear_memory()

# ── Inizializzazione tokenizer Student ───────────────────
tokenizer_student = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID)
if tokenizer_student.pad_token is None:
    tokenizer_student.pad_token = tokenizer_student.eos_token


# ── Aggiunta token speciali <|im_start|> e <|im_end|> ────────────────────────
special_tokens_to_add = []
if "<|im_start|>" not in tokenizer_student.vocab:
    special_tokens_to_add.append("<|im_start|>")
if "<|im_end|>" not in tokenizer_student.vocab:
    special_tokens_to_add.append("<|im_end|>")


# FIX: i token vengono effettivamente aggiunti al vocabolario del tokenizer.
# Senza questa chiamata, special_tokens_to_add veniva costruito ma mai applicato:
# len(tokenizer_student) non cambiava e il resize successivo era un no-op silenzioso.
if special_tokens_to_add:
    tokenizer_student.add_special_tokens({"additional_special_tokens": special_tokens_to_add})
    print(f"Token speciali aggiunti: {special_tokens_to_add}")


# ── Iniezione forzata del ChatML template ──────────────────────
# Struttura del template:
#
#   <|im_start|>system\n
#   [contenuto system]<|im_end|>\n
#   <|im_start|>user\n
#   [contenuto user]<|im_end|>\n
#   <|im_start|>assistant\n
#   {% generation %}[risposta assistant]<|im_end|>\n{% endgeneration %}
#
# I marker {% generation %} delimitano i token su cui calcolare la loss.
# Quando TRL chiama apply_chat_template(..., return_assistant_tokens_mask=True),
# riceve una maschera binaria: 1 per i token dentro {% generation %}, 0 per gli altri.
#
# PERCHÉ <|im_end|> e NON {{ eos_token }}:
#   - <|im_end|> ha significato semantico preciso: "fine turno assistant"
#   - </s> (eos nativo) è ambiguo: significa sia "fine turno" che "fine sequenza"
#   - Impostando eos_token="<|im_end|>" in SFTConfig, il modello impara a stoppare lì
#   - È lo standard ChatML usato da Qwen, Mistral, ecc
tokenizer_student.chat_template = (
    "{% for message in messages %}"
        "<|im_start|>{{ message['role'] }}\n"
        "{% if message['role'] == 'assistant' %}"
            "{% generation %}"
            "{{ message['content'] }}<|im_end|>\n"
            "{% endgeneration %}"
        "{% else %}"
            "{{ message['content'] }}<|im_end|>\n"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
        "<|im_start|>assistant\n"
    "{% endif %}"
)

    
# ── Caricamento modello Student ───────────────────
model = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL_ID,
    dtype=torch.float32,
    device_map="auto"
)

# Resize OBBLIGATORIO dopo add_special_tokens
# Allinea la matrice di embedding (vocab_size × hidden_dim) alla nuova dimensione del vocabolario.
# I nuovi embedding sono inizializzati con la media degli embedding esistenti (comportamento default).
if special_tokens_to_add:
    model.resize_token_embeddings(len(tokenizer_student))
    print(f"Embedding matrix ridimensionata a {len(tokenizer_student)} token")



# Se il dataset non esiste in memoria allora cerca se esiste su disco, altrimenti solleva un errore
if 'distilled_dataset' in globals():
    pc_dataset = distilled_dataset.map(format_example, remove_columns=distilled_dataset.column_names)
elif os.path.exists(TEACHER_DATASET_DIR):
    distilled_dataset = load_from_disk(TEACHER_DATASET_DIR)
    pc_dataset = distilled_dataset.map(format_example, remove_columns=distilled_dataset.column_names)
else:
    raise ValueError(f"Dataset {TEACHER_DATASET_DIR} non trovato. Assicurati che la generazione del Teacher sia completata correttamente.")


# ── Validation split per monitorare l'overfitting ─────────────────────────────
# 5% = ~500 sample su 10k. Abbastanza per avere un segnale affidabile sulla eval loss.
split      = pc_dataset.train_test_split(test_size=0.05, seed=42)
train_data = split["train"]
eval_data  = split["test"]

print(f"\nDataset: {len(train_data)} train / {len(eval_data)} eval")


# Configurazione con SFTConfig
training_args = SFTConfig(
    output_dir=STUDENT_OUTPUT_DIR,
    per_device_train_batch_size=STUDENT_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION,
    gradient_checkpointing=True,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    logging_steps=10,
    logging_first_step=True,
    num_train_epochs=EPOCHS,
    fp16=True,
    optim="adamw_torch_fused",
    report_to="none",
    max_length=MAX_SEQ_LENGTH,

    # ── LOSS SU ASSISTANT ONLY ────────────────────────────────────────────────
    # assistant_only_loss=True è il flag corretto per dataset con colonna "messages".
    # completion_only_loss=True (originale) è per dataset {"prompt": ..., "completion": ...}.
    # Con il dataset messages e completion_only_loss, la loss viene calcolata
    # sull'intera sequenza (system + user + assistant), diluendo il segnale.
    assistant_only_loss=True,

    # ── EOS TOKEN ALLINEATO AL TEMPLATE ──────────────────────────────────────
    # Il chat template usa <|im_end|> come stop token dell'assistant.
    # SFTConfig deve saperlo per allineare correttamente la loss mask.
    # Senza questo, TRL cerca </s> come fine-risposta ma nel template c'è <|im_end|>.
    eos_token="<|im_end|>",

    # Salva il checkpoint migliore basandosi sulla eval loss
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    dataset_kwargs={
        "skip_prepare_dataset": False
    },
)

# Esecuzione
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    processing_class=tokenizer_student,
)

print("Avvio addestramento...")
trainer.train()

# Salvataggio
trainer.save_model(STUDENT_FINAL_DIR)
tokenizer_student.save_pretrained(STUDENT_FINAL_DIR)
print(f"Modello e tokenizer salvati in {STUDENT_FINAL_DIR}.")

# %% [markdown]
# ## Valutazione
# Confronto **Teacher**, **Student Baseline (zero-shot)** e **Student Distillato**.
# Metriche: ROUGE-1, ROUGE-2, ROUGE-L, BERTScore-F1, Perplexity, Latenza/Token, Parametri.

# %%
def get_prompt_and_target(sample):
    if TASK == "summarization":
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are an expert assistant strictly dedicated to abstractive summarization. "
                    "You must extract the core event, problem, or decision from the conversation. "
                    "RULES: "
                    "1) Do NOT copy, repeat, or quote the dialogue. "
                    "2) Do NOT use dialogue format (e.g., 'Name:'). "
                    "3) Write exactly one or two sentences in the third person."
                )
            },
            {
                "role": "user", 
                "content": (
                    f"Dialogue:\n{sample['dialogue']}\n\n"
                    "Task: Write a brief, third-person narrative summary describing what the people are doing or talking about."
                )
            }
        ]
        return messages, sample["summary"], sample["dialogue"]
    
    elif TASK == "question_answering":
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
                    (f"Context:\n{sample.get('context', '')}\n\n" if sample.get('context') else "") +
                    f"Question:\n{sample['instruction']}\n\n"
                    "Task: Answer the question."
                )
            }
        ]
        return messages, sample["response"], messages[1]["content"]

# %%
# ── Cleanup VRAM/RAM per evitare OOM su riavvio/interruzione ──
clear_memory()

# 1. Setup metriche e hardware
device = "cuda" if torch.cuda.is_available() else "cpu"
rouge_metric = evaluate.load("rouge")
bert_metric = evaluate.load("bertscore")

# Caricamento del dataset di test originale
if TASK == "summarization":
    test_raw = load_dataset("knkarthick/samsum", split="test")
elif TASK == "question_answering":
    full_dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    test_raw = full_dataset.train_test_split(test_size=0.1, seed=42)["test"]

if MAX_TEST_SAMPLES != 'all':
    test_raw = test_raw.select(range(min(int(MAX_TEST_SAMPLES), len(test_raw))))

print(f"Test samples: {len(test_raw)}")

# 2. Funzione di generazione per la valutazione
def evaluate_model(model_path, chat_template=False):
    """
    Valuta un modello su ROUGE, BERTScore, Perplexity e Latenza.

    chat_template=True → inietta il template ChatML (per lo Student distillato e baseline).
    chat_template=False → usa il template nativo del modello (per il Teacher).

    NOTA sulla Perplexity:
      Viene calcolata mascherando il prompt (label=-100 sui token del prompt).
      Misura quanto il modello è "sorpreso" dal target, dato il prompt.
      Più bassa = meglio.

    NOTA sulla generazione:
      stop_tokens include sia eos_token_id che <|im_end|> se presente nel vocabolario.
      Questo garantisce che il modello si fermi correttamente con entrambi i template.
    """
    print(f"\nValutazione: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if chat_template and tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
                "<|im_start|>{{ message['role'] }}\n"
                "{% if message['role'] == 'assistant' %}"
                    "{% generation %}"
                    "{{ message['content'] }}<|im_end|>\n"
                    "{% endgeneration %}"
                "{% else %}"
                    "{{ message['content'] }}<|im_end|>\n"
                "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
                "<|im_start|>assistant\n"
            "{% endif %}"
        )

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32).to(device)
    model.eval()
    model.generation_config.max_length = None # Annulla il limite nativo per evitare conflitti

    predictions, references = [], []
    total_loss, total_time, total_gen_tokens = 0.0, 0.0, 0

    # Stop tokens: ferma la generazione sia su </s> che su <|im_end|> se disponibile
    stop_tokens = [tokenizer.eos_token_id]
    if "<|im_end|>" in tokenizer.vocab:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id != tokenizer.eos_token_id:
            stop_tokens.append(im_end_id)


    for sample in tqdm(test_raw, desc=f"Eval {model_path}"):
        messages, target, _ = get_prompt_and_target(sample)
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        prompt_len = inputs["input_ids"].shape[1]

        # Calcolo perplexity: sequenza completa (prompt + target + eos)
        full_text = prompt + target + tokenizer.eos_token
        full_inputs = tokenizer(full_text, return_tensors="pt").to(device)

        # FIX: calcola prompt_len ri-tokenizzando il solo prompt sulla full_inputs,
        # invece di riusare prompt_len da inputs (tokenizzato separatamente).
        # Con BPE il contesto influenza la segmentazione ai bordi: tokenizzare
        # "prompt" da solo puo produrre N token, ma la stessa stringa all'interno
        # di "prompt + target" puo produrne N+1 o N-1 per via del merging.
        # Ri-tokenizzando solo il prompt con lo stesso offset garantisce che
        # la maschera -100 cada esattamente sui token giusti.
        prompt_only_inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len_full = prompt_only_inputs["input_ids"].shape[1]

        
        # Mascheramento del prompt nelle label
        labels = full_inputs["input_ids"].clone()
        labels[0, :prompt_len_full] = -100 # Maschera il prompt: loss solo sul target

        with torch.no_grad():
            loss_out = model(**full_inputs, labels=labels)
            total_loss += loss_out.loss.item()

            # Generazione per ROUGE/BERTScore
            t0 = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_tokens
            )
            total_time += time.time() - t0

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        total_gen_tokens += len(gen_tokens)
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        predictions.append(gen_text)
        references.append(target)

    rouge_res = rouge_metric.compute(predictions=predictions, references=references)
    bert_res = bert_metric.compute(predictions=predictions, references=references, lang="en")
    avg_bert_f1 = sum(bert_res["f1"]) / len(bert_res["f1"])
    avg_loss = total_loss / len(test_raw)
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    ms_per_tok = (total_time * 1000) / total_gen_tokens if total_gen_tokens > 0 else 0
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    del model
    torch.cuda.empty_cache()

    return {
        "Perplexity":        perplexity,
        "ROUGE-1":           rouge_res["rouge1"],
        "ROUGE-2":           rouge_res["rouge2"],
        "ROUGE-L":           rouge_res["rougeL"],
        "BERTScore-F1":      avg_bert_f1,
        "Latency/Token (ms)": ms_per_tok,
        "Parameters (M)":    params_m,
        "predictions":       predictions,
    }

# 3. Esecuzione dei confronti
metrics_distilled = evaluate_model(STUDENT_FINAL_DIR,  chat_template=False)
metrics_baseline  = evaluate_model(STUDENT_MODEL_ID,   chat_template=True)
metrics_teacher   = evaluate_model(TEACHER_MODEL_ID,   chat_template=False)

# 4. Stampa comparativa dei risultati
print("\n=== RISULTATI COMPARATIVI ===")
for name, m in [("Teacher", metrics_teacher), ("Student Baseline", metrics_baseline), ("Student Distilled", metrics_distilled)]:
    print(f"\n{name}:")
    for k, v in m.items():
        if k != "predictions":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

# %% [markdown]
# ### Grafici 

# %%
import matplotlib.pyplot as plt
import numpy as np

plot_metrics = ["Perplexity", "ROUGE-L", "BERTScore-F1", "Latency/Token (ms)"]
models       = ["Teacher", "Student Baseline", "Student Distilled"]
all_results  = [metrics_teacher, metrics_baseline, metrics_distilled]
colors       = ["steelblue", "lightcoral", "mediumseagreen"]

fig, axes = plt.subplots(1, len(plot_metrics), figsize=(14, 5))
fig.suptitle(f"Knowledge Distillation — Task: {TASK}", fontsize=14, fontweight="bold")

for ax, metric in zip(axes, plot_metrics):
    vals = [r[metric] for r in all_results]
    bars = ax.bar(models, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title(metric, fontsize=11)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(f"kd_results_{TASK}.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Grafico salvato come kd_results_{TASK}.png")

# %% [markdown]
# ### Ispezione Qualitativa
# Stampa di 3 esempi casuali a confronto tra i tre modelli.

# %%
# Caricamento modello distillato in fp16 per l'ispezione qualitativa
tok_qual   = AutoTokenizer.from_pretrained(STUDENT_FINAL_DIR)
model_qual = AutoModelForCausalLM.from_pretrained(STUDENT_FINAL_DIR, dtype=torch.float32, device_map="auto")

samples = random.sample(list(test_raw), min(3, len(test_raw)))

stop_tokens = [tok_qual.eos_token_id]
if "<|im_end|>" in tok_qual.vocab:
    im_end_id = tok_qual.convert_tokens_to_ids("<|im_end|>")
    if im_end_id != tok_qual.eos_token_id:
        stop_tokens.append(im_end_id)

print("=== ISPEZIONE QUALITATIVA DEGLI OUTPUT ===\n")
for i, sample in enumerate(samples, 1):
    messages, target, input_text = get_prompt_and_target(sample)
    prompt = tok_qual.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok_qual(prompt, return_tensors="pt").to(model_qual.device)

    with torch.no_grad():
        outputs = model_qual.generate(
            **inputs, max_new_tokens=128, do_sample=False,
            pad_token_id=tok_qual.eos_token_id,
            eos_token_id=stop_tokens
        )
    gen_text = tok_qual.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)

    print(f"--- SAMPLE {i} ---")
    print(f"INPUT:\n{input_text.strip()}\n")
    print(f"TARGET IDEALE (Human):\n{target.strip()}\n")
    print(f"GENERAZIONE STUDENT DISTILLATO:\n{gen_text.strip()}\n")
    print("STUDENT BASELINE prediction:", metrics_baseline["predictions"][test_raw.to_list().index(sample)] if hasattr(test_raw, 'to_list') else "(esegui con indice)")
    print("=" * 60 + "\n")


