# SmolLM-KD
Progetto di **Knowledge Distillation** per il corso di Deep Learning and Applications: le risposte di un modello Teacher vengono usate come pseudo-label per addestrare un modello Student più piccolo, su due task:

- **Summarization** — riassunto di dialoghi (dataset SAMSum)
- **Question Answering** — risposta a domande (dataset Databricks Dolly 15k)

---

## Indice

1. [Come funziona](#come-funziona)
2. [Modelli](#modelli)
3. [Dataset](#dataset)
4. [Struttura del progetto](#struttura-del-progetto)
5. [Risultati](#risultati)
6. [Guida: Training](#guida-training)
7. [Guidao: Chat](#guida-chat)
8. [Script di utilità](#script-di-utilità)

---

## Come funziona

```
Dataset originale
      │
      ▼
┌─────────────┐    pseudo-label    ┌─────────────────┐
│   Teacher   │ ─────────────────► │  Dataset        │
│  (1.1B–4B)  │                    │  Distillato     │
└─────────────┘                    └────────┬────────┘
                                            │  fine-tuning (SFT)
                                            ▼
                                   ┌─────────────────┐
                                   │    Student      │
                                   │    (135M)       │
                                   └─────────────────┘
```

1. Il **Teacher** (modello grande) legge ogni esempio del dataset e genera una risposta.
2. Queste risposte diventano le **pseudo-label** del dataset distillato.
3. Lo **Student** (modello piccolo) viene fine-tunato su questo dataset con SFT.
4. Il risultato è uno Student che si avvicina alle capacità del Teacher pur avendo 8× meno parametri.

---

## Modelli

| Ruolo | Modello | Parametri |
|---|---|---|
| **Teacher** (opzione A) | [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | 1.1B |
| **Teacher** (opzione B) | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | 1.5B |
| **Teacher** (opzione C) | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | 4B |
| **Student** | [HuggingFaceTB/SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M) | 135M |

> Su GPU T4 (Colab gratuito) si consiglia `TinyLlama` senza quantizzazione, oppure `Qwen2.5` o `Qwen3` con `QUANTIZE_TEACHER = True` (4-bit NF4).

---

## Dataset

### SAMSum (`knkarthick/samsum`) — Summarization
Dialoghi in stile chat con riassunti scritti da annotatori umani. ~16.000 esempi, campi: `dialogue`, `summary`.

### Databricks Dolly 15k (`databricks/databricks-dolly-15k`) — Question Answering
15.000 domande con risposta scritte dai dipendenti Databricks. Campi: `instruction` (domanda), `context` (testo di supporto, opzionale), `response` (risposta), `category`.

---

## Struttura del progetto

```
KnowledgeDistillator/
│
├── training.ipynb          ← pipeline completa: Teacher + Student + Valutazione
├── Chat.ipynb              ← chat interattiva con il modello distillato
├── push_to_hub.py          ← script per caricare i modelli su HuggingFace
├── requirements.txt
│
├── scripts/
│   ├── export_dataset_to_csv.py       ← esporta dataset HF in CSV
│   ├── normalize_teacher_summaries.py ← rimuove tag <|assistant|> dall'output Teacher
│   └── plot_metrics.py                ← grafici comparativi delle metriche
│
└── results/
    ├── question_answering/
    │   ├── prompt_1/
    │   │   ├── model/            ← modello Student finale
    │   │   ├── checkpoints/      ← checkpoint intermedi del training
    │   │   └── dataset_teacher/  ← dataset distillato dal Teacher
    │   ├── prompt_2/
    │   └── prompt_3/
    └── summarization/
        ├── prompt_1/
        ├── prompt_2/
        └── prompt_3/
```

---

## Risultati

Metriche del **Student Distillato** (SmolLM 135M) valutate su 150 campioni del test set. Teacher: TinyLlama 1.1B.

### Summarization — SAMSum

| Prompt | Modello | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F1 |
|:---:|---|:---:|:---:|:---:|:---:|
| 1 — negazioni | [Marchisceddu/smollm-sum-prompt1](https://huggingface.co/Marchisceddu/smollm-sum-prompt1) | 0.1777 | 0.0525 | 0.1405 | 0.8599 |
| 2 — diretto | [Marchisceddu/smollm-sum-prompt2](https://huggingface.co/Marchisceddu/smollm-sum-prompt2) | 0.1789 | 0.0494 | 0.1364 | 0.8534 |
| **3 — minimale** | [Marchisceddu/smollm-sum-prompt3](https://huggingface.co/Marchisceddu/smollm-sum-prompt3) | **0.2440** | **0.0667** | **0.1857** | **0.8683** |

### Question Answering — Dolly 15k

| Prompt | Modello | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F1 |
|:---:|---|:---:|:---:|:---:|:---:|
| **1 — negazioni** | [Marchisceddu/smollm-qa-prompt1](https://huggingface.co/Marchisceddu/smollm-qa-prompt1) | **0.2594** | **0.1040** | **0.1992** | **0.8540** |
| 2 — diretto | [Marchisceddu/smollm-qa-prompt2](https://huggingface.co/Marchisceddu/smollm-qa-prompt2) | 0.2468 | 0.0978 | 0.1884 | 0.8507 |
| 3 — minimale | [Marchisceddu/smollm-qa-prompt3](https://huggingface.co/Marchisceddu/smollm-qa-prompt3) | 0.2441 | 0.0936 | 0.1830 | 0.8506 |

> Tutti i modelli sono pubblici e si caricano automaticamente in `Chat.ipynb` senza token.

---

## Guida: Training

### 1. Apri il notebook su Colab

<a target="_blank" href="https://colab.research.google.com/github/WholeNow/KnowledgeDistillator/blob/main/training.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### 2. Abilita la GPU

**Runtime → Cambia tipo di runtime → T4 GPU** (o A100 se disponibile) → Salva.

### 3. Configura i parametri

Nella cella **CONFIGURAZIONE**, modifica i valori in base al tuo esperimento:

```python
TASK        = "summarization"               # "summarization" | "question_answering"
PROMPT_TYPE = 3                             # 1 | 2 | 3  (vedi tabella sotto)

TEACHER_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
QUANTIZE_TEACHER = True                     # True → 4-bit NF4 (risparmia VRAM)

STUDENT_MODEL_ID = "HuggingFaceTB/SmolLM-135M"

MAX_TRAIN_SAMPLES = '10000'                 # campioni per la generazione Teacher
MAX_TEST_SAMPLES  = '150'                   # campioni per la valutazione finale
```

**Tipi di prompt:**

| `PROMPT_TYPE` | Stile | Quando usarlo |
|:---:|---|---|
| `1` | Prompt con negazioni e regole esplicite | Modello tende a copiare il dialogo |
| `2` | Prompt con domande dirette e penalità | Massima qualità richiesta al Teacher |
| `3` | Prompt minimale | Veloce, ideale per baseline |

### 4. Esegui le celle in ordine

Il notebook è diviso in tre sezioni principali:

**Sezione 1 — Generazione dataset distillato (Teacher)**
- Carica il dataset originale.
- Costruisce i prompt con il tipo scelto.
- Il Teacher genera le pseudo-label per ogni esempio.
- Il dataset distillato viene salvato su disco in `results/{TASK}/prompt_{PROMPT_TYPE}/dataset_teacher/`.
- Se il dataset esiste già, viene caricato direttamente (nessuna rigenerazione).

**Sezione 2 — Training dello Student**
- Carica il dataset distillato.
- Aggiunge i token speciali `<|im_start|>` e `<|im_end|>` al tokenizer.
- Inietta il template ChatML.
- Avvia il fine-tuning con `SFTTrainer` (loss solo sui token dell'assistant).
- Salva il modello finale in `results/{TASK}/prompt_{PROMPT_TYPE}/model/`.

**Sezione 3 — Valutazione**
- Confronta **Teacher**, **Student Baseline** (zero-shot) e **Student Distillato**.
- Metriche calcolate: ROUGE-1, ROUGE-2, ROUGE-L, BERTScore-F1, Perplexity, Latenza/Token.
- Genera un grafico comparativo e lo salva come `kd_results_{TASK}.png`.
- Stampa 3 esempi qualitativi casuali.

### 5. Salva il modello

Al termine del training, il modello è già salvato localmente nella sessione Colab.

---

## Guida: Chat

### 1. Apri il notebook su Colab

<a target="_blank" href="https://colab.research.google.com/github/WholeNow/KnowledgeDistillator/blob/main/Chat.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### 2. Abilita la GPU

**Runtime → Cambia tipo di runtime → T4 GPU** → Salva.

### 3. Configura task, prompt e modello

Nella cella **CONFIGURAZIONE**:

```python
TASK        = "question_answering"   # "summarization" | "question_answering"
PROMPT_TYPE = 1                      # deve corrispondere a quello usato nel training

MODELS = {
    # (task,                prompt_type): "ID HuggingFace o percorso locale"
    ("question_answering",  1): "Marchisceddu/smollm-qa-prompt1",   # ← modello pubblico
    ("question_answering",  2): "",                                   # ← non disponibile
    ("summarization",       3): "Marchisceddu/smollm-sum-prompt3",
    # ...
}
```

> Se `MODELS[(TASK, PROMPT_TYPE)]` è vuoto, il notebook cerca automaticamente il modello in locale (`./results/student_distilled_{TASK}_final`), utile se hai appena finito il training nella stessa sessione.

I modelli con ID HuggingFace vengono **scaricati automaticamente**, senza token.

### 4. Esegui tutte le celle

Il modello si carica in ~30 secondi su T4. Poi parte la chat interattiva nell'ultima cella.

### 5. Usa la chat

**Question Answering** — la chat fa due domande separate:
```
Tu: What is the boiling point of water?
Context (opzionale):                       ← premi Invio per saltare
Modello: The boiling point of water is 100°C at sea level.
```

Con contesto:
```
Tu: What is the capital of France?
Context (opzionale): France is a country in Western Europe. Its capital is Paris.
Modello: The capital of France is Paris.
```

**Summarization** — incolla il dialogo su più righe, poi premi Invio su una riga vuota:
```
Tu (dialogo): Hannah: Hey, are you free tonight?
              Peter: Yeah, what's up?
              Hannah: Let's grab dinner at that new place downtown.
              Peter: Sounds great, 7pm?
              Hannah: Perfect!
                                           ← riga vuota per inviare
Modello: Hannah and Peter make plans to have dinner together at a new restaurant downtown.
```

Scrivi `exit` o `quit` per terminare la chat.

---

## Script di utilità

Gli script in `scripts/` si eseguono da riga di comando (o da Colab con `!`).

### `export_dataset_to_csv.py`
Converte un dataset HuggingFace salvato su disco in un file CSV.
```bash
python scripts/export_dataset_to_csv.py \
  --dataset_path ./results/QA/prompt_1/dataset_teacher \
  --output_csv dataset_qa_p1.csv
```

### `normalize_teacher_summaries.py`
Rimuove eventuali tag `<|assistant|>` o artefatti dall'output del Teacher e salva una versione pulita del dataset.
```bash
python scripts/normalize_teacher_summaries.py \
  --dataset_path ./results/QA/prompt_1/dataset_teacher
```

### `plot_metrics.py`
Genera grafici comparativi delle metriche (ROUGE, BERTScore, Perplexity) a partire dai risultati salvati.
```bash
python scripts/plot_metrics.py --results_dir ./results/QA
```
