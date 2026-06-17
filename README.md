# SmolLM-KD

**Knowledge Distillation** project for the Deep Learning and Applications course: the responses of a Teacher model are used as pseudo-labels to train a smaller Student model on two tasks:

- **Summarization** — summarizing dialogues (SAMSum dataset)
- **Question Answering** — answering questions (Databricks Dolly 15k dataset)

---

## Table of Contents

1. [How it works](#how-it-works)
2. [Models](#models)
3. [Datasets](#datasets)
4. [Project Structure](#project-structure)
5. [Results](#results)
6. [Guide: Training](#guide-training)
7. [Guide: Chat](#guide-chat)
8. [Utility Scripts](#utility-scripts)

---

## How it works

```
Original Dataset
      │
      ▼
┌─────────────┐    pseudo-label    ┌─────────────────┐
│   Teacher   │ ─────────────────► │ Distilled       │
│  (1.1B–4B)  │                    │ Dataset         │
└─────────────┘                    └────────┬────────┘
                                            │  fine-tuning (SFT)
                                            ▼
                                   ┌─────────────────┐
                                   │    Student      │
                                   │    (135M)       │
                                   └─────────────────┘
```

1. The **Teacher** (large model) reads each example from the dataset and generates a response.
2. These responses become the **pseudo-labels** of the distilled dataset.
3. The **Student** (small model) is fine-tuned on this dataset with SFT.
4. The result is a Student that approaches the capabilities of the Teacher while having 8× fewer parameters.

---

## Models

| Role | Model | Parameters |
|---|---|---|
| **Teacher** (option A) | [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | 1.1B |
| **Teacher** (option B) | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | 1.5B |
| **Teacher** (option C) | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | 4B |
| **Student** | [HuggingFaceTB/SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M) | 135M |

> On a T4 GPU (free Colab), `TinyLlama` without quantization is recommended, or `Qwen2.5` or `Qwen3` with `QUANTIZE_TEACHER = True` (4-bit NF4).

---

## Datasets

### SAMSum (`knkarthick/samsum`) — Summarization
Chat-style dialogues with summaries written by human annotators. ~16,000 examples, fields: `dialogue`, `summary`.

### Databricks Dolly 15k (`databricks/databricks-dolly-15k`) — Question Answering
15,000 questions and answers written by Databricks employees. Fields: `instruction` (question), `context` (supporting text, optional), `response` (answer), `category`.

---

## Project Structure

```
KnowledgeDistillator/
│
├── training.ipynb          ← complete pipeline: Teacher + Student + Evaluation
├── Chat.ipynb              ← interactive chat with the distilled model
├── push_to_hub.py          ← script to upload models to HuggingFace
├── requirements.txt
│
├── scripts/
│   ├── export_dataset_to_csv.py       ← exports HF dataset to CSV
│   ├── normalize_teacher_summaries.py ← removes <|assistant|> tags from Teacher output
│   └── plot_metrics.py                ← comparative metric charts
│
└── results/
    ├── question_answering/
    │   ├── prompt_1/
    │   │   ├── model/            ← final Student model
    │   │   ├── checkpoints/      ← intermediate training checkpoints
    │   │   └── dataset_teacher/  ← dataset distilled by the Teacher
    │   ├── prompt_2/
    │   └── prompt_3/
    └── summarization/
        ├── prompt_1/
        ├── prompt_2/
        └── prompt_3/
```

---

## Results

Metrics of the **Distilled Student** (SmolLM 135M) evaluated on 150 samples from the test set. Teacher: TinyLlama 1.1B.

### Summarization — SAMSum

| Prompt | Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F1 |
|:---:|---|:---:|:---:|:---:|:---:|
| 1 — negations | [Marchisceddu/smollm-sum-prompt1](https://huggingface.co/Marchisceddu/smollm-sum-prompt1) | 0.1777 | 0.0525 | 0.1405 | 0.8599 |
| 2 — direct | [Marchisceddu/smollm-sum-prompt2](https://huggingface.co/Marchisceddu/smollm-sum-prompt2) | 0.1789 | 0.0494 | 0.1364 | 0.8534 |
| **3 — minimal** | [Marchisceddu/smollm-sum-prompt3](https://huggingface.co/Marchisceddu/smollm-sum-prompt3) | **0.2440** | **0.0667** | **0.1857** | **0.8683** |

### Question Answering — Dolly 15k

| Prompt | Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F1 |
|:---:|---|:---:|:---:|:---:|:---:|
| **1 — negations** | [Marchisceddu/smollm-qa-prompt1](https://huggingface.co/Marchisceddu/smollm-qa-prompt1) | **0.2594** | **0.1040** | **0.1992** | **0.8540** |
| 2 — direct | [Marchisceddu/smollm-qa-prompt2](https://huggingface.co/Marchisceddu/smollm-qa-prompt2) | 0.2468 | 0.0978 | 0.1884 | 0.8507 |
| 3 — minimal | [Marchisceddu/smollm-qa-prompt3](https://huggingface.co/Marchisceddu/smollm-qa-prompt3) | 0.2441 | 0.0936 | 0.1830 | 0.8506 |

> All models are public and are automatically loaded in `Chat.ipynb` without a token.

---

## Guide: Training

### 1. Open the notebook in Colab

<a target="_blank" href="https://colab.research.google.com/github/WholeNow/KnowledgeDistillator/blob/main/training.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### 2. Enable GPU

**Runtime → Change runtime type → T4 GPU** (or A100 if available) → Save.

### 3. Configure parameters

In the **CONFIGURATION** cell, modify the values according to your experiment:

```python
TASK        = "summarization"               # "summarization" | "question_answering"
PROMPT_TYPE = 3                             # 1 | 2 | 3  (see table below)

TEACHER_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
QUANTIZE_TEACHER = True                     # True → 4-bit NF4 (saves VRAM)

STUDENT_MODEL_ID = "HuggingFaceTB/SmolLM-135M"

MAX_TRAIN_SAMPLES = '10000'                 # samples for Teacher generation
MAX_TEST_SAMPLES  = '150'                   # samples for final evaluation
```

**Prompt Types:**

| `PROMPT_TYPE` | Style | When to use it |
|:---:|---|---|
| `1` | Prompt with negations and explicit rules | Model tends to copy the dialogue |
| `2` | Prompt with direct questions and penalties | Maximum quality required from Teacher |
| `3` | Minimal prompt | Fast, ideal for baseline |

### 4. Run the cells in order

The notebook is divided into three main sections:

**Section 1 — Distilled dataset generation (Teacher)**
- Loads the original dataset.
- Builds the prompts with the chosen type.
- The Teacher generates the pseudo-labels for each example.
- The distilled dataset is saved to disk in `results/{TASK}/prompt_{PROMPT_TYPE}/dataset_teacher/`.
- If the dataset already exists, it is loaded directly (no regeneration).

**Section 2 — Student Training**
- Loads the distilled dataset.
- Adds the special tokens `<|im_start|>` and `<|im_end|>` to the tokenizer.
- Injects the ChatML template.
- Starts fine-tuning with `SFTTrainer` (loss only on assistant tokens).
- Saves the final model in `results/{TASK}/prompt_{PROMPT_TYPE}/model/`.

**Section 3 — Evaluation**
- Compares **Teacher**, **Student Baseline** (zero-shot) and **Distilled Student**.
- Calculated metrics: ROUGE-1, ROUGE-2, ROUGE-L, BERTScore-F1, Perplexity, Latency/Token.
- Generates a comparative chart and saves it as `kd_results_{TASK}.png`.
- Prints 3 random qualitative examples.

### 5. Save the model

At the end of training, the model is already saved locally in the Colab session.

---

## Guide: Chat

### 1. Open the notebook in Colab

<a target="_blank" href="https://colab.research.google.com/github/WholeNow/KnowledgeDistillator/blob/main/Chat.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### 2. Enable GPU

**Runtime → Change runtime type → T4 GPU** → Save.

### 3. Configure task, prompt, and model

In the **CONFIGURATION** cell:

```python
TASK        = "question_answering"   # "summarization" | "question_answering"
PROMPT_TYPE = 1                      # must match the one used in training

MODELS = {
    # (task,                prompt_type): "HuggingFace ID or local path"
    ("question_answering",  1): "Marchisceddu/smollm-qa-prompt1",   # ← public model
    ("question_answering",  2): "",                                   # ← not available
    ("summarization",       3): "Marchisceddu/smollm-sum-prompt3",
    # ...
}
```

> If `MODELS[(TASK, PROMPT_TYPE)]` is empty, the notebook automatically looks for the model locally (`./results/student_distilled_{TASK}_final`), useful if you just finished training in the same session.

Models with HuggingFace IDs are **downloaded automatically**, without a token.

### 4. Run all cells

The model loads in ~30 seconds on a T4. Then the interactive chat starts in the last cell.

### 5. Use the chat

**Question Answering** — the chat asks two separate questions:
```
You: What is the boiling point of water?
Context (optional):                        ← press Enter to skip
Model: The boiling point of water is 100°C at sea level.
```

With context:
```
You: What is the capital of France?
Context (optional): France is a country in Western Europe. Its capital is Paris.
Model: The capital of France is Paris.
```

**Summarization** — paste the dialogue on multiple lines, then press Enter on an empty line:
```
You (dialogue): Hannah: Hey, are you free tonight?
                Peter: Yeah, what's up?
                Hannah: Let's grab dinner at that new place downtown.
                Peter: Sounds great, 7pm?
                Hannah: Perfect!
                                             ← empty line to send
Model: Hannah and Peter make plans to have dinner together at a new restaurant downtown.
```

Type `exit` or `quit` to end the chat.

---

## Utility Scripts

The scripts in `scripts/` are run from the command line (or from Colab with `!`).

### `export_dataset_to_csv.py`
Converts a HuggingFace dataset saved on disk to a CSV file.
```bash
python scripts/export_dataset_to_csv.py \
  --dataset_path ./results/QA/prompt_1/dataset_teacher \
  --output_csv dataset_qa_p1.csv
```

### `normalize_teacher_summaries.py`
Removes any `<|assistant|>` tags or artifacts from the Teacher's output and saves a clean version of the dataset.
```bash
python scripts/normalize_teacher_summaries.py \
  --dataset_path ./results/QA/prompt_1/dataset_teacher
```

### `plot_metrics.py`
Generates comparative metric charts (ROUGE, BERTScore, Perplexity) from the saved results.
```bash
python scripts/plot_metrics.py --results_dir ./results/QA
```
