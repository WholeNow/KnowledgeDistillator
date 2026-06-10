# KnowledgeDistillator

Questa repository contiene i codici e i notebook per un progetto di **Knowledge Distillation**. 
L'obiettivo è trasferire la conoscenza da un modello "Teacher" di dimensioni maggiori a un modello "Student" di dimensioni ridotte, affrontando due specifici task di Natural Language Processing (NLP):
1. **Summarization**: Generazione di riassunti (basato sul dataset knkarthick/samsum).
2. **Question Answering**: Generazione di risposte a domande (basato sul dataset Databricks Dolly 15k).

## 🧠 Modelli Utilizzati (Citations)

- **Teacher Model**: [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
  Un modello efficiente da 1.1 miliardi di parametri, veloce da eseguire e performante nella generazione di testo.

- **Student Model**: [HuggingFaceTB/SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M)
  Un modello compatto da soli 135 milioni di parametri, che impara a simulare le capacità del Teacher.

## 📊 Dataset Utilizzati

Il progetto sfrutta due dataset open-source principali per la generazione delle pseudo-label (distillazione) e per il fine-tuning:

### 1. SAMSum Corpus (`knkarthick/samsum`)
Utilizzato per il task di **Summarization**. 
- **Descrizione:** È un dataset progettato per l'astrazione e il riassunto di dialoghi. Contiene conversazioni che simulano chat reali in stile messenger (con tanto di slang, emoticon, typo e linguaggio informale) create appositamente da linguisti.
- **Dimensioni e Formato:** Comprende circa 16.000 istanze (divise in train, validation e test). Ogni record è composto dai campi `id`, `dialogue` (il testo completo della conversazione con i nomi degli interlocutori) e `summary` (un riassunto conciso scritto in terza persona da annotatori umani).

### 2. Databricks Dolly 15k (`databricks/databricks-dolly-15k`)
Utilizzato per il task di **Question Answering**.
- **Descrizione:** Un corpus di alta qualità contenente oltre 15.000 record generati manualmente dai dipendenti di Databricks. È stato specificamente ideato per addestrare i Large Language Models a seguire istruzioni umane (instruction-following).
- **Dimensioni e Formato:** I dati si dividono in varie categorie. Ogni record presenta i campi `instruction` (la domanda o il task da eseguire), `context` (eventuali informazioni di contesto o brani di supporto), `response` (la risposta corretta fornita da un umano) e `category` (che classifica il task, come *closed QA*, *open QA*, *brainstorming*, ecc.).

## 📂 Struttura del Codice

- `Summarization.ipynb`: Notebook Jupyter contenente la pipeline completa (scaricamento, generazione, e fine-tuning) per il task di riassunto.
- `Question_Answering.ipynb`: Notebook Jupyter per il task di Question Answering, basato sul Dolly 15k.
- `Interactive_Chat.ipynb`: Notebook interattivo per testare liberamente in stile chat il modello distillato che hai addestrato.
- `scripts/`: Cartella che contiene script utili per l'elaborazione dei dataset salvati.
- `results/`: Cartella destinata al salvataggio dei risultati, dei grafici e dei pesi del modello.
- `requirements.txt`: Elenco delle dipendenze necessarie per l'ambiente di sviluppo.

---

## 🚀 Guida all'uso su Google Colab

Tutto il codice è stato concepito per essere facilmente eseguibile su Google Colab in modo da sfruttare l'accelerazione hardware (GPU).

### 1. Avvio Rapido
Puoi aprire direttamente i notebook su Google Colab con un click usando i pulsanti qui sotto:

- **Summarization Pipeline**:  
  <a target="_blank" href="https://colab.research.google.com/github/WholeNow/KnowledgeDistillator/blob/main/Summarization.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- **Question Answering Pipeline**:  
  <a target="_blank" href="https://colab.research.google.com/github/WholeNow/KnowledgeDistillator/blob/main/Question_Answering.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- **Interactive Chat**:  
  <a target="_blank" href="https://colab.research.google.com/github/WholeNow/KnowledgeDistillator/blob/main/Interactive_Chat.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Una volta aperto il notebook desiderato, abilita la GPU:
- Vai su **Runtime -> Cambia tipo di runtime**.
- Sotto "Acceleratore hardware", seleziona **T4 GPU** o **A100 GPU** e clicca su *Salva*.

### 2. Installazione delle Dipendenze
In Colab, esegui la cella di installazione iniziale o, se cloni la repository, esegui:
```bash
!pip install -r requirements.txt
```

### 3. Esecuzione della Pipeline
Una volta installate le dipendenze e configurato l'ambiente, procedi eseguendo le celle dei notebook in ordine. I notebook si occuperanno automaticamente di:
- Caricare il dataset base.
- Eseguire l'inferenza con il modello Teacher per generare le "pseudo-label" (distillazione).
- Allenare il modello Student (`SFTTrainer`) sulle risposte generate dal Teacher.
- Valutare i risultati (es. metriche ROUGE, BERTScore, o per QA tramite accuratezza e generazione testuale).

---

## 🛠 Script di Utilità (`scripts/`)

Abbiamo inserito degli script per poter manipolare facilmente i dataset salvati post-generazione. Si possono eseguire da riga di comando (anche in una cella di Colab anteponendo il punto esclamativo `!`).

### `export_dataset_to_csv.py`
Permette di convertire un dataset generato da HuggingFace in un file CSV (comodo per la visualizzazione su Excel).

**Utilizzo:**
```bash
python scripts/export_dataset_to_csv.py --dataset_path ./percorso_del_dataset --output_csv dataset_finale.csv
```

### `normalize_teacher_summaries.py`
Questo script normalizza l'output del modello Teacher, che solitamente contiene il tag `<|assistant|>`, estraendo solamente il testo vero e proprio e salvando una versione "pulita" del dataset.

**Utilizzo:**
```bash
python scripts/normalize_teacher_summaries.py --dataset_path ./dataset_distilled_qa_teacher
```
Verrà automaticamente creato un nuovo dataset nominato `<nome_originale>_normalized`.