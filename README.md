# Legal text contradiction detection: hyperparameter gridsearching and summarisation

## Converting the Excel datasets to CSV

`convert_datasets.py` converts the local `train.xlsx` and `val.xlsx` datasets
into CSV files for both supported tasks:

- ternary: `contradiction`, `entailment`, and `not mentioned`;
- binary: `contradiction` and `no`, with `entailment` and `not mentioned`
  merged into `no`.

The script requires Python 3.10 or newer. Its dependencies are declared in
`pyproject.toml`; install them into a local virtual environment with:

```bash
uv sync --dev
```

Run the conversion from the repository root:

```bash
uv run python convert_datasets.py
```

By default, this reads `train.xlsx` and `val.xlsx` and writes
`train_ternary.csv`, `train_binary.csv`, `val_ternary.csv`, and
`val_binary.csv` in the current directory. Each output preserves the input row
order and the `premise`, `hypothesis`, `source`, and `tag` columns. Alternative
locations can be supplied explicitly:

```bash
uv run python convert_datasets.py \
  --train path/to/train.xlsx \
  --val path/to/val.xlsx \
  --output-dir path/to/csv
```

The same workflow is available from Python through the documented public
interface:

```python
from convert_datasets import run_conversion

output_paths = run_conversion(
    train_path="train.xlsx",
    val_path="val.xlsx",
    output_dir=".",
)
```

This is a local data-preparation utility: it does not require Colab, a GPU, or
GPU/memory configuration.

## Colab experiment workflows

The Python package replaces the executable logic in the legacy notebooks. Its
imports are side-effect free; cloning data, mounting Drive, loading models,
training, validation, document upload, and inference happen only when a public
workflow function is called.

### Colab setup

Start a GPU runtime, then clone and install the repository:

```python
!git clone https://github.com/fortvivlan/JURA_hypersumm.git /content/JURA_hypersumm
%cd /content/JURA_hypersumm
%pip install -q -e ".[colab]"
```

The package reads `train_binary.csv`, `train_ternary.csv`, `val_binary.csv`,
and `val_ternary.csv` directly from this clone. At inference time it clones
`https://github.com/fortvivlan/dms-rag` to `/content/dms-rag` if needed and
uses that repository's `codex.csv` and `faiss_index/` artifacts.

Create a private Colab secret named `HF_TOKEN` before using a gated Hugging
Face model such as Llama, and accept that model's license on Hugging Face. The
token is read from Colab secrets (or the `HF_TOKEN` environment variable) and
is never printed. A token previously embedded in the legacy notebooks must be
considered compromised and revoked.

The existing Drive folder is expected at `/content/drive/MyDrive/jura`.
Only final trained BERT models and final LoRA adapters are written there;
checkpoints, caches, score files, and other temporary artifacts remain in the
Colab runtime.

### BERT baseline

The public interfaces train one task, select the best validation macro-F1
state, save the final model, run final validation, ask for one or more `.docx`
court decisions, and execute exact-citation/FAISS inference:

```python
from jura_hypersumm.bert import run_bert_binary, run_bert_ternary

binary_scores = run_bert_binary()
ternary_scores = run_bert_ternary()
```

Models are saved to `MyDrive/jura/models/bert/binary` and
`MyDrive/jura/models/bert/ternary`. The default is
`ai-forever/sbert_large_nlu_ru` with length 512, batch size 16, six epochs,
and learning rate `2e-5`. Memory-sensitive runtimes can override settings:

```python
binary_scores = run_bert_binary(
    hyperparameters={
        "batch_size": 4,
        "inference_batch_size": 16,
        "gradient_checkpointing": True,
        "precision": "auto",
    }
)
```

### Ready-LLM evaluation

`run_llm_evaluation(model_name)` evaluates one supported ready model on both
binary and ternary validation sets, then tests both task prompts on one set of
uploaded documents:

```python
from jura_hypersumm.llm_evaluation import run_llm_evaluation

scores = run_llm_evaluation("ministral")
```

Run it once per experiment model. Accepted aliases and full model IDs are:

- `llama` — `meta-llama/Llama-3.1-8B`;
- `ministral` — `mistralai/Ministral-8B-Instruct-2410`;
- `qwen` — `Qwen/Qwen3-8B`;
- `t-lite` — `t-tech/T-lite-it-2.1`.

Models load in 4-bit NF4 by default. Important memory knobs are
`batch_size`, `document_batch_size`, `max_input_length`, `device_map`,
`precision`, and `quantization`:

```python
scores = run_llm_evaluation(
    "qwen",
    inference_parameters={"batch_size": 1, "document_batch_size": 4},
)
```

### LoRA/QLoRA fine-tuning

`run(model_name, task, hyperparameters=None)` trains one task adapter for one
of the same four models, stores the final adapter, validates it, and performs
full document inference:

```python
from jura_hypersumm.lora import run

scores = run("ministral", "ternary")
```

Final adapters are stored at
`MyDrive/jura/models/lora/<model_slug>/<binary-or-ternary>`. The default QLoRA
configuration uses 4-bit NF4, rank 16, alpha 32, dropout 0.05, sequence length
1024, batch size 2, gradient accumulation 8, three epochs, and learning rate
`2e-4`. Pass a partial dictionary to change the common hyperparameters:

```python
scores = run(
    "qwen",
    "binary",
    hyperparameters={
        "lora_rank": 32,
        "lora_alpha": 64,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "epochs": 2,
        "learning_rate": 1e-4,
    },
)
```

Unknown hyperparameter names raise an error instead of being ignored.

### Results and uploaded documents

Every public workflow returns and displays its main `pandas.DataFrame` score
table. It also downloads one XLSX workbook containing summary and per-class
metrics, a confusion matrix, raw validation predictions, document-level
aggregates, every retrieved premise/pair prediction, errors, and reproducibility
metadata. Results are written under `/content/jura_results`, not Drive.

When at least one document is successfully analysed, the workflow also
downloads a ZIP prepared for human review. For every document it contains:

- `<document>_<task>_model_predictions.xlsx`, with every RAG premise/model pair
  and the columns `hypothesis`, `premise`, `model_prediction`, `expert_label`,
  and `expert_comment`. A specialist can fill the last two columns so these
  predictions can later be scored against human labels.
- `<document>_rag_retrieval.xlsx`, with exactly `sentence`, `article_number`,
  and `article_text`. It contains the top-ranked article used for each
  processed sentence and is intended for separate manual RAG evaluation.

LoRA and BERT runs produce one model-prediction workbook per document. A ready
LLM run produces binary and ternary model-prediction workbooks for each
document, while sharing one deduplicated RAG workbook. The detailed main
workbook still retains all retrieval candidates and ranks; only the compact
RAG review workbook is limited to the top-ranked article per sentence.

Document inference extracts the final `ПОСТАНОВИЛ` section, splits it with
`razdel`, performs deterministic citation lookup before a maximum of 20 FAISS
matches, and preserves the premise responsible for every contradiction. A
document without `ПОСТАНОВИЛ` is reported and skipped. Uploaded `.docx` files
are isolated in a temporary directory and deleted after processing even if an
error occurs.
