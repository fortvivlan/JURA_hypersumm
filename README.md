# Legal text contradiction detection: RAG, hyperparameter gridsearching and summarisation

## Legal SBERT RAG experiment

`run_rag_experiment_local.py` exposes `run_rag_experiment(...)`, the complete
sentence-embedding experiment. It reads the original ternary `train.xlsx` and
`val.xlsx`, maps `contradiction` and `entailment` to `similar`, keeps
`not mentioned` as the negative class, fine-tunes
`ai-forever/sbert_large_nlu_ru` with contrastive loss, rebuilds a matching
FAISS index, and compares the unchanged baseline RAG with the tuned version.
Semantic FAISS candidates can then be reranked with
`Alibaba-NLP/gte-multilingual-reranker-base`, a multilingual cross-encoder
that supports Russian. Deterministic citation matches always stay first in the
pipeline and bypass both reranking and semantic top-k truncation.

Required inputs are `train.xlsx`, `val.xlsx`, `dms-rag/codex.csv` and its
baseline `faiss_index/`, `rag_tests/RAG_DIALOGUE_test.xlsx`,
`rag_tests/RAG_FULL_test.xlsx`,
`rag_tests/RAG_FULL_additional_test.xlsx`, and the paired
`test_docx/{Dialogue,Full}` folders. Install the GPU dependencies before
running it:

```powershell
uv sync --extra colab --dev
uv run python run_rag_experiment_local.py `
  --experiment-id sbert_legal_v1 `
  --batch-size 8 `
  --gradient-accumulation 4 `
  --embedding-device cuda `
  --index-device cpu `
  --reranker-mode pretrained
```

Every run evaluates candidate/final depths `20:10` and `40:20`. Override the
matrix with one or more repeatable arguments such as
`--retrieval-depth 100:20`; every depth must be positive with final no greater
than candidate.
The CLI also exposes reranker train/eval batch sizes, gradient accumulation,
precision, maximum sequence length, learning rate, epochs, device, and
gradient checkpointing for sub-24GB GPUs.

The pretrained reranker is the default. This produces baseline/tuned embedding
variants with and without that reranker. To fine-tune it on the same legal
pairs, run a separate experiment ID with `--reranker-mode finetuned`; this
retains the pretrained comparisons and adds baseline/tuned variants using the
fine-tuned reranker.
Fine-tuning uses `hypothesis` as query, `premise` as document, maps
`contradiction`/`entailment` to relevance 1 and `not mentioned` to 0, and
selects the best checkpoint by validation average precision. The default GTE
model uses Hugging Face custom model code, so the workflow enables
`trust_remote_code` for this exact model after resolving an immutable revision;
alternative models require `--reranker-trust-remote-code` when needed.

The trained encoder, rebuilt index, `run_config.json`, and
`rag_manifest.json` are written below
`local_artifacts/rag/sbert_legal_v1/`. Converted datasets, validation
similarities, and compact retrieval Recall are written below
`local_results/rag/sbert_legal_v1/`. Pretrained and fine-tuned reranker runs
should use different experiment IDs.

`rag_recall.xlsx` is DOCX-driven. It starts with every hypothesis occurrence
that survives the same operative-section extraction and sentence filtering as
full inference, then aligns it to the corresponding workbook:

- Dialogue uses `RAG_DIALOGUE_test.xlsx`.
- Full uses `RAG_FULL_test.xlsx`, then checks
  `RAG_FULL_additional_test.xlsx` only when the hypothesis is absent from the
  primary workbook.
- Missing DOCX hypotheses emit a warning, appear in `missing_hypotheses`, and
  are excluded because their relevance is unknown.
- Zero-article hypotheses and workbook-only hypotheses do not enter Recall.

The first sheet contains one row per embedding/reranker/depth combination and
exactly six article-level micro Recall values for Dialogue and Full. Their
denominators follow production routing: rules-only Recall uses only hypotheses
where deterministic lookup returned at least one premise, while FAISS-only
Recall uses only hypotheses that fell back to semantic retrieval. Total Recall
uses every annotated hypothesis and selects rules when available, otherwise
FAISS. A detected but unresolved citation belongs to the FAISS subset because
that is the path used by inference. The second sheet contains only missing DOCX
hypotheses. Pretrained mode writes eight rows (four variants at two depths);
fine-tuned mode writes twelve.

Rerunning an experiment ID with a complete `rag_manifest.json`,
`run_config.json`, embedding model/index, and requested fine-tuned reranker
reuses those artifacts. The stored RAG commit is restored and only Recall is
recalculated, so updated DOCX annotations or retrieval depths do not trigger
encoder/reranker fine-tuning or FAISS rebuilding. Use a new experiment ID when
training settings or model identities should change.

The Python interface accepts all important memory and reproducibility knobs:

```python
from jura_hypersumm.rag import run_rag_experiment

scores = run_rag_experiment(
    experiment_id="sbert_legal_v1",
    train_path="train.xlsx",
    val_path="val.xlsx",
    rag_dir="dms-rag",
    retrieval_depths=((20, 10), (40, 20)),
    reranker_mode="finetuned",
    hyperparameters={
        "batch_size": 4,
        "eval_batch_size": 8,
        "gradient_accumulation_steps": 8,
        "max_seq_length": 512,
        "precision": "auto",
        "embedding_device": "cuda",
        "index_device": "cpu",
        "reranker_batch_size": 4,
        "reranker_eval_batch_size": 8,
        "reranker_gradient_accumulation_steps": 8,
        "reranker_max_length": 1024,
        "reranker_device": "cuda",
    },
)
```

### Stage-two top-k sweep

`run_rag_depth_sweep_local.py` exposes `run_rag_depth_sweep(...)`, a focused
evaluation-only workflow for the selected baseline-embedding/fine-tuned-
reranker variant. It uses the saved reranker from `sbert_legal_60` for `80:60`,
the reranker from `sbert_legal_40` for `60:40`, and `sbert_legal_v1` for
`100:80`, `40:20`, `20:10`, `100:60`, `100:40`, `100:20`, and `100:10`.
All three manifests must agree on corpus, embedding, and reranker base
revisions. No training dataset is loaded, no model is fine-tuned, and no FAISS
index is rebuilt.

The workflow requires the existing three artifact folders, `dms-rag/codex.csv`
and `faiss_index/`, the RAG annotation workbooks, the paired test DOCX folders,
and the optional GPU dependencies. Run the predefined matrix locally with:

```powershell
uv sync --extra colab --dev
uv run python run_rag_depth_sweep_local.py
```

The equivalent Python/Colab interface is:

```python
from jura_hypersumm.rag import run_rag_depth_sweep

report = run_rag_depth_sweep(
    artifact_root="/content/JURA_hypersumm/local_artifacts/rag",
    rag_dir="/content/dms-rag",
    rag_test_dir="/content/JURA_hypersumm/rag_tests",
    test_docx_dir="/content/JURA_hypersumm/test_docx",
    results_dir="/content/results/top_k_stage2",
    embedding_device="cuda",
    reranker_device="cuda",
    reranker_precision="auto",
    reranker_batch_size=16,
)
```

The workflow loads one local reranker at a time and releases GPU memory between
artifact groups. Combined CSV/XLSX results and `evaluation_config.json` are
written below `local_results/rag/sbert_legal_v1/top_k_stage2/`; per-artifact
subreports remain available below `by_artifact/`. Rows preserve the requested
depth order and record the artifact run, dense Full total-Recall rank, and tied
best status. Existing first-stage results are not modified. Reduce reranker
batch size or select `float16`/`bfloat16` explicitly when GPU memory is tight.

For Colab, clone this repository and `dms-rag`, install `.[colab]`, then copy
or mount the private XLSX/DOCX inputs explicitly before calling the same
interface. The module does not mount Drive, upload files, install packages, or
read Colab secrets at import time.

### Stage-three embedding-model sweep

`run_rag_embedding_sweep_local.py` exposes
`run_rag_embedding_sweep(...)`. It keeps the winning stage-two configuration
fixed—the original SBERT baseline, the fine-tuned reranker saved in
`local_artifacts/rag/sbert_legal_v1`, and candidate/final depth `100:60`—and
changes only the pretrained model used to build the FAISS candidate pool.
Deterministic citation matches still bypass semantic retrieval and reranking.

The default candidates are:

- [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3), using plain query and
  document text;
- [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B),
  using its instruction-aware query format;
- [`intfloat/multilingual-e5-large-instruct`](https://huggingface.co/intfloat/multilingual-e5-large-instruct),
  using its required retrieval instruction on queries only.

All three are sentence-embedding models with Russian/multilingual retrieval
support. Corpus chunks remain unprefixed, while Qwen and E5 queries receive the
same English task description for retrieving Russian legal provisions. Jina
Embeddings v3 is not a default because its checkpoint has a non-commercial
license, and the much larger GigaEmbeddings checkpoint is outside this local
comparison's intended memory profile.

The inputs are the existing winner manifest and reranker, `dms-rag/codex.csv`,
the RAG annotation workbooks, and `test_docx/{Dialogue,Full}`. Install the GPU
dependencies and run all three models locally with:

```powershell
uv sync --extra colab --dev
uv run python run_rag_embedding_sweep_local.py `
  --embedding-device cuda `
  --embedding-precision auto `
  --embedding-batch-size 16 `
  --reranker-device cuda `
  --reranker-batch-size 16
```

Use repeatable `--model` arguments to run a subset, for example
`--model bge_m3 --model qwen3_embedding_0_6b`. A custom plain-text embedding
model can be supplied as `--model my_alias=owner/model`; models requiring a
special prompt should instead be passed as `EmbeddingModelSpec` through the
Python interface.

Each resolved model revision, saved checkpoint, run configuration, and matching
FAISS index is written under `local_artifacts/rag/embedding_stage3/<alias>/`.
Combined CSV/XLSX Recall, improvement over SBERT, Full Recall ranks, an
evaluation manifest, and per-model reports are written under
`local_results/rag/embedding_stage3/`. Matching completed artifacts are reused;
incompatible or partial artifacts fail explicitly. Plan for several gigabytes
of model artifacts plus the Hugging Face download cache.

The Python interface exposes the important resource knobs:

```python
from jura_hypersumm.rag import run_rag_embedding_sweep

report = run_rag_embedding_sweep(
    winner_manifest="local_artifacts/rag/sbert_legal_v1/rag_manifest.json",
    rag_dir="dms-rag",
    rag_test_dir="rag_tests",
    test_docx_dir="test_docx",
    embedding_device="cuda",
    embedding_precision="auto",
    embedding_batch_size=8,
    reranker_device="cuda",
    reranker_precision="auto",
    reranker_batch_size=8,
)
print(report)
```

For Colab, mount or upload every private input explicitly before calling the
same interface; the module does not access Drive or install packages itself:

```python
from google.colab import drive

drive.mount("/content/drive")
%cd /content/JURA_hypersumm
%pip install -e ".[colab]"

from jura_hypersumm.rag import run_rag_embedding_sweep

report = run_rag_embedding_sweep(
    winner_manifest="/content/drive/MyDrive/jura/rag/sbert_legal_v1/rag_manifest.json",
    artifact_root="/content/drive/MyDrive/jura/rag/embedding_stage3",
    rag_dir="/content/dms-rag",
    rag_test_dir="/content/drive/MyDrive/jura/rag_tests",
    test_docx_dir="/content/drive/MyDrive/jura/test_docx",
    results_dir="/content/drive/MyDrive/jura/results/embedding_stage3",
    embedding_batch_size=8,
    reranker_batch_size=8,
)
```

## Rule-based citation audit

`run_citation_audit_local.py` exposes `run_citation_audit(...)`, a CPU-only
workflow for diagnosing deterministic citation Recall. It applies the same
`ПОСТАНОВИЛ` extraction, sentence splitting, and filtration as full inference,
then compares detected citations and exact `codex.csv` matches with the expert
references in the three `rag_tests` workbooks. It reports article-level
`(code, article)` matches separately from complete references that also include
part and point.

The base project dependencies are sufficient; no embedding model, FAISS index,
reranker, GPU, or quantization is used. Required inputs are `dms-rag/codex.csv`,
`test_docx/{Dialogue,Full}`, and the three RAG annotation workbooks. Run it
locally with:

```powershell
uv run python run_citation_audit_local.py
```

To diagnose only the hypotheses where deterministic lookup returned at least
one premise and therefore prevented FAISS fallback, use:

```powershell
uv run python run_citation_audit_local.py --routing-scope rules
```

Within this rules-only report, `missed_expert` rows identify partial rule
failures: the rules resolved something for the hypothesis, but omitted another
expert-relevant article. `--routing-scope faiss` audits the complementary
fallback subset, while `all` remains the default.

The Python interface is equally suitable for Colab after explicitly cloning or
mounting those inputs:

```python
from jura_hypersumm.rag import run_citation_audit

report = run_citation_audit(
    codex_path="/content/dms-rag/codex.csv",
    rag_test_dir="/content/JURA_hypersumm/rag_tests",
    test_docx_dir="/content/JURA_hypersumm/test_docx",
    results_dir="/content/citation_audit",
    routing_scope="rules",
)
print(report)
```

By default, the function returns a timestamped XLSX below
`local_results/rag/citation_audit/`. Its sheets contain aggregate Recall,
one row per pipeline hypothesis, one row per compared article, and hypotheses
missing from expert workbooks. Missing annotations are retained for inspection
but excluded from Recall. Existing output files are never overwritten.

## Inference-only full-pipeline evaluation

`run_full_pipeline_evaluation_local.py` exposes
`run_full_pipeline_evaluation(...)`. It never trains. It recursively discovers
all complete BERT and LoRA artifacts below a campaign/folder, infers the unique
ready base LLMs referenced by those adapters, and runs validation plus the
Dialogue and Full DOCX benchmarks for every model/task.

The default scans the completed baseline campaign and uses the baseline RAG:

```powershell
uv run python run_full_pipeline_evaluation_local.py `
  --models-source local_artifacts/campaigns/full_pipeline_v1 `
  --rag-source dms-rag `
  --prompt-set base
```

The selected production candidate is packaged locally in the self-contained
`rag-qwen/` folder: Qwen3-Embedding-0.6B, its matching FAISS index and corpus,
and the fine-tuned GTE reranker. It achieved Full total Recall
`0.8990825688` at the winning `100:60` depth. Run the full pipeline with the
directory itself as the RAG source:

```powershell
uv run python run_full_pipeline_evaluation_local.py `
  --models-source local_artifacts/campaigns/full_pipeline_v1 `
  --rag-source rag-qwen `
  --reranker-mode bundle `
  --candidate-top-k 100 `
  --final-top-k 60 `
  --embedding-device cuda `
  --reranker-device cuda
```

The folder is intentionally ignored by Git because it contains model weights,
the corpus, and generated FAISS artifacts. Its manifest uses relative paths and
hashes, so the whole folder can be moved or mounted elsewhere unchanged.

Use a tuned RAG by passing its manifest:

```powershell
uv run python run_full_pipeline_evaluation_local.py `
  --models-source local_artifacts/campaigns/full_pipeline_v1 `
  --rag-source local_artifacts/rag/sbert_legal_v1/rag_manifest.json
```

The full pipeline can use no reranker (the backward-compatible default), the
reranker recorded in a tuned bundle, or a separately selected pretrained
reranker:

```powershell
uv run python run_full_pipeline_evaluation_local.py `
  --models-source local_artifacts/campaigns/full_pipeline_v1 `
  --rag-source local_artifacts/rag/sbert_legal_v1/rag_manifest.json `
  --reranker-mode bundle `
  --candidate-top-k 100 `
  --final-top-k 20
```

Use `--reranker-mode pretrained --reranker-model <model-or-local-path>` to
override the bundle. Reranker mode/model and both retrieval depths are recorded
in every score table and in resumable run state.

`--models-source` may instead point to JSON. BERT binary and ternary artifacts
are independent entries, so any number of variants can be evaluated:

```json
{
  "models": [
    {"name": "bert-v1-binary", "family": "bert", "task": "binary", "path": "models/bert/v1/binary"},
    {"name": "bert-v1-ternary", "family": "bert", "task": "ternary", "path": "models/bert/v1/ternary"},
    {"name": "qwen-base", "family": "base_llm", "model_id": "Qwen/Qwen3-8B"},
    {"name": "qwen-lora-r32", "family": "lora", "task": "ternary", "path": "models/lora/qwen-r32"}
  ]
}
```

Relative JSON paths resolve from the JSON file's directory. Base LLM entries
default to both tasks. LoRA base IDs are read from `adapter_config.json` unless
overridden with `base_model_path` or `base_model_id`. The resolved job list is
always saved with the results.

Prompt set `base` loads `prompt.py` (`PROMPT_TEXT`) and `prompt_binary.py`
(`PROMPT_TEXT_BIN`). Prompt set `<suffix>` loads `<suffix>_prompt.py` and
`<suffix>_prompt_binary.py` with the same constants. Prompt files are parsed as
literal data rather than executed. BERT ignores prompts; LoRA result rows flag
when the inference prompt hash differs from its training prompt.

Results and resumable job state default to
`local_results/full_pipeline_evaluation/`. Important GPU knobs available from
the Python interface include quantization, precision, device map, validation
and document batch sizes, input length, embedding device, and retry count.
Gated models read `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` without printing it.
The Hugging Face token embedded in legacy notebooks must be revoked before the
repository is published.

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
%pip install -q uv
!uv export --extra colab --no-dev --no-emit-project --frozen \
  --output-file /tmp/jura-requirements.txt
%pip install -q --upgrade -r /tmp/jura-requirements.txt
%pip install -q --no-deps -e .
```

Restart the Colab runtime after installation, then return to
`/content/JURA_hypersumm` before importing the workflows. Exporting from
`uv.lock` installs the exact dependency versions used by the experiment rather
than whichever compatible versions happen to be newest.

The package reads `train_binary.csv`, `train_ternary.csv`, `val_binary.csv`,
and `val_ternary.csv` directly from this clone. At inference time it clones
`https://github.com/fortvivlan/dms-rag` to `/content/dms-rag` if needed and
fetches its current `main` commit on every run before using that commit's
`codex.csv` and `faiss_index/` artifacts. A dirty local RAG checkout is rejected
so modified or stale artifacts cannot be used silently.

Create a private Colab secret named `HF_TOKEN` before using a gated Hugging
Face model such as Llama, and accept that model's license on Hugging Face. The
token is read from Colab secrets (or the `HF_TOKEN` environment variable) and
is never printed. A token previously embedded in the legacy notebooks must be
considered compromised and revoked.

In Colab, the existing Drive folder is expected at
`/content/drive/MyDrive/jura`. Native runs instead accept an explicit local
artifact root. Only final trained BERT models and final LoRA adapters are
written to the configured artifact root; checkpoints, caches, and score files
remain in the configured runtime/result directories.

### BERT baseline

The public interfaces train one task, select the best validation macro-F1
state, save the final model, run final validation, ask for one or more `.docx`
court decisions, and execute exact-citation/FAISS inference:

```python
from jura_hypersumm.bert import run_bert_binary, run_bert_ternary

binary_scores = run_bert_binary()
ternary_scores = run_bert_ternary()
```

To skip training and use the compatible final model already stored in Drive:

```python
binary_scores = run_bert_binary(use_existing_model=True)
ternary_scores = run_bert_ternary(use_existing_model=True)
```

Reuse is opt-in and strict. The saved directory must contain complete model,
tokenizer, and `run_config.json` artifacts matching the requested task, model
ID, and current train/validation file hashes. If it is absent, incomplete, or
incompatible, the function raises an error and does not start training.

Models are saved below the configured artifact root at `models/bert/binary`
and `models/bert/ternary` (under `MyDrive/jura` by default in Colab). The default is
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

This ready-LLM workflow has no reuse switch because it never trains or saves a
fine-tuned model; every call already performs validation and full inference
only.

### LoRA/QLoRA fine-tuning

`run(model_name, task, hyperparameters=None)` trains one task adapter for one
of the same four models, stores the final adapter, validates it, and performs
full document inference:

```python
from jura_hypersumm.lora import run

scores = run("ministral", "ternary")
```

Every model alias and task can reuse its final adapter from Drive without
fine-tuning it again:

```python
scores = run("ministral", "ternary", use_existing_model=True)
```

For inference only, `run_lora_document_inference(...)` loads an existing adapter
without training, validation, or `autotest` scoring. It recursively processes
the configured `test_docx` directory with the current parsing, filtering, RAG,
and prompting modules, then leaves one persistent prediction XLSX per document:

```python
from jura_hypersumm.lora import run_lora_document_inference

prediction_dir = run_lora_document_inference(
    "ministral",
    "ternary",
    rag_dir="dms-rag",
    drive_root="local_artifacts",
    test_docx_dir="test_docx",
    results_dir="local_results/lora_test_docx_predictions",
    hyperparameters={"document_batch_size": 8, "embedding_device": "cpu"},
)
```

The workflow requires the saved adapter, `dms-rag/codex.csv`,
`dms-rag/faiss_index/`, and the DOCX files. Quantization, precision, device map,
document batch size, input length, generation length, retrieval depth, and
embedding device can be overridden through `hyperparameters`. In Colab, install
the GPU dependencies and mount or upload those artifacts explicitly before
calling the same interface. `lora_local.ipynb` is the native-Windows thin
wrapper for the saved Ministral ternary adapter.

The adapter manifest must match the model ID, task, base-model revision,
prompt, prompt-processing strategy, and current train/validation file hashes.
A missing or incompatible adapter raises an error rather than silently
launching training. Validation, RAG retrieval, document upload, full inference,
XLSX generation, ZIP creation, and downloads still run normally. Ministral
training duplicates the canonical prompt in its user turn to accommodate that
model's chat template while keeping the inference prompt unchanged.

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

### Staged local LoRA hyperparameter search

The local coordinate search evaluates all four LLMs in binary and ternary
modes. It starts from the earlier Qwen recipe: 4-bit NF4 with FP16 compute,
rank 16, alpha 32, `q_proj`/`v_proj`, dropout 0.1, batch size 2, gradient
accumulation 4, five epochs, learning rate `2e-5`, a linear schedule, and
fused AdamW. Run the five stage modules in order, followed by the comparison:

| Module | Public interface | Purpose |
| --- | --- | --- |
| `run_lora_target_modules_local.py` | `run_lora_target_module_experiments(...)` | Compare q/v, q/k/v, and all-linear targets. |
| `run_lora_rank_local.py` | `run_lora_rank_experiments(...)` | Compare rank 8, 16, and 32, initially with alpha = 2r. |
| `run_lora_lr_experiments_local.py` | `run_lora_lr_experiments(...)` | Compare inherited `2e-5` with `1e-4`, `2e-4`, and `1e-5`. |
| `run_lora_alpha_local.py` | `run_lora_alpha_experiments(...)` | Compare alpha = r and alpha = 2r. |
| `run_lora_dropout_local.py` | `run_lora_dropout_experiments(...)` | Compare dropout 0, 0.05, and 0.1. |
| `run_lora_vs_llm_comparison_local.py` | `run_lora_vs_llm_comparison(...)` | Re-evaluate matched ready LLMs and calculate final LoRA-minus-LLM deltas. |

Each stage inherits the validation winner separately for every model/task
pair. Selection uses validation macro F1, then contradiction F1, then fewer
invalid predictions, then grid order. Dialogue and Full
`autotest_model`/`autotest_total` scores are computed for every candidate but
never select a winner. The search has 120 logical candidates; canonical recipe
deduplication leaves 88 unique tuning workflows when inherited defaults remain
among the candidates.

Run these only after other local GPU campaigns have exited. The default cap is
six workflow attempts per invocation, conservatively below 24 hours at the
observed two-to-three-hour runtime. Rerun the identical command until a stage
reports completion, then advance:

```bash
uv run python run_lora_target_modules_local.py --dry-run
uv run python run_lora_target_modules_local.py --search-id lora_coordinate_v1
uv run python run_lora_rank_local.py --search-id lora_coordinate_v1
uv run python run_lora_lr_experiments_local.py --search-id lora_coordinate_v1
uv run python run_lora_alpha_local.py --search-id lora_coordinate_v1
uv run python run_lora_dropout_local.py --search-id lora_coordinate_v1
uv run python run_lora_vs_llm_comparison_local.py --search-id lora_coordinate_v1
```

The engine atomically records progress, retries one failed workflow by default,
resumes the latest Trainer checkpoint, and reuses a completed adapter when
training succeeded but scoring was interrupted. Model revisions, the RAG
commit, dataset and prompt hashes, source fingerprint, seed, full recipe, and
inference settings are locked to the search ID. Use a new ID after changing a
locked input or setting.

Required inputs are `train_binary.csv`, `train_ternary.csv`,
`val_binary.csv`, `val_ternary.csv`, `dms-rag/codex.csv`, the existing
`dms-rag/faiss_index/`, and paired `autotest/{Dialogue,Full}` and
`test_docx/{Dialogue,Full}` folders (13 Dialogue and 30 Full decisions). Install
the optional `.[colab]` dependencies; local execution also needs Git, an NVIDIA
GPU with at least 20 GiB VRAM, and `HF_TOKEN` for gated Llama access.

State and consolidated scores are written below
`local_results/lora_searches/<search-id>/`. Each stage writes `scores.csv` and
`results.xlsx` with candidate, validation, benchmark, winner, experiment, and
failure sheets. Isolated checkpoints and workflow workbooks live under its
`experiments/` directory. Adapters are stored below
`local_artifacts/lora_searches/<search-id>/experiments/`. The final comparison
is `comparison/lora_vs_llm.xlsx`.

All stage interfaces accept `models`, `tasks`, `max_attempts_per_run`,
`max_retries`, and non-swept `hyperparameters`. Important memory knobs include
`batch_size`, `gradient_accumulation_steps`, `eval_batch_size`,
`document_batch_size`, `max_seq_length`, `gradient_checkpointing`,
`device_map`, `precision`, and `quantization`:

```python
from run_lora_target_modules_local import run_lora_target_module_experiments

ranking = run_lora_target_module_experiments(
    repo_root=".",
    search_id="lora_coordinate_v1",
    hyperparameters={"eval_batch_size": 4, "document_batch_size": 2},
)
```

In Colab, mount/copy datasets and private benchmark inputs explicitly before
calling the same interfaces; the modules do not mount Drive or install
packages:

```python
!git clone https://github.com/fortvivlan/JURA_hypersumm /content/JURA_hypersumm
%cd /content/JURA_hypersumm
!pip install -e ".[colab]"
!git clone https://github.com/fortvivlan/dms-rag /content/JURA_hypersumm/dms-rag

from google.colab import drive, userdata
from pathlib import Path
import os
import shutil

drive.mount("/content/drive")
repo = Path("/content/JURA_hypersumm")
inputs = Path("/content/drive/MyDrive/jura_inputs")
for name in ("train_binary.csv", "train_ternary.csv", "val_binary.csv", "val_ternary.csv"):
    shutil.copy2(inputs / name, repo / name)
for folder in ("autotest", "test_docx"):
    shutil.copytree(inputs / folder, repo / folder, dirs_exist_ok=True)
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

from run_lora_target_modules_local import run_lora_target_module_experiments
from run_lora_rank_local import run_lora_rank_experiments
from run_lora_lr_experiments_local import run_lora_lr_experiments
from run_lora_alpha_local import run_lora_alpha_experiments
from run_lora_dropout_local import run_lora_dropout_experiments
from run_lora_vs_llm_comparison_local import run_lora_vs_llm_comparison

run_lora_target_module_experiments(repo_root=repo)
# After each preceding stage reports completion:
run_lora_rank_experiments(repo_root=repo)
run_lora_lr_experiments(repo_root=repo)
run_lora_alpha_experiments(repo_root=repo)
run_lora_dropout_experiments(repo_root=repo)
comparison = run_lora_vs_llm_comparison(repo_root=repo)
```

### Native Windows runs

BERT, ready-LLM, and LoRA public workflows can also run from a native Windows
Python environment. The environment must contain the optional GPU dependencies,
`git` must be on `PATH`, and an NVIDIA CUDA GPU is required. Pass repository-local
paths explicitly; generated files are reported rather than downloaded:

```python
from jura_hypersumm.bert import run_bert_binary

scores = run_bert_binary(
    rag_dir="dms-rag",
    drive_root="local_artifacts",
    results_dir="local_results",
    multiple_test=True,
)
```

The same local path pattern applies to `run_llm_evaluation(...)` and
`jura_hypersumm.lora.run(...)`. Local DOCX inputs are never deleted; only files
uploaded through Colab use temporary storage and automatic cleanup.

### Benchmark scoring and results

Every full document workflow runs the repository benchmark by default. Each
reviewed XLSX in `autotest/` is matched to one DOCX in `test_docx/` by surname,
or by its first organization-name word; leading `Тест` and `ООО` words and filename
decorations do not affect matching. Explicit `document_paths` override the
default set, while `score_autotest=False` disables scoring. The five excluded
decisions are retained separately in `problematic_docx/`.

Set `multiple_test=True` to evaluate paired immediate child folders separately,
for example `test_docx/Dialogue` against `autotest/Dialogue` and then
`test_docx/Full` against `autotest/Full`:

```python
scores = run(
    "ministral",
    "ternary",
    multiple_test=True,
)
```

The flag is available on the BERT, LoRA, ready-LLM, and standalone scoring
interfaces and defaults to `False`, preserving root-level discovery. All child
folder names must be paired between the two roots. Explicit `document_paths`
still take precedence over child-folder discovery.
The scorer accepts both current `model_predictions` review sheets and the
legacy single-sheet Dialogue files, where `prediction` is the reviewed label
and the article reference is recovered from the premise text.

Fresh pairs are aligned by document, hypothesis, premise, and article reference;
the historical `model_prediction` column in the reviewed workbook is ignored.
A newly retrieved pair absent from XLSX receives gold `not mentioned` for a
ternary task or `no` for a binary task. A reviewed relevant pair absent from
current retrieval is a RAG miss. Ternary totals count missed contradictions and
entailments as false negatives. Binary totals count only missed contradictions;
missed entailments are omitted.

Every workflow returns and displays one score table with `validation`,
`autotest_model`, and `autotest_total` scopes. Multi-dataset benchmark rows also
carry a `test_dataset` column such as `Dialogue` or `Full`. The XLSX additionally contains
per-class metrics, confusion matrices, current and manually marked RAG counts,
pair alignment, inferred-gold pairs, excluded rows, file matching, raw
predictions, errors, and reproducibility metadata. Results are written under
`/content/jura_results`, not Drive.
Reused-model workbooks record `used_existing_model=True` and
`training_skipped=True`; their training-history sheet is copied from the saved
artifact manifest when available.

Long-running Colab workflows print immediately flushed `[JURA]` stage messages
before and after model loading or reuse, training, each validation task, RAG
setup, document testing, and result generation. Training and per-document
inference retain their batch/sentence progress bars.

An existing results workbook can be rescored without loading a model:

```python
from jura_hypersumm.autotest_scoring import run_autotest_scoring

scores = run_autotest_scoring(
    "local_results/lora_results.xlsx",
    autotest_dir="autotest",
    docx_dir="test_docx",
    multiple_test=True,
    output_dir="local_results",
)
```

```bash
python -m jura_hypersumm.autotest_scoring RESULTS.xlsx \
  --autotest-dir autotest --docx-dir test_docx --multiple-test \
  --output-dir local_results
```

Standalone scoring requires pandas, openpyxl, and scikit-learn and does not
need a GPU. Full workflows retain their existing batch-size, precision,
quantization, device, and checkpointing controls.

When at least one document is successfully analysed, the workflow also
downloads a ZIP prepared for human review. For every document/task it contains
`<document>_<task>_model_predictions.xlsx` with every RAG premise/model pair and
the columns `hypothesis`, `premise`, `article_number`, `model_prediction`,
`expert_label`, and `expert_comment`. Article values include the code, dotted
article number, part, and point where available, for example
`КоАП РФ Статья 32.9 Часть 1 Пункт 2`. A specialist can fill the last two
columns so the predictions can later be scored against human labels. No separate
top-ranked RAG workbook is generated; all retrieved candidates remain available
in the model workbook and in the detailed main results workbook.
In multi-dataset runs, these workbooks are stored below dataset-named folders in
the ZIP so duplicate Dialogue and Full filenames remain distinct.

LoRA and BERT runs produce one model-prediction workbook per document. A ready
LLM run produces binary and ternary model-prediction workbooks for each
document.

Document inference extracts the final `ПОСТАНОВИЛ` section, splits it with
`razdel`, removes signatures and payment-detail sentences containing `судья`,
`реквизит`, `ре...изит`, `квитанци`, or standalone case-sensitive
`БИК`, `ИНН`, `УИН`, `КБК`, `ОКТМО`, `л/с`, or `р/с`, as well as
sentences containing only digits.
The `Отд.` abbreviation is protected from sentence splitting so a bank-details
sentence is filtered as a whole. The pipeline then performs deterministic
citation lookup before a maximum of 20 FAISS matches, and preserves the premise
responsible for every contradiction. Deterministic lookup accepts abbreviated
and full code names, both `п. … ч. … ст. …` and `ст. … ч. … п. …` orders,
multiple citations,
and article lists/ranges. A cited part returns all of its points; an article-only
reference that maps to several corpus rows falls back to FAISS. Detected and
unresolved citations are retained in the audit sheets. A document without
`ПОСТАНОВИЛ` is reported and skipped.
Uploaded `.docx` files are isolated in a temporary directory and deleted after
processing even if an error occurs.

## Reproducibility

Strict deterministic execution is enabled by default for BERT, ready-LLM, and
LoRA workflows. Each run:

- seeds Python, NumPy, PyTorch, CUDA, trainer sampling, and BERT data-loader
  workers with the recorded seed (42 by default);
- enables deterministic PyTorch algorithms and deterministic cuBLAS/cuDNN,
  disables cuDNN benchmarking and TF32, and uses greedy LLM generation;
- pins BERT and RAG embeddings to an immutable Hugging Face commit and pins
  Ministral, Qwen, and T-lite to immutable model commits;
- fetches the latest `dms-rag/main` at run start, checks out the resolved commit
  for the complete run, and records both requested `main` and the resolved hash;
- resolves gated Llama to an immutable commit using `HF_TOKEN` before loading
  it;
- records SHA-256 hashes for train/validation CSVs, uploaded documents,
  prompts, and executable source files;
- records the repository commit/dirty state, hyperparameters, resolved model
  and RAG commits, package and Python versions, CUDA, cuDNN, and GPU model in
  the result workbook. Trained-model `run_config.json` files contain the
  corresponding manifest in Drive.

For an exact rerun, use the same repository commit and `uv.lock`, unchanged
input files, the recorded hyperparameters and resolved revisions, and the same
GPU model/driver/CUDA stack. For example, a later Llama rerun should pass the
`resolved_revision` value from the first workbook:

```python
scores = run(
    "llama",
    "ternary",
    revision="<40-character resolved_revision from run_metadata>",
    rag_revision="<40-character rag_commit from run_metadata>",
)
```

Omit `rag_revision` (or leave it as `"main"`) for a normal run that must use the
latest remote RAG artifacts. Passing the recorded hash intentionally disables
that update for an exact rerun.

Bitwise equality is not guaranteed across different GPU architectures or CUDA
drivers because their floating-point kernels differ. Under strict mode, an
operation that PyTorch knows cannot be deterministic raises an error instead
of silently producing a non-reproducible run. Setting
`hyperparameters={"deterministic": False}` (or the corresponding ready-LLM
inference parameter) explicitly opts out of this guarantee.
