# Indonesian Sentiment Analysis (IndoNLU SmSA)

A comparative deep learning study evaluating custom baseline architectures against pretrained language models for Indonesian colloquial sentiment analysis.

This project is developed as the Final Project for the **Deep Learning Course** following IEEE conference paper requirements.

---

## 1. Problem Statement
Given an informal Indonesian social media or customer review text containing slang, colloquialisms, and non-standard syntax, classify the sentiment polarity into one of three classes:
- **Positive**
- **Neutral**
- **Negative**

---

## 2. Dataset Overview & Distribution

We utilize the official **SmSA (Sentiment Analysis for Indonesian)** benchmark from the **IndoNLU** project ([Wilie et al., 2020](https://aclanthology.org/2020.aacl-main.85/)).

### Official Split Distribution (Post-Leakage Audit)

| Sentiment Class | Train Count | Train (%) | Val Count | Val (%) | Test Count | Test (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Positive** | 6,407 | 58.33% | 735 | 58.33% | 208 | 41.60% |
| **Negative** | 3,431 | 31.24% | 394 | 31.27% | 204 | 40.80% |
| **Neutral** | 1,146 | 10.43% | 131 | 10.40% | 88 | 17.60% |
| **Total** | **10,984** | **100%** | **1,260** | **100%** | **500** | **100%** |

> **Leakage Audit Note**: An automated exact-match check identified 16 training records whose text also appeared in validation or test. Removing those records leaves 10,984 training records. An audit of the saved splits finds zero exact text overlap for train/validation, train/test, and validation/test after stripping surrounding whitespace. Near-duplicate leakage was not checked.

The cleaned split proportions are **86.19% train / 9.89% validation / 3.92% test**.
Split hashes differ between recorded machines because of Unix versus Windows
line endings; normalizing line endings reproduces the recorded hashes.

---

## 3. Environment Setup

### Prerequisites
- Python 3.11 or later is required.
- Python runtime with CUDA or Apple MPS for GPU training; CPU is supported but slower.

### Installation
Clone this repository and install the dependencies:
```bash
git clone https://github.com/group-11-campusportal/Indonesian-Sentiment-Analysis.git
cd Indonesian-Sentiment-Analysis
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

---

## 4. Methods & Experimental Setup

### Models Under Comparison
1. **Baseline**: Custom 2-layer Bidirectional LSTM (BiLSTM) with learned word embeddings trained from scratch.
2. **Method A**: Pretrained Multilingual Transformer (`xlm-roberta-base`).
3. **Method B (Custom Design Decision)**: Monolingual `IndoBERT` (`indobenchmark/indobert-base-p1`) fine-tuned with **Class-Weighted Cross-Entropy Loss** to combat severe neutral class imbalance.

### Hypothesis
> We hypothesized that **Method B** (IndoBERT with class-weighted loss) would achieve higher Macro-F1 than **Method A** and the **Baseline**, expecting Indonesian pretraining to help represent colloquial vocabulary and inverse-frequency weighting to improve recognition of the underrepresented neutral class.

The reported experiments did **not** support this hypothesis. Method A had the
highest mean test Macro-F1. Method B changes both backbone and loss, so the
comparison cannot isolate the effect of weighting alone.

### Evaluation Rigor
- All models were trained for **5 epochs** with seeds **42, 123, 999**.
- Each run selects the checkpoint with highest validation Macro-F1, then
  evaluates that checkpoint on test. Test scores do not select checkpoints.
- Metrics are **Accuracy**, **Macro-F1**, and **Weighted-F1**, reported as
  mean ± sample standard deviation across seeds.
- The common budget is five passes through the same training data, not equal
  compute or optimizer updates. Baseline batch size is 32; transformer
  microbatch size is 4 with 4 accumulation steps (effective batch size 16).
- The baseline used a Tesla T4; both transformers used an RTX 3050 Laptop GPU.
  Recorded runtimes are comparable between A and B, but not across those GPUs.

## 5. Reproduce the Training Pipeline

Run commands from the repository root. Data preparation downloads the public
IndoNLU TSV files, removes the exact training overlaps, and writes the split
CSVs and class distribution under `data/`:

```bash
python -m sentiment.prepare_data
python -m sentiment.check_baseline
python -m sentiment.check_models --method a
python -m sentiment.check_models --method b
```

The transformer checks download pretrained weights if they are not cached.
Training supports `--device auto`, `cuda`, `mps`, or `cpu`; the following commands
choose an available device automatically. CPU training may take much longer.

Use a fresh output directory for each complete experiment. The transformer
trainer rejects an existing run directory, while the baseline trainer can
overwrite one. The following commands execute all nine runs:

```bash
for seed in 42 123 999; do
  python -m sentiment.train_baseline --seed "$seed" --output-dir outputs/reproduction
done

for method in a b; do
  for seed in 42 123 999; do
    python -m sentiment.train_transformers --method "$method" --seed "$seed" --output-dir outputs/reproduction
  done
done
```

Each completed run writes:

```text
outputs/reproduction/<baseline|method_a|method_b>/seed_<42|123|999>/
  summary.json
  history.csv
  val_predictions.csv
  test_predictions.csv
  best_checkpoint/
```

Baseline checkpoints contain `model.pt` and `vocab.json`; transformer checkpoints
contain the model and tokenizer saved by Hugging Face.

## 6. Aggregate, Evaluate, and Plot

```bash
python -m sentiment.aggregate_results --output-dir outputs/reproduction
python -m sentiment.evaluation outputs/reproduction/baseline/seed_*/test_predictions.csv
python -m sentiment.evaluation outputs/reproduction/method_a/seed_*/test_predictions.csv
python -m sentiment.evaluation outputs/reproduction/method_b/seed_*/test_predictions.csv
python comparison/analyze.py --output-dir outputs/reproduction
python -m sentiment.visualize --output-dir outputs/reproduction --skip-confidence
```

Aggregation produces `all_runs.csv` and `results_table.csv`. Check that all three
models have all three seeds; the aggregator warns about missing runs but can
still write a partial table. Evaluation prints per-run metrics, per-class scores,
confusion matrices, and mean ± sample SD across the supplied seeds. The comparison
script verifies the A/B predictions against their saved summaries.

Visualization writes `figures/loss_curves.png` and `figures/confusion_matrix.png`.
The loss panels are stacked for readability in a single paper column. Lines show
seed means, and bands show seed minima and maxima. Method B uses weighted loss,
so absolute losses should not be compared across methods. Model/seed selection
for the confusion matrix uses validation Macro-F1.

To also regenerate `figures/test_misclassified.csv` with softmax confidence,
the selected run's complete checkpoint is required:

```bash
python -m sentiment.visualize --output-dir outputs/reproduction --device auto
```

### Inspect the existing results without retraining

The shared [training-results folder](https://drive.google.com/drive/folders/1YW4VHfRNiC21PWwezfBY2w4SUvIMrBk0)
contains the reported tables, figures, misclassified examples, and an
[`outputs` subfolder](https://drive.google.com/drive/folders/1h0MgHHB0DrGOQi8eqvR3G4wkYwywXFZh)
with per-method, per-seed artifacts. Download that subfolder, unzip it if needed,
and place its contents so `outputs/downloaded/baseline/seed_42/summary.json`
and the corresponding A/B and remaining seed folders exist. Point the commands
above at `--output-dir outputs/downloaded` (and use that prefix for prediction
paths). `--skip-confidence` avoids loading a model checkpoint, but still requires
the saved histories and selected run's test predictions.

`data/` CSV/JSON files and `outputs/` are Git-ignored; a clean clone does not
contain those artifacts. Generate them or obtain the recorded outputs before
running analysis.