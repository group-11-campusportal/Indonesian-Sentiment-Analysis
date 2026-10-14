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

## 2. Team Members & Roles

| Member | Role | Key Responsibilities |
| :--- | :--- | :--- |
| **Member 1 (Current)** | **Data & Baseline Engineer** | Ingestion pipeline, split verification, leakage audit, data distribution profiling, and scratch PyTorch BiLSTM architecture. |
| **Member 2** | **Deep Learning / Transformer Engineer** | Method A & Method B implementation, IndoBERT/IndoBERTweet fine-tuning, custom design decision (loss/head). |
| **Member 3** | **Experiments, Analysis & Paper Lead** | Multi-seed execution framework, metric aggregation (`mean ± std`), learning curves, confusion matrix, failure analysis, and IEEE LaTeX compilation. |

---

## 3. Dataset Overview & Distribution

We utilize the official **SmSA (Sentiment Analysis for Indonesian)** benchmark from the **IndoNLU** project ([Wilie et al., 2020](https://arxiv.org/abs/2011.00677)).

### Official Split Distribution (Post-Leakage Audit)

| Sentiment Class | Train Count | Train (%) | Val Count | Val (%) | Test Count | Test (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Positive** | 6,407 | 58.33% | 735 | 58.33% | 208 | 41.60% |
| **Negative** | 3,431 | 31.24% | 394 | 31.27% | 204 | 40.80% |
| **Neutral** | 1,146 | 10.43% | 131 | 10.40% | 88 | 17.60% |
| **Total** | **10,984** | **100%** | **1,260** | **100%** | **500** | **100%** |

> **Leakage Audit Note**: An automated exact-match check identified 16 overlapping samples between the raw training set and validation/test sets. To ensure strict evaluation integrity, those 16 samples were deduplicated from the training split, leaving 10,984 clean training records with zero test-set contamination.

---

## 4. Environment Setup

### Prerequisites
- Python 3.9+
- GPU runtime (Google Colab T4 or local CUDA-enabled environment)

### Installation
Clone this repository and install the dependencies:
```bash
git clone <your-github-repo-url>
cd FinalProject
pip install -r requirements.txt
```

---

## 5. Reproducing Milestone 1 (Data Preparation)

To download the raw IndoNLU SmSA benchmark, audit for data leakage, compute class distributions, and generate clean split CSVs:
```bash
python prepare_data.py
```

Generated artifacts in `data/`:
- `data/train_split.csv` (10,984 rows)
- `data/val_split.csv` (1,260 rows)
- `data/test_split.csv` (500 rows)
- `data/class_distribution.csv`

---

## 6. Planned Methodology & Experimental Setup

### Models Under Comparison
1. **Baseline**: Custom 2-layer Bidirectional LSTM (BiLSTM) with learned word embeddings trained from scratch.
2. **Method A**: Pretrained Multilingual Transformer (`xlm-roberta-base` or `bert-base-multilingual-cased`).
3. **Method B (Custom Design Decision)**: Monolingual `IndoBERT` (`indobenchmark/indobert-base-p1`) fine-tuned with **Class-Weighted Cross-Entropy Loss** to combat severe neutral class imbalance.

### Hypothesis
> We hypothesize that **Method B** (IndoBERT with class-weighted loss) will achieve higher Macro-F1 than **Method A** and the **Baseline**, because generic language models struggle with Indonesian colloquial subwords, and standard cross-entropy fails on the underrepresented neutral class.

### Evaluation Rigor
- All models are trained across **at least 3 random seeds** (e.g. 42, 123, 999).
- Metrics reported: **Accuracy**, **Macro-F1**, and **Weighted-F1** (`mean ± std`).