**Finding:** Method B did not improve average neutral-class performance in these
runs. Method A achieved **6.59 percentage points higher test neutral F1**.

## What is being compared?

**Method A** uses XLM-RoBERTa (`xlm-roberta-base`) with standard cross-entropy.
**Method B** uses IndoBERT (`indobenchmark/indobert-base-p1`) with class-weighted
cross-entropy. Neutral comments express no clear positive or negative opinion
and represent only **10.43% of training examples** (1,146 of 10,984). Method B
assigns neutral errors a larger loss weight (3.19, versus 1.07 for negative and
0.57 for positive), aiming to improve recognition of this smaller class.

Neutral **precision** measures how often a neutral prediction is correct;
neutral **recall** measures how many actual neutral comments are recognized.
Neutral **F1** balances these two measures. Macro-F1 gives equal importance to
all three sentiment classes, helping check overall performance.

## Results

Both methods used the same splits, five training epochs, and seeds **42, 123,
999**. Each checkpoint was selected using validation Macro-F1. The full test
set contains **500 examples, including 88 neutral examples**. Scores below are
percentages, reported as **mean ± sample standard deviation across seeds**.
Differences use unrounded scores and are **B − A in percentage points (pp)**;
positive differences favor B.

| Test metric (%) | Method A | Method B | B − A (pp) |
| :--- | ---: | ---: | ---: |
| Accuracy | 92.07 ± 0.81 | 89.60 ± 1.22 | -2.47 |
| Macro-F1 | 89.88 ± 1.54 | 86.47 ± 1.98 | -3.41 |
| Neutral precision | 94.37 ± 4.47 | 90.41 ± 1.48 | -3.95 |
| Neutral recall | 72.35 ± 6.94 | 64.39 ± 6.46 | -7.95 |
| Neutral F1 | **81.69 ± 4.06** | **75.10 ± 4.53** | **-6.59** |

| Seed | A: test neutral F1 (%) | B: test neutral F1 (%) | B − A (pp) |
| ---: | ---: | ---: | ---: |
| 42 | 84.42 | 71.72 | -12.69 |
| 123 | 77.03 | 80.25 | +3.23 |
| 999 | 83.64 | 73.33 | -10.30 |

## Interpretation

Method B improves test neutral F1 only for seed 123. Its lower mean neutral
recall means more neutral comments are classified as positive or negative:
B misses **36, 25, and 33** of the 88 neutral examples, compared with A's
**23, 31, and 19**. B also has lower mean neutral precision, so its recall
decrease is not balanced by more reliable neutral predictions. Overall
accuracy and Macro-F1 also favor A.

Validation supports the same direction: neutral F1 is **86.76 ± 0.37% for A**
versus **85.40 ± 0.74% for B**, across 1,260 examples including 131 neutral.

B has fewer parameters (**124.44M vs 278.05M**) and shorter mean recorded
training-plus-validation time (**1,348.9 vs 6,969.9 seconds**, about 5.17×
faster) on the same recorded GPU; timings exclude downloads, checkpoint writes,
and final inference.

The hypothesized improvement over A was not observed. However, both the
backbone and loss change between methods, so these results **cannot isolate
the effect of class weighting alone**. Three seeds describe observed
variability without establishing statistical significance.