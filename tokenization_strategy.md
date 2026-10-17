# Tokenization Strategy — Indonesian Sentiment Analysis (SmSA)

## 1. Surface Stats (Train split, n=10,984)

| Metric      | Min | Max | Mean  | p95 |
| :---------- | --: | --: | ----: | --: |
| Char length |   3 | 567 | 189.1 | 413 |
| Word count  |   1 | 110 |  33.0 |  71 |

**Takeaway**: texts are short-to-medium length, but word counts do not establish
subword token lengths. The optional pretrained-tokenizer analysis in
`analyze_tokenization.py` reproduces the following results on the same training
split, including special tokens and without padding or truncation:

| Tokenizer | Min | Max | Mean | p95 | Texts >128 tokens | Texts >256 tokens | Texts >512 tokens |
| :-------- | --: | --: | ---: | --: | ----------------: | ----------------: | ----------------: |
| IndoBERT | 3 | 132 | 37.81 | 79 | 1 | 0 | 0 |
| XLM-R | 3 | 164 | 45.59 | 96 | 8 | 0 | 0 |

Use **max_length = 256** for both transformers with truncation enabled and
dynamic padding per batch. This covers every current training example without
truncation. These measurements apply only to the training split; coverage of
validation, test, and future inputs has not been established by this check.

---

## 2. Slang & Noise Audit (Train split, n=10,984)

| Pattern         | Count | % of corpus |
| :-------------- | ----: | ----------: |
| informal_abbrev | 2,913 |       26.5% |
| english_word    | 1,846 |       16.8% |
| elongation      |   248 |        2.3% |
| wkwk_laugh      |   163 |        1.5% |
| hashtag_mention |     0 |        0.0% |
| url             |     0 |        0.0% |
| emoji_non_ascii |     0 |        0.0% |

**Takeaway**: `analyze_tokenization.py` finds informal-pattern matches in
26.5% of training texts and English-word-pattern matches in 16.8%. The former
includes variants such as `gak`, `nggak`, `bgt`, `yg`, and `kalo`; the latter
includes review and restaurant terms such as `food`, `service`, `delivery`,
`cozy`, and `recommended`. These are regex-based indicators, not exhaustive
linguistic annotations: the informal list also includes common words and
particles, and an English-word match alone does not establish code-switching.

Each count measures texts with at least one match, rather than occurrences.
Categories can overlap, so percentages should not be summed. Elongation and
laugh-pattern matches each occur in fewer than 3% of texts. The URL,
hashtag/mention, and non-ASCII detectors found zero matches; `emoji_non_ascii`
checks all non-ASCII characters, rather than identifying emoji specifically.

---

## 3. Tokenization Decisions

### BiLSTM (trained from scratch)

- Build vocab from train split with **min_freq = 2** to filter one-off noise.
- Apply **elongation normalization** before building vocab:
  `re.sub(r'(.)\1{3,}', r'\1\1', text)` — reduces runs of **4 or more**
  identical characters to 2, leaving runs of 3 unchanged.
- The audit detects runs of **3 or more** (`r'(.)\1{2,}'`, 248 texts), whereas
  this proposed normalization affects only 30 training texts. These thresholds
  serve different purposes and should not be reported as the same count.
- Keep informal abbreviations as-is to preserve potentially useful sentiment
  cues.
- Use `<UNK>` for rare/unseen tokens at inference.
- Apply the same preprocessing at training and inference; build the vocabulary
  from training data only.

**Rationale**: BiLSTM learns embeddings from scratch, so reducing rare spelling
variants may help vocabulary coverage. Conservative elongation normalization
and `min_freq = 2` are starting choices, not demonstrated improvements.
Repetition can also convey emphasis, and the corpus-level informal-pattern
rate does not establish that each abbreviation is frequent enough to learn a
strong embedding.

### IndoBERT (`indobenchmark/indobert-base-p1`)

- Use the pretrained **IndoBERT WordPiece tokenizer as-is**, without additional
  slang expansion, stemming, or elongation normalization. Its built-in
  processing still applies, including lowercasing for this uncased checkpoint.
- Use **max_length = 256** with truncation and dynamic padding. Measured
  training lengths are p95 = 79 and max = 132 tokens, including special tokens.

**Rationale**: retaining the pretrained tokenizer provides a consistent starting
point for fine-tuning and avoids introducing untested external normalization.
Preserving informal forms may retain useful cues; whether this improves
sentiment classification must be established experimentally.

### XLM-RoBERTa (`xlm-roberta-base`)

- Use the pretrained **XLM-R SentencePiece tokenizer as-is**, without additional
  text normalization. The tokenizer's built-in normalization still applies.
- Use **max_length = 256** with truncation and dynamic padding. Measured
  training lengths are p95 = 96 and max = 164 tokens, including special tokens.

**Rationale**: the existing pretrained tokenizer is a consistent starting point
for the multilingual candidate. Its measured token lengths fit the proposed
training window, but the effects of slang fragmentation and English-word
usage on sentiment performance remain hypotheses to test.

---

## 4. Summary

| Model | Tokenizer | Additional preprocessing | Starting decision |
| :---- | :-------- | :----------------------- | :---------------- |
| BiLSTM | Whitespace + Counter | Reduce character runs of 4+ to 2 | Train-only vocabulary; min_freq=2; `<UNK>` for unseen tokens |
| IndoBERT | Pretrained WordPiece | None; retain built-in lowercasing | max_length=256; dynamic padding; train max=132 tokens |
| XLM-R | Pretrained SentencePiece | None; retain built-in normalization | max_length=256; dynamic padding; train max=164 tokens |

These are tokenization starting choices, not measured classification results.
