# Tokenization Strategy — Indonesian Sentiment Analysis (SmSA)

## 1. Surface Stats (Train split, n=10,984)

| Metric      | Min | Max | Mean  | p95 |
| :---------- | --: | --: | ----: | --: |
| Char length |   3 | 567 | 189.1 | 413 |
| Word count  |   1 | 110 |  33.0 |  71 |

**Takeaway**: texts are short-to-medium length. At p95 = 71 words, transformer
512-token limits are not a concern — virtually all texts fit within the window.

---

## 2. Slang & Noise Audit (Train split, n=10,984)

| Pattern         | Count | % of corpus |
| :-------------- | ----: | ----------: |
| informal_abbrev | 1,161 |       10.6% |
| english_word    |   287 |        2.6% |
| elongation      |   248 |        2.3% |
| wkwk_laugh      |   163 |        1.5% |
| hashtag_mention |     0 |        0.0% |
| url             |     0 |        0.0% |
| emoji_non_ascii |     0 |        0.0% |

**Takeaway**: dataset is relatively clean. No URLs, hashtags, or emoji. Main
noise is informal abbreviations (~10.6%) — words like `gak`, `bgt`, `yg`,
`kalo`. Elongation and laugh patterns are minor (<3%). Mixed English appears
in ~2.6% of texts (code-switching).

---

## 3. Tokenization Decisions

### BiLSTM (trained from scratch)

- Build vocab from train split with **min_freq = 2** to filter one-off noise.
- Apply **elongation normalization** before building vocab:
  `re.sub(r'(.)\1{3,}', r'\1\1', text)` — caps repeats at 2, reduces hapax.
- Keep informal abbreviations as-is; they carry sentiment signal.
- Use `<UNK>` for rare/unseen tokens at inference.

**Rationale**: BiLSTM learns embeddings from scratch so vocab quality matters.
Elongated forms inflate vocab size without adding meaning — normalize them.
Abbreviations are frequent enough (10.6%) to learn good representations.

### IndoBERT (`indobenchmark/indobert-base-p1`)

- Use pretrained **IndoBERT tokenizer as-is**. No text normalization.
- IndoBERT was pretrained on informal Indonesian text — handles `gak`, `bgt`,
  `yg`, and code-switched English natively via its WordPiece vocabulary.
- All texts fit within 512 tokens (p95 word count = 71).

**Rationale**: normalizing slang before IndoBERT destroys signal the model
already knows how to use. Trust the pretrained tokenizer.

### XLM-RoBERTa (`xlm-roberta-base`)

- Use pretrained **XLM-R tokenizer as-is**. No text normalization.
- XLM-R covers 100 languages including Indonesian. Low noise rate means
  subword fragmentation of slang is acceptable.
- Same 512-token safety margin applies.

**Rationale**: XLM-R's SentencePiece tokenizer will fragment some slang into
subwords, but given the low noise level (<3% elongation, 10.6% abbrev) and the
model's multilingual pretraining, this does not significantly hurt performance.

---

## 4. Summary

| Model    | Tokenizer           | Normalization        | Notes                             |
| :------- | :------------------ | :------------------- | :-------------------------------- |
| BiLSTM   | Whitespace + Counter | Elongation normalize | min_freq=2, build own vocab       |
| IndoBERT | IndoBERT tokenizer  | None                 | Handles Indonesian slang natively |
| XLM-R    | XLM-R tokenizer     | None                 | 512-token limit not an issue      |
