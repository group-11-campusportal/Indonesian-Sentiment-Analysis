"""
Tokenization Analysis — Indonesian Sentiment (SmSA)
Produces numbers used directly in tokenization_strategy.md.
"""

import re
import json
import numpy as np
import pandas as pd

# ── Load data ──────────────────────────────────────────────────────────────────
train_df = pd.read_csv("data/train_split.csv")
texts = train_df["text"].astype(str).tolist()

# ── 1. Surface Stats ───────────────────────────────────────────────────────────
char_lens = [len(t) for t in texts]
word_lens  = [len(t.split()) for t in texts]

def stats(arr, label):
    a = np.array(arr)
    print(f"{label}: min={a.min()}  max={a.max()}  mean={a.mean():.1f}  p95={np.percentile(a,95):.0f}")

print("=== Surface Stats ===")
stats(char_lens, "Char length")
stats(word_lens,  "Word count ")

# ── 2. Slang & Noise Audit ────────────────────────────────────────────────────
slang_words = (
    r'\b(gk|ga|gak|nggak|ngga|enggak|kagak|gpp|bgt|banget|aja|udah|udh|dah|'
    r'krn|karna|utk|emg|yg|dgn|tdk|sy|lg|lagi|jg|sdh|tp|dr|pd|bs|msh|skrg|'
    r'blm|klo|kalo|kayak|kek|gmn|gimana|sih|dong|deh|nih|tuh|nah|lho|kan|'
    r'gaes|guys|bro|sis|gan|min|kak|mager|gabut|kepo|baper|santuy|mantul|'
    r'receh|sabi|parah|bener|beneran|cuma|wah)\b'
)

english_words = (
    r'\b(the|is|are|good|bad|best|great|very|for|with|not|nice|love|like|'
    r'price|quality|service|product|recommend|recommended|review|food|place|'
    r'taste|tasty|delicious|yummy|fresh|menu|staff|friendly|clean|cozy|'
    r'crowded|queue|worth|affordable|expensive|discount|promo|delivery|'
    r'order|takeaway|take-away|dine-in|booking|location|ambience|ambiance|'
    r'vibe|spot|hangout|must-try|musttry|disappointed|satisfied|experience|'
    r'overall|fast|slow|enjoy|thanks|please|sorry|update)\b'
)

patterns = {
    "informal_abbrev" : lambda t: bool(re.search(slang_words, t, re.IGNORECASE)),
    "elongation"      : lambda t: bool(re.search(r'(.)\1{2,}', t)),
    "wkwk_laugh"      : lambda t: bool(re.search(r'\b(wk){2,}|(ha){3,}|(hi){3,}|(hehe)+', t, re.IGNORECASE)),
    "hashtag_mention" : lambda t: bool(re.search(r'[@#]\w+', t)),
    "url"             : lambda t: bool(re.search(r'https?://\S+|www\.\S+', t)),
    "emoji_non_ascii" : lambda t: bool(re.search(r'[^\x00-\x7F]', t)),
    "english_word"    : lambda t: bool(re.search(english_words, t, re.IGNORECASE)),
}

n = len(texts)
print("\n=== Slang & Noise Audit ===")
audit = {}
for name, fn in patterns.items():
    hits = sum(1 for t in texts if fn(t))
    pct  = hits / n * 100
    audit[name] = {"count": hits, "pct": round(pct, 2)}
    print(f"  {name:<20} {hits:>5}  ({pct:.1f}%)")

# ── Save audit for doc ─────────────────────────────────────────────────────────
with open("data/slang_audit.json", "w") as f:
    json.dump(audit, f, indent=2)

print("\nDone. Audit saved to data/slang_audit.json")
