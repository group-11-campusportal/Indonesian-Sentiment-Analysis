"""
Tokenization Analysis — Indonesian Sentiment (SmSA)
Produces numbers used directly in tokenization_strategy.md.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

# Keep the model revision explicit for reproducible configuration.
TOKENIZERS = {
    "indobert": (
        "indobenchmark/indobert-base-p1",
        "main",
    ),
    "xlm_roberta": (
        "xlm-roberta-base",
        "main",
    ),
}

def stats(arr, label):
    a = np.array(arr)
    print(f"{label}: min={a.min()}  max={a.max()}  mean={a.mean():.1f}  p95={np.percentile(a,95):.0f}")

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

def analyze_tokenizers(texts, offline=False):
    # Keep the default surface/slang audit independent of Transformers.
    from transformers import AutoTokenizer

    results = {}
    print("\n=== Pretrained Tokenizer Lengths (including special tokens) ===")
    for name, (model_id, revision) in TOKENIZERS.items():
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            use_fast=True,
            local_files_only=offline,
        )
        lengths = []
        for start in range(0, len(texts), 256):
            encoded = tokenizer(
                texts[start:start + 256],
                add_special_tokens=True,
                padding=False,
                truncation=False,
            )
            lengths.extend(len(ids) for ids in encoded["input_ids"])
        a = np.array(lengths)
        result = {
            "model_id": model_id,
            "revision": revision,
            "tokenizer_class": type(tokenizer).__name__,
            "is_fast": tokenizer.is_fast,
            "min": int(a.min()),
            "max": int(a.max()),
            "mean": round(float(a.mean()), 2),
            "p95": float(np.percentile(a, 95)),
            **{f"count_over_{limit}": int((a > limit).sum())
               for limit in (128, 256, 512)},
        }
        results[name] = result
        print(f"  {model_id} @ {revision}")
        print(
            f"    min={result['min']}  max={result['max']}  "
            f"mean={result['mean']:.2f}  p95={result['p95']:g}  "
            f">128={result['count_over_128']}  "
            f">256={result['count_over_256']}  "
            f">512={result['count_over_512']}"
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parent / "data",
        help="Directory containing train_split.csv and receiving audit JSON files.",
    )
    parser.add_argument(
        "--tokenizer-analysis", action="store_true",
        help="Also measure both pinned pretrained tokenizers; downloads if needed.",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Use only cached tokenizer files (requires --tokenizer-analysis).",
    )
    args = parser.parse_args()
    if args.offline and not args.tokenizer_analysis:
        parser.error("--offline requires --tokenizer-analysis")

    train_path = args.data_dir / "train_split.csv"
    try:
        train_df = pd.read_csv(train_path)
        if "text" not in train_df.columns:
            raise ValueError("train_split.csv must contain a 'text' column")
        if train_df.empty or train_df["text"].isna().any():
            raise ValueError("training texts must be nonempty and contain no missing values")
        texts = train_df["text"].astype(str).tolist()
        if any(not text.strip() for text in texts):
            raise ValueError("training texts must not be blank")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Cannot load training data: {exc}\n")

    print("=== Surface Stats ===")
    stats([len(t) for t in texts], "Char length")
    stats([len(t.split()) for t in texts], "Word count ")

    print("\n=== Slang & Noise Audit ===")
    audit = {}
    for name, fn in patterns.items():
        hits = sum(1 for t in texts if fn(t))
        pct = hits / len(texts) * 100
        audit[name] = {"count": hits, "pct": round(pct, 2)}
        print(f"  {name:<20} {hits:>5}  ({pct:.1f}%)")

    audit_path = args.data_dir / "slang_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"\nAudit saved to {audit_path}")

    if args.tokenizer_analysis:
        try:
            results = analyze_tokenizers(texts, offline=args.offline)
        except (OSError, ImportError, ValueError) as exc:
            hint = (
                "Run once without --offline to download the pinned tokenizer files."
                if args.offline else
                "Check your connection and install dependencies from requirements.txt."
            )
            parser.exit(1, f"Tokenizer analysis failed: {exc}\n{hint}\n")
        report = {
            "source_file": train_path.name,
            "source_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
            "num_texts": len(texts),
            "settings": {
                "add_special_tokens": True, "padding": False, "truncation": False,
            },
            "versions": {
                package: importlib.metadata.version(package)
                for package in ("transformers", "tokenizers", "numpy", "pandas")
            },
            "models": results,
        }
        report_path = args.data_dir / "tokenizer_audit.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Tokenizer audit saved to {report_path}")


if __name__ == "__main__":
    main()
