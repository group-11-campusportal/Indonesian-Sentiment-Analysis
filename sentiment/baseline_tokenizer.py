"""Simple vocabulary and tokenizer for the baseline BiLSTM model."""

from collections import Counter
import re
from typing import Iterable


class WordVocabulary:
    def __init__(self, pad_token="<pad>", unk_token="<unk>", min_freq=2):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.min_freq = min_freq
        
        self.w2i = {pad_token: 0, unk_token: 1}
        self.i2w = {0: pad_token, 1: unk_token}

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def build_vocab(self, texts: Iterable[str]):
        counter = Counter()
        for text in texts:
            counter.update(self.tokenize(text))

        idx = len(self.w2i)
        for word, count in counter.items():
            if count >= self.min_freq and word not in self.w2i:
                self.w2i[word] = idx
                self.i2w[idx] = word
                idx += 1

    def encode(self, text: str, max_length: int = 128) -> list[int]:
        tokens = self.tokenize(text)[:max_length]
        unk_idx = self.w2i[self.unk_token]
        ids = [self.w2i.get(token, unk_idx) for token in tokens]
        # Pad to max_length
        ids += [self.w2i[self.pad_token]] * (max_length - len(ids))
        return ids

    def __len__(self):
        return len(self.w2i)