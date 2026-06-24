"""
Vocabulary
==========
Builds a word <-> index mapping from a corpus of captions, with
special tokens for start-of-sequence, end-of-sequence, padding, and
unknown words. Used to convert text captions into integer sequences
the decoder can consume, and back again.
"""

import re
import json
from collections import Counter
from typing import List, Iterable


class Vocabulary:
    PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}

    def build(self, captions: Iterable[str]):
        counter = Counter()
        for caption in captions:
            counter.update(self._tokenize(caption))

        specials = [self.PAD, self.START, self.END, self.UNK]
        words = [w for w, c in counter.items() if c >= self.min_freq]

        vocab = specials + sorted(words)
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9' ]", "", text)
        return text.split()

    def encode(self, caption: str, max_len: int = None) -> List[int]:
        tokens = [self.START] + self._tokenize(caption) + [self.END]
        ids = [self.word2idx.get(t, self.word2idx[self.UNK]) for t in tokens]
        if max_len:
            if len(ids) < max_len:
                ids = ids + [self.word2idx[self.PAD]] * (max_len - len(ids))
            else:
                ids = ids[:max_len]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        words = []
        for i in ids:
            word = self.idx2word.get(i, self.UNK)
            if skip_special and word in (self.PAD, self.START, self.END):
                continue
            words.append(word)
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"word2idx": self.word2idx, "min_freq": self.min_freq}, f)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(min_freq=data.get("min_freq", 2))
        vocab.word2idx = {k: int(v) for k, v in data["word2idx"].items()}
        vocab.idx2word = {v: k for k, v in vocab.word2idx.items()}
        return vocab
