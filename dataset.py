"""
Dataset
=======
Loads (precomputed image feature, caption) pairs for training the
caption decoder. Expects:
  - a pickled dict of {image_filename: feature_vector} from feature_extractor.py
  - a captions file mapping {image_filename: [caption1, caption2, ...]}
    (the standard Flickr8k/Flickr30k format works well here)
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset

from vocabulary import Vocabulary


class CaptionDataset(Dataset):
    def __init__(self, features: dict, captions_map: dict, vocab: Vocabulary, max_len: int = 20):
        """
        features: {filename: np.ndarray} from FeatureExtractor
        captions_map: {filename: [caption_str, ...]}
        """
        self.vocab = vocab
        self.max_len = max_len
        self.samples = []  # (filename, caption)

        for fname, captions in captions_map.items():
            if fname not in features:
                continue
            for cap in captions:
                self.samples.append((fname, cap))

        self.features = features

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, caption = self.samples[idx]
        feature = torch.tensor(self.features[fname], dtype=torch.float32)
        token_ids = self.vocab.encode(caption, max_len=self.max_len)
        return feature, torch.tensor(token_ids, dtype=torch.long)


def load_captions_json(path: str) -> dict:
    """Loads a captions file shaped like:
    {"image1.jpg": ["a dog runs in the park", "a dog playing outside"], ...}
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
