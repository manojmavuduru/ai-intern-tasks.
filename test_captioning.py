import unittest
import torch

from vocabulary import Vocabulary
from model import CaptionDecoder


class TestVocabulary(unittest.TestCase):
    def setUp(self):
        self.vocab = Vocabulary(min_freq=1)
        self.vocab.build(["a dog runs in the park", "a cat sleeps on the mat"])

    def test_build_creates_mappings(self):
        self.assertIn("dog", self.vocab.word2idx)
        self.assertIn(self.vocab.START, self.vocab.word2idx)
        self.assertIn(self.vocab.END, self.vocab.word2idx)

    def test_encode_decode_roundtrip(self):
        ids = self.vocab.encode("a dog runs")
        decoded = self.vocab.decode(ids)
        self.assertIn("dog", decoded)
        self.assertIn("runs", decoded)

    def test_unknown_word_handling(self):
        ids = self.vocab.encode("a spaceship flies")
        # "spaceship" and "flies" aren't in vocab -> should map to <unk>
        self.assertIn(self.vocab.word2idx[self.vocab.UNK], ids)

    def test_padding(self):
        ids = self.vocab.encode("a dog", max_len=10)
        self.assertEqual(len(ids), 10)


class TestCaptionDecoder(unittest.TestCase):
    def setUp(self):
        self.feature_dim = 2048
        self.vocab = Vocabulary(min_freq=1)
        self.vocab.build(["a dog runs in the park"])
        self.model = CaptionDecoder(self.feature_dim, len(self.vocab), embed_dim=32, hidden_dim=64)

    def test_forward_output_shape(self):
        batch_size, seq_len = 4, 6
        features = torch.randn(batch_size, self.feature_dim)
        captions = torch.randint(0, len(self.vocab), (batch_size, seq_len))
        logits = self.model(features, captions)
        self.assertEqual(logits.shape, (batch_size, seq_len, len(self.vocab)))

    def test_generate_produces_valid_ids(self):
        features = torch.randn(1, self.feature_dim)
        ids = self.model.generate(features, self.vocab, max_len=10)
        self.assertTrue((ids >= 0).all())
        self.assertTrue((ids < len(self.vocab)).all())


if __name__ == "__main__":
    unittest.main()
