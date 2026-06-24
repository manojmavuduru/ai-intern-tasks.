"""
Caption Decoder Model
======================
An LSTM-based decoder that generates captions word-by-word, conditioned
on a CNN image-feature vector (from feature_extractor.py). This
follows the classic "Show and Tell" / encoder-decoder architecture:

    Image -> CNN (encoder, frozen/pretrained) -> feature vector
    feature vector -> Linear projection -> LSTM initial hidden state
    LSTM (decoder) generates caption tokens one at a time

Author: <your name>
"""

import torch
import torch.nn as nn


class CaptionDecoder(nn.Module):
    def __init__(self, feature_dim: int, vocab_size: int, embed_dim: int = 256,
                 hidden_dim: int = 512, num_layers: int = 1):
        super().__init__()
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, image_features: torch.Tensor, captions: torch.Tensor):
        """Teacher-forced training forward pass.

        image_features: (batch, feature_dim)
        captions: (batch, seq_len) integer token ids, including <start> but
                  the model is trained to predict the *next* token at
                  each step (so captions[:, :-1] is fed in, captions[:, 1:]
                  is the target — handled by the training loop).
        """
        batch_size = image_features.size(0)
        h0 = self.feature_projection(image_features).unsqueeze(0)
        h0 = h0.repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)

        embeddings = self.embedding(captions)
        outputs, _ = self.lstm(embeddings, (h0, c0))
        logits = self.fc(outputs)
        return logits

    @torch.no_grad()
    def generate(self, image_features: torch.Tensor, vocab, max_len: int = 20):
        """Greedy decoding: generate a caption one token at a time
        starting from <start>, feeding each prediction back in as the
        next input, until <end> or max_len is reached.
        """
        self.eval()
        device = image_features.device
        batch_size = image_features.size(0)

        h = self.feature_projection(image_features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c = torch.zeros_like(h)

        token = torch.full((batch_size, 1), vocab.word2idx[vocab.START], dtype=torch.long, device=device)
        generated = [token]

        for _ in range(max_len):
            embed = self.embedding(token)
            out, (h, c) = self.lstm(embed, (h, c))
            logits = self.fc(out.squeeze(1))
            token = logits.argmax(dim=-1, keepdim=True)
            generated.append(token)
            if (token == vocab.word2idx[vocab.END]).all():
                break

        ids = torch.cat(generated, dim=1)
        return ids
