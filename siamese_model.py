"""
Face Recognition via Siamese Network
========================================
Implements face *recognition* (as opposed to just detection) using a
Siamese network architecture: a CNN embedding model that maps face
crops to a vector space where the same person's faces are close
together and different people's faces are far apart, trained with
triplet loss (anchor, positive, negative).

This is the same family of approach used by FaceNet/ArcFace, scaled
down to something trainable on a laptop for learning purposes.

Author: <your name>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    """Small CNN that maps a face image to a fixed-length embedding
    vector. In production you'd swap this for a pretrained backbone
    (e.g. InceptionResNetV1 as used in FaceNet, or an ArcFace model),
    but this version is intentionally lightweight and fully
    trainable from scratch for educational clarity.
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)  # L2-normalize so distances are comparable


class TripletLoss(nn.Module):
    """Pulls (anchor, positive) embeddings together and pushes
    (anchor, negative) embeddings apart by at least `margin`.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = (anchor - positive).pow(2).sum(dim=1)
        neg_dist = (anchor - negative).pow(2).sum(dim=1)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class FaceVerifier:
    """High-level wrapper: embeds two face images and decides if
    they're the same person based on embedding distance vs a
    threshold (tunable per-deployment using a validation set).
    """

    def __init__(self, model: EmbeddingNet, threshold: float = 0.8, device: str = None):
        self.model = model
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, face_tensor: torch.Tensor) -> torch.Tensor:
        """face_tensor: (batch, 3, H, W), already preprocessed."""
        return self.model(face_tensor.to(self.device))

    @torch.no_grad()
    def verify(self, face1: torch.Tensor, face2: torch.Tensor) -> dict:
        emb1 = self.embed(face1.unsqueeze(0))
        emb2 = self.embed(face2.unsqueeze(0))
        distance = (emb1 - emb2).pow(2).sum().sqrt().item()
        return {
            "distance": round(distance, 4),
            "same_person": distance < self.threshold,
            "threshold": self.threshold,
        }
