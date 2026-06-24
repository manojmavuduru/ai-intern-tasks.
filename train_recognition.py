"""
Triplet Dataset + Training Script for the Siamese Face Recognition Model
============================================================================
Expects a directory structure like:

    dataset/
        person_1/
            img1.jpg
            img2.jpg
        person_2/
            img1.jpg
            ...

Each subfolder = one identity (this is the standard layout used by
datasets like LFW — Labeled Faces in the Wild). For each training
step, a triplet (anchor, positive, negative) is sampled: anchor and
positive are different photos of the same person; negative is a
photo of a different person.

Usage:
    python train_recognition.py --data_dir dataset --epochs 20
"""

import os
import random
import argparse
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from siamese_model import EmbeddingNet, TripletLoss


TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class TripletFaceDataset(Dataset):
    def __init__(self, data_dir: str, transform=TRANSFORM, samples_per_epoch: int = 2000):
        self.data_dir = data_dir
        self.transform = transform
        self.samples_per_epoch = samples_per_epoch

        self.identities = {
            person: [os.path.join(data_dir, person, f) for f in os.listdir(os.path.join(data_dir, person))
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            for person in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, person))
        }
        self.identities = {k: v for k, v in self.identities.items() if len(v) >= 2}
        self.people = list(self.identities.keys())

        if len(self.people) < 2:
            raise ValueError(
                "Need at least 2 identities, each with >= 2 images, to form triplets."
            )

    def __len__(self):
        return self.samples_per_epoch

    def _load(self, path):
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx):
        anchor_person = random.choice(self.people)
        negative_person = random.choice([p for p in self.people if p != anchor_person])

        anchor_path, positive_path = random.sample(self.identities[anchor_person], 2)
        negative_path = random.choice(self.identities[negative_person])

        return self._load(anchor_path), self._load(positive_path), self._load(negative_path)


def train(data_dir: str, epochs: int = 20, batch_size: int = 32, lr: float = 1e-3,
          embedding_dim: int = 128, out_path: str = "face_recognition_model.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = TripletFaceDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = EmbeddingNet(embedding_dim=embedding_dim).to(device)
    criterion = TripletLoss(margin=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for anchor, positive, negative in loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            loss = criterion(emb_a, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs} — triplet loss: {avg_loss:.4f}")

    torch.save({"model_state": model.state_dict(), "embedding_dim": embedding_dim}, out_path)
    print(f"Saved trained model to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory with one subfolder per identity")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--out", default="face_recognition_model.pt")
    args = parser.parse_args()

    train(args.data_dir, args.epochs, args.batch_size, args.lr, args.embedding_dim, args.out)
