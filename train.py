"""
Training Script for Image Captioning Model
=============================================
End-to-end training loop: builds vocabulary from captions, loads
precomputed image features, and trains the LSTM decoder with
teacher forcing using cross-entropy loss.

Usage:
    python train.py --features features.pkl --captions captions.json --epochs 20

Expected inputs:
    features.pkl   -> output of feature_extractor.py (dict: filename -> np.ndarray)
    captions.json  -> dict: filename -> list of caption strings
                       (this is the standard shape for Flickr8k/Flickr30k
                       once converted to JSON; see README for the
                       conversion snippet)
"""

import argparse
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from vocabulary import Vocabulary
from dataset import CaptionDataset, load_captions_json
from model import CaptionDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Path to features.pkl")
    parser.add_argument("--captions", required=True, help="Path to captions.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--out", default="caption_model.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with open(args.features, "rb") as f:
        features = pickle.load(f)
    captions_map = load_captions_json(args.captions)

    all_captions = [c for caps in captions_map.values() for c in caps]
    vocab = Vocabulary(min_freq=2)
    vocab.build(all_captions)
    vocab.save("vocab.json")
    print(f"Vocabulary size: {len(vocab)}")

    dataset = CaptionDataset(features, captions_map, vocab, max_len=args.max_len)
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    feature_dim = next(iter(features.values())).shape[0]
    model = CaptionDecoder(feature_dim, len(vocab), args.embed_dim, args.hidden_dim).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx[vocab.PAD])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for feats, caps in train_loader:
            feats, caps = feats.to(device), caps.to(device)
            inputs, targets = caps[:, :-1], caps[:, 1:]

            logits = model(feats, inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for feats, caps in val_loader:
                feats, caps = feats.to(device), caps.to(device)
                inputs, targets = caps[:, :-1], caps[:, 1:]
                logits = model(feats, inputs)
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= max(1, len(val_loader))
        print(f"Epoch {epoch}/{args.epochs} — train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "feature_dim": feature_dim,
                "vocab_size": len(vocab),
                "embed_dim": args.embed_dim,
                "hidden_dim": args.hidden_dim,
            }, args.out)
            print(f"  -> saved new best model to {args.out}")

    print("Training complete.")


if __name__ == "__main__":
    main()
