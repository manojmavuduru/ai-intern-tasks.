"""
Inference: Generate a Caption for a New Image
================================================
Loads a trained decoder checkpoint + vocabulary, extracts CNN
features for an input image, and greedily decodes a caption.

Usage:
    python infer.py path/to/image.jpg --model caption_model.pt --vocab vocab.json
"""

import argparse
import torch

from feature_extractor import FeatureExtractor
from vocabulary import Vocabulary
from model import CaptionDecoder


def generate_caption(image_path: str, model_path: str, vocab_path: str, backbone: str = "resnet50") -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab = Vocabulary.load(vocab_path)
    checkpoint = torch.load(model_path, map_location=device)

    model = CaptionDecoder(
        feature_dim=checkpoint["feature_dim"],
        vocab_size=checkpoint["vocab_size"],
        embed_dim=checkpoint["embed_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    extractor = FeatureExtractor(backbone=backbone, device=device)
    feature = extractor.extract(image_path)
    feature_tensor = torch.tensor(feature, dtype=torch.float32, device=device).unsqueeze(0)

    ids = model.generate(feature_tensor, vocab)
    caption = vocab.decode(ids.squeeze(0).tolist())
    return caption


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to an image file")
    parser.add_argument("--model", default="caption_model.pt")
    parser.add_argument("--vocab", default="vocab.json")
    parser.add_argument("--backbone", default="resnet50", choices=["resnet50", "vgg16"])
    args = parser.parse_args()

    caption = generate_caption(args.image, args.model, args.vocab, args.backbone)
    print(f"Generated caption: {caption}")
