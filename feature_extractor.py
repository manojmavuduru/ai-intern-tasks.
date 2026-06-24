"""
Feature Extraction for Image Captioning
=========================================
Uses a pre-trained CNN (ResNet-50 or VGG16, configurable) to extract
a fixed-length feature vector from images. These features feed into
the caption-generating decoder (RNN/LSTM, see model.py).

Author: <your name>
"""

import os
import pickle
import numpy as np
from typing import Dict

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


class FeatureExtractor:
    """Wraps a pre-trained CNN backbone (with its final classification
    layer removed) to produce image embeddings suitable for captioning.
    """

    SUPPORTED_BACKBONES = ("resnet50", "vgg16")

    def __init__(self, backbone: str = "resnet50", device: str = None):
        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(f"backbone must be one of {self.SUPPORTED_BACKBONES}")

        self.backbone_name = backbone
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.feature_dim = self._build_backbone(backbone)
        self.model.to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def _build_backbone(backbone: str):
        if backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            net = models.resnet50(weights=weights)
            feature_dim = net.fc.in_features
            net.fc = nn.Identity()  # strip classification head -> 2048-d features
        else:  # vgg16
            weights = models.VGG16_Weights.IMAGENET1K_V1
            net = models.vgg16(weights=weights)
            feature_dim = net.classifier[0].in_features  # 25088 from conv features
            net.classifier = nn.Identity()
        return net, feature_dim

    @torch.no_grad()
    def extract(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        features = self.model(tensor)
        return features.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def extract_batch(self, image_dir: str, extensions=(".jpg", ".jpeg", ".png")) -> Dict[str, np.ndarray]:
        """Extract features for every image in a directory. Returns a
        dict mapping filename -> feature vector. Useful for
        precomputing features over a training dataset (e.g. Flickr8k).
        """
        features = {}
        files = [f for f in os.listdir(image_dir) if f.lower().endswith(extensions)]
        for i, fname in enumerate(files):
            path = os.path.join(image_dir, fname)
            features[fname] = self.extract(path)
            if (i + 1) % 50 == 0:
                print(f"Extracted features for {i + 1}/{len(files)} images")
        return features

    def save_features(self, features: Dict[str, np.ndarray], out_path: str):
        with open(out_path, "wb") as f:
            pickle.dump(features, f)

    @staticmethod
    def load_features(path: str) -> Dict[str, np.ndarray]:
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract CNN features from images.")
    parser.add_argument("image_dir", help="Directory of images")
    parser.add_argument("--backbone", default="resnet50", choices=FeatureExtractor.SUPPORTED_BACKBONES)
    parser.add_argument("--out", default="features.pkl")
    args = parser.parse_args()

    extractor = FeatureExtractor(backbone=args.backbone)
    feats = extractor.extract_batch(args.image_dir)
    extractor.save_features(feats, args.out)
    print(f"Saved {len(feats)} feature vectors to {args.out}")
