"""
End-to-End Face Recognition Pipeline
========================================
Combines face DETECTION (Haar/DNN) with face RECOGNITION (Siamese
embedding model) into a usable pipeline:

  1. Detect face(s) in an image.
  2. Crop + preprocess each detected face.
  3. Compute an embedding for each face.
  4. Compare against a "database" of known people's embeddings to
     identify who's in the photo (nearest-neighbor match within a
     distance threshold; otherwise "Unknown").

This mirrors how production face-recognition systems are structured,
just with a small from-scratch model instead of a massive pretrained
one.

Author: <your name>
"""

import os
import json
import pickle
from typing import List, Dict, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from face_detector import HaarFaceDetector
from siamese_model import EmbeddingNet, FaceVerifier


TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class FaceRecognitionPipeline:
    def __init__(self, model_path: Optional[str] = None, threshold: float = 0.8, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        embedding_dim = 128

        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            embedding_dim = checkpoint.get("embedding_dim", 128)
            model = EmbeddingNet(embedding_dim=embedding_dim)
            model.load_state_dict(checkpoint["model_state"])
        else:
            # No trained weights yet — uses a randomly initialized
            # embedding network. Detection still works fully; matching
            # quality will be poor until train_recognition.py is run.
            model = EmbeddingNet(embedding_dim=embedding_dim)

        self.verifier = FaceVerifier(model, threshold=threshold, device=self.device)
        self.detector = HaarFaceDetector()
        self.database: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Database management
    # ------------------------------------------------------------------
    def register_face(self, name: str, image_path: str):
        """Detects the (first) face in an image and stores its
        embedding under `name` for later recognition."""
        image = cv2.imread(image_path)
        boxes = self.detector.detect(image)
        if not boxes:
            raise ValueError(f"No face detected in {image_path}")

        x, y, w, h = boxes[0]
        face_crop = image[y:y + h, x:x + w]
        embedding = self._embed_crop(face_crop)
        self.database[name] = embedding

    def save_database(self, path: str = "face_database.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.database, f)

    def load_database(self, path: str = "face_database.pkl"):
        with open(path, "rb") as f:
            self.database = pickle.load(f)

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------
    def _embed_crop(self, face_crop_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        tensor = TRANSFORM(pil_image)
        with torch.no_grad():
            embedding = self.verifier.embed(tensor.unsqueeze(0))
        return embedding.cpu().numpy().squeeze(0)

    def recognize(self, image_path: str) -> List[dict]:
        """Detects all faces in an image and matches each against the
        registered database. Returns a list of dicts with box,
        matched name (or 'Unknown'), and distance.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(image_path)

        boxes = self.detector.detect(image)
        results = []

        for box in boxes:
            x, y, w, h = box
            face_crop = image[y:y + h, x:x + w]
            embedding = self._embed_crop(face_crop)

            name, distance = self._match(embedding)
            results.append({"box": box, "name": name, "distance": distance})

        return results

    def _match(self, embedding: np.ndarray):
        if not self.database:
            return "Unknown (no database)", None

        best_name, best_distance = None, float("inf")
        for name, db_embedding in self.database.items():
            distance = float(np.linalg.norm(embedding - db_embedding))
            if distance < best_distance:
                best_distance = distance
                best_name = name

        if best_distance < self.verifier.threshold:
            return best_name, round(best_distance, 4)
        return "Unknown", round(best_distance, 4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Register or recognize faces.")
    sub = parser.add_subparsers(dest="command", required=True)

    reg = sub.add_parser("register")
    reg.add_argument("name")
    reg.add_argument("image")

    rec = sub.add_parser("recognize")
    rec.add_argument("image")

    parser.add_argument("--model", default=None, help="Path to trained recognition model")
    parser.add_argument("--db", default="face_database.pkl")

    args = parser.parse_args()
    pipeline = FaceRecognitionPipeline(model_path=args.model)

    if os.path.exists(args.db):
        pipeline.load_database(args.db)

    if args.command == "register":
        pipeline.register_face(args.name, args.image)
        pipeline.save_database(args.db)
        print(f"Registered '{args.name}' from {args.image}")
    else:
        results = pipeline.recognize(args.image)
        for r in results:
            print(f"Box {r['box']} -> {r['name']} (distance: {r['distance']})")
