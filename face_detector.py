"""
Face Detection
================
Detects faces in images or video using two interchangeable backends:

  1. Haar Cascades (OpenCV's classic, fast, CPU-friendly detector)
  2. Deep learning-based detector (OpenCV's DNN module using a
     pre-trained ResNet-SSD face detector — more accurate, especially
     for off-angle / low-light faces)

Author: <your name>
"""

import os
import cv2
import numpy as np
from typing import List, Tuple

Box = Tuple[int, int, int, int]  # (x, y, w, h)


class HaarFaceDetector:
    """Classic, fast, CPU-friendly face detector using OpenCV's
    built-in Haar Cascade classifier. Great baseline, struggles more
    with non-frontal faces or poor lighting than DNN-based detectors.
    """

    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")

    def detect(self, image: np.ndarray, scale_factor: float = 1.1,
               min_neighbors: int = 5, min_size: Tuple[int, int] = (30, 30)) -> List[Box]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size
        )
        return [tuple(map(int, box)) for box in faces]


class DnnFaceDetector:
    """Deep-learning-based face detector using OpenCV's DNN module
    with a pre-trained ResNet-10 SSD model (Caffe). More robust than
    Haar cascades for varied poses/lighting. Model weights are
    downloaded automatically on first use (see download_models.py).
    """

    PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

    def __init__(self, model_dir: str = "models", confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        proto_path = os.path.join(model_dir, "deploy.prototxt")
        model_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")

        if not (os.path.exists(proto_path) and os.path.exists(model_path)):
            raise FileNotFoundError(
                f"Model files not found in '{model_dir}'. Run "
                f"`python download_models.py` first to fetch them."
            )

        self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

    def detect(self, image: np.ndarray) -> List[Box]:
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.confidence_threshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes


def draw_boxes(image: np.ndarray, boxes: List[Box], color=(0, 255, 0), thickness=2,
               labels: List[str] = None) -> np.ndarray:
    output = image.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
        if labels and i < len(labels):
            cv2.putText(output, labels[i], (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect faces in an image.")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--backend", choices=["haar", "dnn"], default="haar")
    parser.add_argument("--out", default="detected.jpg")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    detector = HaarFaceDetector() if args.backend == "haar" else DnnFaceDetector()
    boxes = detector.detect(image)
    print(f"Detected {len(boxes)} face(s) using {args.backend} backend.")

    output = draw_boxes(image, boxes)
    cv2.imwrite(args.out, output)
    print(f"Saved annotated image to {args.out}")
