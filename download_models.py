"""
Downloads the pre-trained DNN face-detector model weights from
OpenCV's official model repository, so DnnFaceDetector can be used.

Usage:
    python download_models.py
"""

import os
import urllib.request

MODEL_DIR = "models"
FILES = {
    "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
}


def download():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for filename, url in FILES.items():
        dest = os.path.join(MODEL_DIR, filename)
        if os.path.exists(dest):
            print(f"✓ {filename} already exists, skipping.")
            continue
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"✓ Saved to {dest}")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            print(f"  You can manually download it from: {url}")


if __name__ == "__main__":
    download()
