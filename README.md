# Task 5 — Face Detection & Recognition

An AI application that **detects** faces in images (using either
classic Haar cascades or a deep-learning detector) and optionally
**recognizes** who they are using a from-scratch **Siamese network**
trained with triplet loss — the same family of technique behind
FaceNet and ArcFace, scaled down to something trainable without a
GPU cluster.

## Architecture

### Detection (two interchangeable backends)
- **Haar Cascades** (`HaarFaceDetector`): OpenCV's classic, CPU-fast detector. Great baseline, used by default.
- **DNN detector** (`DnnFaceDetector`): a pre-trained ResNet-10 SSD model (OpenCV's DNN module) — more robust to off-angle faces and poor lighting. Requires downloading model weights once (`download_models.py`).

### Recognition (Siamese network + triplet loss)
- `EmbeddingNet`: a small CNN that maps a face crop to a 128-d embedding vector, L2-normalized so distances are directly comparable.
- `TripletLoss`: trains the network so that (anchor, positive) pairs — two photos of the *same* person — end up close together, while (anchor, negative) pairs — different people — end up far apart.
- `FaceVerifier` / `FaceRecognitionPipeline`: wraps the trained model into a usable "register known faces, then identify new photos" workflow.

## Project structure

```
task5-face-detection-recognition/
├── face_detector.py          # Haar + DNN face detectors (CLI runnable)
├── download_models.py        # Fetches DNN detector weights
├── siamese_model.py          # EmbeddingNet, TripletLoss, FaceVerifier
├── train_recognition.py      # Trains the Siamese network on a labeled face dataset
├── recognition_pipeline.py   # End-to-end: detect -> crop -> embed -> match against database (CLI runnable)
├── app.py                    # Streamlit web UI (detect, recognize, register new faces)
├── test_face_recognition.py  # Unit tests
└── requirements.txt
```

## Running detection only (works immediately, no training needed)

```bash
pip install -r requirements.txt
python face_detector.py path/to/photo.jpg --backend haar --out detected.jpg
```

For the more accurate DNN backend:
```bash
python download_models.py   # one-time download of model weights
python face_detector.py path/to/photo.jpg --backend dnn --out detected.jpg
```

## Adding recognition (requires training on a labeled dataset)

Recognition needs photos of the **same people you want to recognize**,
so — like Task 3 — there's no way to skip a training step entirely.
The standard learning dataset is
[LFW (Labeled Faces in the Wild)](http://vis-www.cs.umass.edu/lfw/),
already organized as one folder per identity (exactly the layout
`train_recognition.py` expects):

```
dataset/
    Person_A/
        img1.jpg
        img2.jpg
    Person_B/
        img1.jpg
        ...
```

```bash
python train_recognition.py --data_dir dataset --epochs 20
```

Then register known faces and recognize new photos:
```bash
python recognition_pipeline.py register "Alice" alice_photo.jpg --model face_recognition_model.pt
python recognition_pipeline.py recognize new_photo.jpg --model face_recognition_model.pt
```

## Running the demo without training first

```bash
streamlit run app.py
```
Detection works immediately. The "Register a Face" tab lets you add
people to the recognition database — but without a trained model
(`face_recognition_model.pt`), the embeddings come from randomly
initialized weights, so matches won't be meaningful yet. The app
clearly flags this in the sidebar.

## Running tests
```bash
python -m unittest test_face_recognition.py -v
```
Tests cover detector loading/behavior, embedding shape and
normalization, triplet loss correctness, and the face-verification
distance logic — all without needing a GPU or trained weights.

## Possible extensions
- Swap the from-scratch `EmbeddingNet` for a pretrained **FaceNet (InceptionResNetV1)** or **ArcFace** model for production-quality recognition without training your own.
- Add **real-time webcam** detection/recognition using `cv2.VideoCapture`.
- Add **liveness detection** (blink detection, etc.) to guard against photo spoofing in security applications.
- Store the face database in a real vector database (e.g. FAISS, Pinecone) for fast lookup at scale.

## Concepts demonstrated
Object detection (Haar cascades, SSD), transfer learning, metric learning, triplet loss, Siamese networks, embedding spaces, nearest-neighbor matching.

## ⚠️ Ethical note
Face recognition technology raises real privacy and consent
concerns. This project is for educational purposes — if extending it
toward a real deployment, make sure you have explicit, informed
consent from anyone whose face is registered, and check applicable
local laws (e.g. BIPA, GDPR) around biometric data.
