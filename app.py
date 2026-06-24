"""
Streamlit web interface for Face Detection & Recognition.
Run with: streamlit run app.py
"""

import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from face_detector import HaarFaceDetector, draw_boxes
from recognition_pipeline import FaceRecognitionPipeline

st.set_page_config(page_title="Face Detection & Recognition", page_icon="🙂", layout="wide")
st.title("🙂 Face Detection & Recognition")
st.caption("Haar cascade detection + Siamese-network-based recognition")

DB_PATH = "face_database.pkl"
MODEL_PATH = "face_recognition_model.pt"

if "pipeline" not in st.session_state:
    model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else None
    st.session_state.pipeline = FaceRecognitionPipeline(model_path=model_path)
    if os.path.exists(DB_PATH):
        st.session_state.pipeline.load_database(DB_PATH)

pipeline: FaceRecognitionPipeline = st.session_state.pipeline

tab1, tab2 = st.tabs(["🔍 Detect & Recognize", "➕ Register a Face"])

with tab1:
    st.subheader("Upload an image")
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="detect_upload")

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        detector = HaarFaceDetector()
        boxes = detector.detect(img_array)

        if not boxes:
            st.warning("No faces detected. Try a clearer, more front-facing photo.")
            st.image(image, use_container_width=True)
        else:
            labels = []
            if pipeline.database:
                tmp_path = "tmp_detect.jpg"
                cv2.imwrite(tmp_path, img_array)
                results = pipeline.recognize(tmp_path)
                os.remove(tmp_path)
                labels = [r["name"] for r in results]
                st.write(f"Detected **{len(boxes)}** face(s):")
                for r in results:
                    st.write(f"- Box {r['box']} → **{r['name']}** (distance: {r['distance']})")
            else:
                st.info("No registered faces in the database yet — showing detection only. Register faces in the other tab to enable recognition.")
                labels = [f"Face {i+1}" for i in range(len(boxes))]

            annotated = draw_boxes(img_array, boxes, labels=labels)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_container_width=True)

with tab2:
    st.subheader("Register a known face")
    st.write("Upload a clear photo of one person's face and give them a name. "
             "Their face embedding will be stored so future photos of them can be recognized.")

    name = st.text_input("Name")
    reg_upload = st.file_uploader("Photo", type=["jpg", "jpeg", "png"], key="register_upload")

    if st.button("Register") and name and reg_upload:
        image = Image.open(reg_upload).convert("RGB")
        img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        tmp_path = "tmp_register.jpg"
        cv2.imwrite(tmp_path, img_array)
        try:
            pipeline.register_face(name, tmp_path)
            pipeline.save_database(DB_PATH)
            st.success(f"Registered '{name}'! ({len(pipeline.database)} people in database)")
        except ValueError as e:
            st.error(str(e))
        finally:
            os.remove(tmp_path)

    if pipeline.database:
        st.write("**Currently registered:**", ", ".join(pipeline.database.keys()))

with st.sidebar:
    st.header("About")
    st.markdown(
        "**Detection**: OpenCV Haar Cascade classifier finds face "
        "bounding boxes (a DNN-based detector is also available in "
        "`face_detector.py` for better accuracy — see README).\n\n"
        "**Recognition**: a small CNN (Siamese network) trained with "
        "triplet loss maps each face to a 128-d embedding. New faces "
        "are matched against registered embeddings by distance.\n\n"
        "⚠️ The bundled recognition model is **untrained** "
        "(random weights) until you run `train_recognition.py` on a "
        "labeled face dataset — see README for instructions. "
        "Detection works fully out of the box regardless."
    )
