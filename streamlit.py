# streamlit_yoga_intellij.py
"""
Streamlit app using MoveNet + classifier trained from YogaIntelliJ, with both frame-level and overall workout feedback.
Run with:
  streamlit run streamlit.py
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
st.set_page_config(page_title="Yoga Pose Analyzer", page_icon="🧘️")

import sys
from pathlib import Path
import tempfile
import cv2
import numpy as np
import pandas as pd
import requests
from PIL import Image
from dotenv import load_dotenv
import tensorflow as tf
from tensorflow import keras
import openai
import time

# ───────────────────────────────────────────────────────────────
# Setup path and load modules
REPO_ROOT = Path(__file__).parent
CLASS_MODEL_PATH = REPO_ROOT / "classification_model"
sys.path.insert(0, str(CLASS_MODEL_PATH))
try:
    from movenet import Movenet
    from data import person_from_keypoints_with_scores, BodyPart
except ImportError as e:
    st.error(f"❌ Failed to import backend modules: {e}")
    st.stop()

# ───────────────────────────────────────────────────────────────
# Load environment
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY", "")
if not openai.api_key:
    st.error("🚨 Set OPENAI_API_KEY in .env")
    st.stop()

# ───────────────────────────────────────────────────────────────
# Load MoveNet model
def load_movenet():
    model_file = CLASS_MODEL_PATH / "movenet_thunder.tflite"
    if not model_file.exists():
        st.error(f"❌ Missing MoveNet model at {model_file}")
        st.stop()
    return Movenet(str(model_file))
movenet = st.cache_resource(load_movenet)()

# ───────────────────────────────────────────────────────────────
# Embedding functions

def get_center_point(landmarks, left_bp, right_bp):
    left = tf.gather(landmarks, left_bp.value, axis=1)
    right = tf.gather(landmarks, right_bp.value, axis=1)
    return left * 0.5 + right * 0.5

def normalize_pose_landmarks(landmarks):
    center = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    center = tf.expand_dims(center, axis=1)
    center = tf.broadcast_to(center, [tf.shape(landmarks)[0], 17, 2])
    landmarks -= center
    torso_size = tf.linalg.norm(
        get_center_point(landmarks, BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER)
        - get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    )
    max_dist = tf.reduce_max(tf.linalg.norm(landmarks, axis=-1))
    size = tf.maximum(torso_size * 2.5, max_dist)
    landmarks /= size
    return landmarks

def landmarks_to_embedding(xy_scores):
    reshaped = tf.reshape(xy_scores, (1, 17, 3))
    norm = normalize_pose_landmarks(reshaped[:, :, :2])
    return tf.reshape(norm, (1, 34))

# ───────────────────────────────────────────────────────────────
# Load classifier
def load_classifier():
    df = pd.read_csv(CLASS_MODEL_PATH / "train_data.csv")
    classes = df['class_name'].unique().tolist()
    inp = keras.Input(shape=(34,))
    x = keras.layers.Dense(128, activation='relu')(inp)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    out = keras.layers.Dense(len(classes), activation='softmax')(x)
    model = keras.Model(inp, out)
    model.load_weights(CLASS_MODEL_PATH / "weights.best.hdf5")
    return model, classes
classifier, class_names = st.cache_resource(load_classifier)()

# ───────────────────────────────────────────────────────────────
# Frame-level processing
def process_frame(frame: np.ndarray):
    h, w = frame.shape[:2]
    person = movenet.detect(frame)
    lm = [(kp.coordinate.x / w, kp.coordinate.y / h, kp.score) for kp in person.keypoints]
    flat = np.array(lm, dtype=np.float32).flatten()
    if len(flat) != 51:
        return None, None, None
    emb = landmarks_to_embedding(flat)
    preds = classifier.predict(emb, verbose=0)[0]
    idx = int(np.argmax(preds))
    label = class_names[idx]
    conf = preds[idx]
    vis = frame.copy()
    for kp in person.keypoints:
        cv2.circle(vis, (int(kp.coordinate.x), int(kp.coordinate.y)), 4, (0,255,0), -1)
    cv2.putText(vis, f"{label} ({conf:.2f})", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    return Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)), label, conf

# ───────────────────────────────────────────────────────────────
# Video-level analysis
def analyze_video_metrics(path, sample_rate=10):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    label_counts = {}
    confidences = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if i % sample_rate != 0:
            i += 1
            continue
        i += 1
        _, lbl, cf = process_frame(frame)
        if lbl:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
            confidences.append(cf)
    cap.release()
    avg_conf = round(float(np.mean(confidences)) if confidences else 0, 2)
    return {
        "total_frames": total,
        "sampled_frames": len(confidences),
        "average_confidence": avg_conf,
        "label_distribution": label_counts
    }

# ───────────────────────────────────────────────────────────────
# Overall AI feedback (migrated to new API)
def get_overall_feedback(metrics):
    prompt = (
        "You are a workout coach. Summary metrics for the user session:\n"
        f"{metrics}\n"
        "Provide a concise report: performance strengths, two improvement tips, and motivational line."
    )
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.7
    )
    return resp.choices[0].message.content.strip()

# ───────────────────────────────────────────────────────────────
# Streamlit UI
st.title("🧘 Complete Workout Analyzer & Coach")
st.write("Upload or URL a video, then browse frames for frame-level analysis or get an overall report.")

# Input
col1, col2 = st.columns([1,2])
with col1:
    source = st.radio("Input", ["Upload", "URL"])
with col2:
    overall = st.button("Get Overall Report")

video_path = None
temp_files = []
if source == "Upload":
    f = st.file_uploader("Video", type=["mp4","mov","avi"])
    if f:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); tmp.write(f.read()); tmp.flush(); video_path=tmp.name; temp_files.append(video_path)
else:
    url = st.text_input("Video URL")
    if url:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        for chunk in requests.get(url, stream=True).iter_content(8192): tmp.write(chunk)
        tmp.flush(); video_path=tmp.name; temp_files.append(video_path)

if video_path:
    cap = cv2.VideoCapture(video_path)
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    frame_idx = st.slider("Frame", 0, cnt-1, 0)
    if st.checkbox("Show frame-level analysis"):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if ret:
            img, lbl, cf = process_frame(frame)
            if img:
                st.image(img, use_container_width=True)
                st.metric("Pose", lbl)
                st.metric("Confidence", f"{cf:.2f}")
    if overall:
        with st.spinner("Analyzing entire video..."):
            metrics = analyze_video_metrics(video_path)
        st.subheader("📊 Overall Metrics")
        st.json(metrics)
        report = get_overall_feedback(metrics)
        st.subheader("📝 Overall Feedback")
        st.write(report)

# Cleanup
import atexit
for f in temp_files:
    atexit.register(lambda f=f: os.remove(f) if os.path.exists(f) else None)