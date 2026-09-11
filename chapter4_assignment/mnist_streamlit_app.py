"""
Streamlit app: predict handwritten digits from your own photos.

Run with:
    streamlit run mnist_streamlit_app.py

On first run it trains and caches a RandomForest + ExtraTrees VotingClassifier
on the MNIST dataset (same style as the book's Kodexempel 1), then lets you
upload a photo (e.g. taken with your phone) of a handwritten digit and
predicts it after preprocessing.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import streamlit as st
from PIL import Image, ImageOps

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------------------------
# 1. Train (and cache) the model
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training model on MNIST (first run only)...")
def train_model():
    mnist = fetch_openml("mnist_784", version=1, cache=True, as_frame=False)
    X = mnist["data"][:10000]
    y = mnist["target"][:10000].astype(np.uint8)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    et_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)

    voting_clf = VotingClassifier(
        estimators=[("rf", rf_clf), ("et", et_clf)],
        voting="soft",
    )
    voting_clf.fit(X_train, y_train)

    test_acc = accuracy_score(y_test, voting_clf.predict(X_test))
    return voting_clf, test_acc


# ---------------------------------------------------------------------------
# 2. Preprocessing: turn a phone photo into MNIST-style input
# ---------------------------------------------------------------------------
def preprocess_image(pil_img: Image.Image, threshold: int = 50):
    """
    Converts a PIL image of a handwritten digit into a
    784-length vector matching MNIST's format: grayscale,
    white digit on black background, centered, 28x28.

    Returns (feature_vector, preview_28x28_array).
    """
    img = pil_img.convert("L")          # grayscale
    img = ImageOps.invert(img)          # ink-on-paper -> MNIST's white-on-black

    img_np = np.array(img)
    img_np = np.where(img_np > threshold, img_np, 0)  # clean up shadows/noise

    # Crop tightly around the digit
    coords = np.column_stack(np.where(img_np > threshold))
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        img_np = img_np[y0 : y1 + 1, x0 : x1 + 1]
    else:
        # Nothing above threshold - return a blank canvas rather than crashing
        img_np = np.zeros((28, 28), dtype=np.uint8)

    # Pad to a square so resizing doesn't distort the digit
    h, w = img_np.shape
    size = max(h, w, 1)
    padded = np.zeros((size, size), dtype=np.uint8)
    y_off, x_off = (size - h) // 2, (size - w) // 2
    padded[y_off : y_off + h, x_off : x_off + w] = img_np

    # Add a border, since real MNIST digits aren't edge-to-edge
    border = max(size // 5, 1)
    bordered = np.zeros((size + 2 * border, size + 2 * border), dtype=np.uint8)
    bordered[border : border + size, border : border + size] = padded

    final_img = Image.fromarray(bordered).resize((28, 28), Image.LANCZOS)
    final_np = np.array(final_img).astype("float64")

    return final_np.reshape(1, -1), final_np


# ---------------------------------------------------------------------------
# 3. Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="MNIST Digit Recognizer", page_icon="🔢")
st.title("🔢 Handwritten Digit Recognizer")
st.write(
    "Upload a photo of a handwritten digit (e.g. taken with your phone) "
    "and the model will preprocess it and predict which digit it is."
)

clf, test_acc = train_model()
st.caption(f"Model test accuracy on held-out MNIST data: **{test_acc:.2%}**")

st.divider()

source = st.radio("Image source", ["Upload a file", "Use camera"], horizontal=True)

uploaded_file = None
if source == "Upload a file":
    uploaded_file = st.file_uploader(
        "Choose an image", type=["png", "jpg", "jpeg"]
    )
else:
    uploaded_file = st.camera_input("Take a photo of a handwritten digit")

threshold = st.slider(
    "Preprocessing threshold (adjust if the digit isn't detected well)",
    min_value=0,
    max_value=150,
    value=50,
    help="Pixels darker than this (after inverting) are treated as background.",
)

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(pil_img, use_column_width=True)

    features, preview = preprocess_image(pil_img, threshold=threshold)

    with col2:
        st.subheader("Preprocessed (28×28)")
        fig, ax = plt.subplots()
        ax.imshow(preview, cmap=mpl.cm.binary)
        ax.axis("off")
        st.pyplot(fig)

    pred = clf.predict(features)[0]
    probs = clf.predict_proba(features)[0]

    st.divider()
    st.subheader(f"Prediction: **{pred}**")
    st.write(f"Confidence: **{probs[pred]:.2%}**")

    st.bar_chart(
        {"digit": list(range(10)), "probability": probs},
        x="digit",
        y="probability",
    )

    if probs[pred] < 0.5:
        st.warning(
            "Low confidence — try adjusting the threshold slider, using better "
            "lighting, or making sure the digit is written clearly and fills "
            "most of the frame."
        )
else:
    st.info("Upload or take a photo of a single handwritten digit to get started.")