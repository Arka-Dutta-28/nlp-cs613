import streamlit as st
import torch
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import pathlib
from model import TransformerEncoder
from tokenizer import IndicSentencePieceTokenizer
from dataset import get_indic_processor

# ==========================================================
# CONFIG
# ==========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = pathlib.Path(__file__).resolve().parent
TOKENIZER_FILE = str(BASE_DIR / "indic_tokenizer.model")
LABEL_MAP_FILE = str(BASE_DIR / "label2id.json")
CHECKPOINT_FILE = str(BASE_DIR / "shortest_model.pt")

MAX_LEN = 256
VOCAB_SIZE = 32000

st.set_page_config(page_title="Indic Language Identifier", layout="wide")
st.title("🌏 Indic Language Identification")
st.caption("Model inference (single + batch) using the trained TransformerEncoder.")

# ==========================================================
# LOAD ALL (cached)
# ==========================================================
@st.cache_resource(show_spinner=False)
def load_all():
    # Load processor
    processor = get_indic_processor()

    # Load tokenizer
    tokenizer = IndicSentencePieceTokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.load(TOKENIZER_FILE)

    # Load label maps
    with open(LABEL_MAP_FILE, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {int(v): k for k, v in label2id.items()}
    num_langs = len(label2id)

    # Initialize model (same as training)
    model = TransformerEncoder(
        vocab_size=len(tokenizer),
        embed_dim=256,
        num_layers=3,
        num_heads=8,
        ff_dim=1024,
        phase="phase2",
    ).to(DEVICE)

    # Add classifier head
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(256, num_langs)
    ).to(DEVICE)

    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()

    return processor, tokenizer, model, label2id, id2label, num_langs


processor, tokenizer, model, label2id, id2label, NUM_LANGS = load_all()

st.sidebar.success(f"Model loaded successfully on {DEVICE}")
st.sidebar.info(f"Languages supported: {NUM_LANGS}")

# with st.expander("Model & environment info", expanded=True):
#     st.write(f"Device: **{torch.cuda.get_device_name(0)}**")

# ==========================================================
# INFERENCE FUNCTION
# ==========================================================
def predict_texts(texts, model, tokenizer, processor, id2label, max_len=256):
    model.eval()
    preds, probs_all = [], []

    with torch.no_grad():
        for text in texts:
            text_proc = processor.process(text)
            enc = tokenizer.batch_encode([text_proc], max_length=max_len)
            ids = enc["input_ids"].to(DEVICE)
            masks = enc["attention_mask"].to(DEVICE)

            logits, _ = model(ids, masks)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            pred_id = int(np.argmax(probs))
            preds.append(id2label[pred_id])
            probs_all.append(probs)
    return preds, probs_all


# ==========================================================
# UI SECTIONS
# ==========================================================
mode = st.sidebar.radio("Select Mode", ["Single Prediction", "Batch CSV"])

# ------------------------
# SINGLE SENTENCE MODE
# ------------------------
if mode == "Single Prediction":
    st.header("Single Sentence Prediction")
    text_input = st.text_area("Enter a sentence:", "", height=100)

    if st.button("Predict Language"):
        if not text_input.strip():
            st.warning("Please enter some text to predict.")
        else:
            with st.spinner("Predicting..."):
                preds, probs_all = predict_texts([text_input], model, tokenizer, processor, id2label, MAX_LEN)
                pred = preds[0]
                probs = probs_all[0]

            st.success(f"**Predicted Language:** {pred}")
            df_probs = pd.DataFrame({
                "Language": [id2label[i] for i in range(len(probs))],
                "Confidence": probs
            }).sort_values("Confidence", ascending=False).head(10)

            st.bar_chart(df_probs.set_index("Language"))

# ------------------------
# BATCH CSV MODE
# ------------------------
elif mode == "Batch CSV":
    st.header("Batch CSV Prediction")
    uploaded = st.file_uploader("Upload CSV file with a 'text' column", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            st.write("📄 File preview:", df.head())

            import time

            if st.button("Run Predictions"):
                total_rows = len(df)
                st.info(f"Processing {total_rows:,} samples...")
                progress_bar = st.progress(0)
                status_text = st.empty()

                preds = []
                batch_size = 512  # adjust as needed
                start_time = time.time()

                with st.spinner("Running inference..."):
                    for i in range(0, total_rows, batch_size):
                        batch_start = time.time()
                        batch_texts = df["text"].iloc[i:i + batch_size].astype(str).tolist()
                        batch_preds, _ = predict_texts(batch_texts, model, tokenizer, processor, id2label, MAX_LEN)
                        preds.extend(batch_preds)

                        # Progress computation
                        progress = min((i + batch_size) / total_rows, 1.0)
                        elapsed = time.time() - start_time
                        rate = (i + batch_size) / elapsed if elapsed > 0 else 0
                        remaining = (total_rows - (i + batch_size)) / rate if rate > 0 else 0

                        # Update Streamlit UI
                        progress_bar.progress(int(progress * 100))
                        status_text.text(
                            f"✅ Processed {min(i + batch_size, total_rows)}/{total_rows} "
                            f"({progress * 100:.1f}%) | "
                            f"⏱️ Elapsed: {elapsed:.1f}s | "
                            f"⌛ ETA: {remaining:.1f}s"
                        )

                df["predicted"] = preds
                total_time = time.time() - start_time
                progress_bar.progress(100)
                status_text.text(f"✅ Prediction complete! Total time: {total_time:.1f}s")

                st.dataframe(df.head())

                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Predictions", csv_data, "predictions.csv", "text/csv")

                # If ground-truth labels are available, show metrics
                if "label" in df.columns:
                    y_true = df["label"].astype(str)
                    y_pred = df["predicted"].astype(str)
                    acc = accuracy_score(y_true, y_pred)
                    st.metric("Accuracy", f"{acc*100:.2f}%")

                    report = classification_report(y_true, y_pred, output_dict=True)
                    st.write("### Classification Report")
                    st.dataframe(pd.DataFrame(report).transpose())

                    # Confusion matrix
                    cm = confusion_matrix(y_true, y_pred, labels=sorted(id2label.values()))
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=False, cmap="Blues",
                                xticklabels=sorted(id2label.values()),
                                yticklabels=sorted(id2label.values()))
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("True")
                    st.pyplot(fig)

