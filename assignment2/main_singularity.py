# main_singularity.py
import os
import json
import time
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from tokenizer import IndicSentencePieceTokenizer
from model import TransformerEncoder, TripletLoss
from dataset import (
    get_indic_processor,
    build_triplet_dataloaders,
    build_phase2_dataloader,
    safe_read_csv
)

# ===============================
# CONFIG
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_FILE = "train.log"
LOG_INTERVAL = 100  # log every 100 batches


# ===============================
# Utility Logging
# ===============================
def log(message):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    msg = f"{timestamp} {message}"
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }, path)
    log(f"Saved checkpoint -> {path}")


# ===============================
# Phase 1 Training
# ===============================
def train_phase1(model, optimizer, tokenizer, loader, epochs=5, save_dir="checkpoints"):
    model.train()
    triplet_loss = TripletLoss(margin=0.5)
    log(f"Starting Phase 1 training for {epochs} epochs")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(loader, start=1):
            a_ids = batch['anchor_ids'].to(DEVICE)
            p_ids = batch['pos_ids'].to(DEVICE)
            n_ids = batch['neg_ids'].to(DEVICE)
            a_mask = batch['anchor_mask'].to(DEVICE)
            p_mask = batch['pos_mask'].to(DEVICE)
            n_mask = batch['neg_mask'].to(DEVICE)

            emb_a = model(a_ids, a_mask)
            emb_p = model(p_ids, p_mask)
            emb_n = model(n_ids, n_mask)
            loss = triplet_loss(emb_a, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % LOG_INTERVAL == 0:
                avg = total_loss / batch_idx
                log(f"[Phase1][Epoch {epoch}] Batch {batch_idx}/{len(loader)} | Loss={loss.item():.6f} | Avg={avg:.6f}")

        avg_epoch = total_loss / max(1, len(loader))
        elapsed = (time.time() - start_time) / 60
        log(f"Phase1 Epoch {epoch} done | AvgLoss={avg_epoch:.6f} | Time={elapsed:.2f} min")
        save_checkpoint(model, optimizer, epoch, os.path.join(save_dir, f"phase1_epoch{epoch}.pt"))


# ===============================
# Phase 2 Training
# ===============================
def train_phase2(model, optimizer, tokenizer, loader, epochs=5, save_dir="checkpoints"):
    model.train()
    criterion = nn.CrossEntropyLoss()
    log(f"Starting Phase 2 training for {epochs} epochs")

    for epoch in range(1, epochs + 1):
        total_loss, correct, total = 0.0, 0, 0
        start_time = time.time()

        for batch_idx, batch in enumerate(loader, start=1):
            ids = batch['input_ids'].to(DEVICE)
            masks = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            logits, _ = model(ids, masks)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

            if batch_idx % LOG_INTERVAL == 0:
                acc = correct / total
                avg = total_loss / batch_idx
                log(f"[Phase2][Epoch {epoch}] Batch {batch_idx}/{len(loader)} | Loss={loss.item():.6f} | Avg={avg:.6f} | Acc={acc:.4f}")

        avg_loss = total_loss / max(1, len(loader))
        avg_acc = correct / max(1, total)
        elapsed = (time.time() - start_time) / 60
        log(f"Phase2 Epoch {epoch} done | AvgLoss={avg_loss:.6f} | Acc={avg_acc:.4f} | Time={elapsed:.2f} min")
        save_checkpoint(model, optimizer, epoch, os.path.join(save_dir, f"phase2_epoch{epoch}.pt"))


# ===============================
# Evaluation
# ===============================
def evaluate_on_bhasha(model, tokenizer, processor, bhasha_csv="bhasha-abhijnaanam.csv", label_map_path="label2id.json", max_len=256):
    model.eval()
    df = pd.read_csv(bhasha_csv, encoding="utf-8", on_bad_lines="skip", engine="python")
    with open(label_map_path, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {int(v): k for k, v in label2id.items()}

    y_true, y_pred = [], []
    log(f"Starting evaluation on {len(df)} samples")

    with torch.no_grad():
        for i, row in enumerate(df.itertuples(), start=1):
            text = processor.process(row.text, getattr(row, 'label', None))
            enc = tokenizer.batch_encode([text], max_length=max_len)
            ids = enc['input_ids'].to(DEVICE)
            masks = enc['attention_mask'].to(DEVICE)
            logits, _ = model(ids, masks)
            pred = int(torch.argmax(logits, dim=-1).item())
            gold_name = getattr(row, 'label', None)
            gold = label2id.get(gold_name, None)
            if gold is None:
                continue
            y_true.append(gold)
            y_pred.append(pred)

            if i % 200 == 0:
                log(f"Eval progress: {i}/{len(df)} samples...")

    acc = accuracy_score(y_true, y_pred) * 100
    report = classification_report(y_true, y_pred, target_names=[id2label[i] for i in sorted(id2label)])
    log(f"Evaluation complete | Accuracy={acc:.4f}%")
    with open("bhasha_eval_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}%\n\n{report}")
    log("Saved report -> bhasha_eval_report.txt")


# ===============================
# Main
# ===============================
def main():
    processor = get_indic_processor()

    tokenizer = IndicSentencePieceTokenizer(vocab_size=64000)
    if not os.path.exists("indic_tokenizer.model"):
        log("⚙Training SentencePiece tokenizer...")
        tokenizer.train(["bhasha-abhijnaanam.csv", "phase1.csv", "phase2.csv"], sample_limit=200000)
    else:
        tokenizer.load("indic_tokenizer.model")
        log("Loaded existing SentencePiece tokenizer")

    # ========== PHASE 1 ==========
    loader_nre = build_triplet_dataloaders("phase1.csv", processor, tokenizer, batch_size=128)
    model = TransformerEncoder(
        vocab_size=len(tokenizer),
        embed_dim=256,
        num_layers=6,
        num_heads=8,
        ff_dim=1024,
        phase="phase1"
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    train_phase1(model, optimizer, tokenizer, loader_nre, epochs=1)
    save_checkpoint(model, optimizer, 1, "checkpoints/phase1_final.pt")

    # ========== PHASE 2 ==========
    model.phase = "phase2"
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(256, 22)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    loader_p2 = build_phase2_dataloader("phase2.csv", processor, tokenizer, batch_size=128)
    train_phase2(model, optimizer, tokenizer, loader_p2, epochs=2)
    save_checkpoint(model, optimizer, 2, "checkpoints/phase2_final.pt")

    # ========== EVALUATION ==========
    evaluate_on_bhasha(model, tokenizer, processor, bhasha_csv="bhasha-abhijnaanam.csv", label_map_path="label2id.json")


if __name__ == "__main__":
    log("======== TRAINING STARTED ========")
    log(f"Device: {DEVICE}")
    main()
    log("======== TRAINING COMPLETED ========")
