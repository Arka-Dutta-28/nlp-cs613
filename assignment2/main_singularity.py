# main_singularity.py
import os
import json
import torch
import torch.nn as nn
import time
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

from tokenizer import SimpleTokenizer
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
LOG_INTERVAL = 100  # Log every N batches


# ===============================
# Utility Functions
# ===============================
def log(msg):
    """Write logs both to stdout and file"""
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    msg = f"{timestamp} {msg}"
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }, path)
    log(f"✅ Saved checkpoint -> {path}")


def build_or_load_vocab(tokenizer, processor, files, sample_limit=None, vocab_path='vocab.json'):
    """Build vocab if missing"""
    def iter_texts():
        for f in files:
            df = safe_read_csv(f, sample_limit=sample_limit)
            for col in df.columns:
                df[col] = df[col].astype(str).apply(lambda x: x[:5000])
                for v in df[col].astype(str).values:
                    yield processor.process(v)

    if os.path.exists(vocab_path):
        tokenizer.load(vocab_path)
        log(f"Loaded vocab from {vocab_path} ({len(tokenizer)} tokens)")
    else:
        log("⚠️ No vocab.json found, rebuilding...")
        tokenizer.build_vocab(iter_texts(), vocab_path=vocab_path)
        log(f"✅ Rebuilt vocab -> {vocab_path}")


# ===============================
# Phase 1 Training
# ===============================
def train_phase1(model, optimizer, tokenizer, loader_nre, epochs=5, save_dir='checkpoints'):
    model.train()
    triplet_loss = TripletLoss(margin=0.5)
    log(f"🚀 Starting Phase 1 Training for {epochs} epochs")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(loader_nre, start=1):
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
                log(f"[Phase1][Epoch {epoch}] Batch {batch_idx}/{len(loader_nre)} | "
                    f"Loss={loss.item():.6f} | Avg={total_loss / batch_idx:.6f}")

        avg_loss = total_loss / max(1, len(loader_nre))
        elapsed = (time.time() - start_time) / 60
        log(f"✅ Epoch {epoch} done | AvgLoss={avg_loss:.6f} | Time={elapsed:.2f} min")

        save_checkpoint(model, optimizer, epoch, os.path.join(save_dir, f'phase1_epoch{epoch}.pt'))


# ===============================
# Phase 2 Training
# ===============================
def train_phase2(model, optimizer, tokenizer, loader, epochs=5, start_epoch=1, save_dir='checkpoints'):
    model.train()
    criterion = nn.CrossEntropyLoss()
    log(f"🚀 Starting Phase 2 Training for {epochs} epochs")

    for epoch in range(start_epoch, epochs + 1):
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
                log(f"[Phase2][Epoch {epoch}] Batch {batch_idx}/{len(loader)} | "
                    f"Loss={loss.item():.6f} | Acc={acc:.4f}")

        avg_loss = total_loss / max(1, len(loader))
        avg_acc = correct / max(1, total)
        elapsed = (time.time() - start_time) / 60
        log(f"✅ Epoch {epoch} done | AvgLoss={avg_loss:.6f} | Acc={avg_acc:.4f} | Time={elapsed:.2f} min")

        save_checkpoint(model, optimizer, epoch, os.path.join(save_dir, f'phase2_epoch{epoch}.pt'))


# ===============================
# Evaluation
# ===============================
def evaluate_on_bhasha(model, tokenizer, processor, bhasha_csv='bhasha-abhijnaanam.csv', label_map_path='label2id.json', max_len=256):
    model.eval()
    df = pd.read_csv(bhasha_csv, encoding='utf-8', on_bad_lines='skip', engine='python')
    with open(label_map_path, 'r', encoding='utf-8') as f:
        label2id = json.load(f)
    id2label = {int(v): k for k, v in label2id.items()}

    y_true, y_pred = [], []
    log(f"🔍 Starting Evaluation on {bhasha_csv} ({len(df)} samples)")

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
                log(f"Eval progress: {i}/{len(df)} samples processed...")

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=[id2label[i] for i in sorted(id2label)])
    log(f"✅ Evaluation Done | Accuracy={acc:.6f}")
    with open("bhasha_eval_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.6f}\n\n")
        f.write(report)
    log("📄 Saved evaluation -> bhasha_eval_report.txt")
    return acc, report


# ===============================
# Main Entry
# ===============================
def main():
    processor = get_indic_processor()
    tokenizer = SimpleTokenizer(vocab_path='vocab.json', min_freq=2)
    build_or_load_vocab(tokenizer, processor, ['triplet_nre.csv', 'phase2.csv'], sample_limit=10000)

    # ========== PHASE 1 ==========
    model = TransformerEncoder(
        vocab_size=len(tokenizer),
        embed_dim=256,
        num_layers=6,
        num_heads=8,
        ff_dim=1024,
        phase='phase1'
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    loader_nre = build_triplet_dataloaders('triplet_nre.csv', processor, tokenizer, batch_size=256)
    train_phase1(model, optimizer, tokenizer, loader_nre, epochs=1)
    save_checkpoint(model, optimizer, 1, 'checkpoints/phase1_final.pt')

    # ========== PHASE 2 ==========
    model.set_phase('phase2', num_langs=22)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    loader_p2 = build_phase2_dataloader('phase2.csv', processor, tokenizer, batch_size=256)
    train_phase2(model, optimizer, tokenizer, loader_p2, epochs=2)
    save_checkpoint(model, optimizer, 2, 'checkpoints/phase2_final.pt')

    # ========== EVALUATION ==========
    evaluate_on_bhasha(model, tokenizer, processor, bhasha_csv='bhasha-abhijnaanam.csv', label_map_path='label2id.json')


if __name__ == "__main__":
    log("======== TRAINING STARTED ========")
    main()
    log("======== TRAINING COMPLETED ========")
