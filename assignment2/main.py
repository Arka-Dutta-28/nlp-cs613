## file: main.py
# main.py
import os
import math
import json
import torch
import torch.nn as nn
from torch.utils.data import Subset
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from tokenizer import SimpleTokenizer
from model import TransformerEncoder, TripletLoss
from dataset import (
    get_indic_processor,
    build_triplet_dataloaders,
    build_phase2_dataloader,
    collate_triplet_batch,
    collate_phase2_batch,
    Phase2Dataset
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, path)


def build_or_load_vocab(tokenizer, processor, files, sample_limit=None, vocab_path='vocab.json'):
    # sample_limit: if provided, read only first N rows per file for speed
    def iter_texts():
        for f in files:
            # read robustly in chunks
            try:
                for chunk in pd.read_csv(f, encoding='utf-8', on_bad_lines='skip', engine='python', chunksize=10000):
                    for col in chunk.columns:
                        for v in chunk[col].astype(str).values:
                            yield processor.process(v)
                    if sample_limit:
                        break
            except Exception:
                # fallback simpler read
                df = pd.read_csv(f, encoding='latin1', engine='python', nrows=sample_limit or 1000)
                for col in df.columns:
                    for v in df[col].astype(str).values:
                        yield processor.process(v)
    if tokenizer.vocab_path and os.path.exists(tokenizer.vocab_path):
        tokenizer.load(tokenizer.vocab_path)
    else:
        tokenizer.build_vocab(iter_texts(), vocab_path=vocab_path)


def train_phase1(model, optimizer, tokenizer, loader_native, loader_nre, epochs=5, max_len=256, save_dir='checkpoints'):
    model.train()
    triplet_loss = TripletLoss(margin=0.5)
    for epoch in range(1, epochs+1):
        total_loss = 0.0
        it = zip(loader_nre, loader_native)
        pbar = tqdm(it, total=min(len(loader_nre), len(loader_native)), desc=f'Phase1 E{epoch}')
        for batch_a, batch_b in pbar:
            batch = collate_triplet_batch(batch_a, batch_b, tokenizer, max_len=max_len)
            a_ids = batch['anchor_ids'].to(DEVICE); p_ids = batch['pos_ids'].to(DEVICE); n_ids = batch['neg_ids'].to(DEVICE)
            a_mask = batch['anchor_mask'].to(DEVICE); p_mask = batch['pos_mask'].to(DEVICE); n_mask = batch['neg_mask'].to(DEVICE)
            emb_a = model(a_ids, a_mask)
            emb_p = model(p_ids, p_mask)
            emb_n = model(n_ids, n_mask)
            loss = triplet_loss(emb_a, emb_p, emb_n)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        avg = total_loss / max(1, min(len(loader_nre), len(loader_native)))
        print(f'Phase1 Epoch {epoch} avg_loss={avg:.4f}')
        save_checkpoint(model, optimizer, epoch, os.path.join(save_dir, f'phase1_epoch{epoch}.pt'))


def train_phase2(model, optimizer, tokenizer, phase2_loader, epochs=5, max_len=256, save_dir='checkpoints'):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs+1):
        total_loss = 0.0; correct = 0; total = 0
        pbar = tqdm(phase2_loader, desc=f'Phase2 E{epoch}')
        for batch in pbar:
            data = collate_phase2_batch(batch, tokenizer, max_len=max_len)
            ids = data['input_ids'].to(DEVICE); masks = data['attention_mask'].to(DEVICE); labels = data['labels'].to(DEVICE)
            logits, _ = model(ids, masks)
            loss = criterion(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            preds = logits.argmax(dim=-1)
            correct += (preds==labels).sum().item(); total += labels.size(0)
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.3f}'})
        avg = total_loss / max(1, len(phase2_loader))
        print(f'Phase2 Epoch {epoch} avg_loss={avg:.4f} acc={correct/total:.4f}')
        save_checkpoint(model, optimizer, epoch, os.path.join(save_dir, f'phase2_epoch{epoch}.pt'))


def evaluate_on_bhasha(model, tokenizer, processor, bhasha_csv='bhasha-abhijnaanam.csv', label_map_path='label2id.json', max_len=256):
    model.eval()
    df = pd.read_csv(bhasha_csv, encoding='utf-8', on_bad_lines='skip', engine='python')
    # load label map
    with open(label_map_path, 'r', encoding='utf-8') as f:
        label2id = json.load(f)
    id2label = {int(v): k for k, v in label2id.items()}

    y_true = []; y_pred = []
    pbar = tqdm(df.iterrows(), total=len(df), desc='Eval Bhasha')
    with torch.no_grad():
        for _, row in pbar:
            text = processor.process(row['text'], row.get('label', None))
            enc = tokenizer.batch_encode([text], max_length=max_len)
            ids = enc['input_ids'].to(DEVICE); masks = enc['attention_mask'].to(DEVICE)
            logits, _ = model(ids, masks)
            pred = int(torch.argmax(logits, dim=-1).item())
            gold_name = row.get('label')
            gold = label2id.get(gold_name, None)
            if gold is None:
                # unknown label in mapping — skip
                continue
            y_true.append(gold); y_pred.append(pred)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=[id2label[i] for i in sorted(id2label)])
    print(f'Bhasha Eval Acc: {acc:.4f}')
    print(report)
    return acc, report


def main():
    processor = get_indic_processor()
    tokenizer = SimpleTokenizer(vocab_path='vocab.json', min_freq=2)

    # Build or load vocab (fast mode: sample_limit=1000) — set sample_limit=None to scan entire files
    build_or_load_vocab(tokenizer, processor, ['triplet_native.csv', 'triplet_nre.csv', 'phase2.csv'], vocab_path='vocab.json')

    # Phase1 loaders
    loader_native, loader_nre = build_triplet_dataloaders('triplet_native.csv', 'triplet_nre.csv', processor, tokenizer, batch_size=32)

    # small model
    model = TransformerEncoder(vocab_size=len(tokenizer), embed_dim=256, num_layers=6, num_heads=8, ff_dim=1024, phase='phase1').to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    train_phase1(model, optimizer, tokenizer, loader_native, loader_nre, epochs=5)
    save_checkpoint(model, optimizer, 5, 'checkpoints/phase1_final.pt')

    # Phase2: add classifier head and switch
    model.phase = 'phase2'
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(256, 22)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    phase2_loader = build_phase2_dataloader('phase2.csv', processor, tokenizer, batch_size=64)
    train_phase2(model, optimizer, tokenizer, phase2_loader, epochs=5)
    save_checkpoint(model, optimizer, 5, 'checkpoints/phase2_final.pt')

    # Evaluation
    evaluate_on_bhasha(model, tokenizer, processor, bhasha_csv='bhasha-abhijnaanam.csv', label_map_path='label2id.json')


if __name__ == '__main__':
    main()