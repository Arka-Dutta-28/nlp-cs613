# main.py
import os
import json
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from tokenizer import IndicSentencePieceTokenizer
from model import TransformerEncoder, TripletLoss
from dataset import (
    get_indic_processor,
    build_triplet_dataloaders,
    build_phase2_dataloader,
    collate_triplet_batch,
    collate_phase2_batch,
    safe_read_csv
)
from sklearn.metrics import classification_report, accuracy_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Using {DEVICE}")

def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }, path)


def build_or_load_vocab(tokenizer, processor, files, sample_limit=None, vocab_path='vocab.json'):
    def iter_texts():
        for f in files:
            df = safe_read_csv(f, sample_limit=sample_limit)

            for col in df.columns:
                df[col] = df[col].astype(str).apply(lambda x: x)
                for v in df[col].astype(str).values:
                    yield processor.process(v)
    if tokenizer.vocab_path and os.path.exists(tokenizer.vocab_path):
        tokenizer.load(tokenizer.vocab_path)
    else:
        tokenizer.build_vocab(iter_texts(), vocab_path=vocab_path)


def train_phase1(model, optimizer, tokenizer, loader_nre, epochs=5, save_dir='checkpoints'):
    model.train()
    triplet_loss = TripletLoss(margin=0.5)
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        pbar = tqdm(loader_nre, desc=f'Phase1 E{epoch}')
        for batch in pbar:
            # ✅ Batch is already collated and tokenized here!
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
            pbar.set_postfix({'loss': f'{loss.item():.10f}'})

        avg = total_loss / max(1, len(loader_nre))
        print(f'Phase1 Epoch {epoch} avg_loss={avg:.10f}')
        save_checkpoint(model, optimizer, epoch, os.path.join(save_dir, f'phase1_epoch{epoch}.pt'))




def train_phase2(model, optimizer, tokenizer, phase2_loader, epochs=5, start_epoch=1, save_dir='checkpoints'):
    model.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(phase2_loader, desc=f'Phase2 E{epoch}')

        for batch in pbar:
            # ✅ batch is already tokenized
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
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{(correct / total) * 100:.3f}'})

        avg = total_loss / max(1, len(phase2_loader))
        print(f'Phase2 Epoch {epoch} avg_loss={avg:.4f} acc={(correct / total) * 100:.4f}')
        save_checkpoint(model, optimizer, epoch, os.path.join(save_dir, f'phase2_epoch{epoch}.pt'))



def evaluate_on_bhasha(model, tokenizer, processor, bhasha_csv='bhasha-abhijnaanam.csv', label_map_path='label2id.json', max_len=256):
    model.eval()
    df = pd.read_csv(bhasha_csv, encoding='utf-8', on_bad_lines='skip', engine='python')

    with open(label_map_path, 'r', encoding='utf-8') as f:
        label2id = json.load(f)
    id2label = {int(v): k for k, v in label2id.items()}

    y_true, y_pred = [], []
    pbar = tqdm(df.iterrows(), total=len(df), desc='Eval Bhasha')
    with torch.no_grad():
        for _, row in pbar:
            text = processor.process(row['text'], row.get('label', None))
            enc = tokenizer.batch_encode([text], max_length=max_len)
            ids = enc['input_ids'].to(DEVICE)
            masks = enc['attention_mask'].to(DEVICE)
            logits, _ = model(ids, masks)
            pred = int(torch.argmax(logits, dim=-1).item())
            gold_name = row.get('label')
            gold = label2id.get(gold_name, None)
            if gold is None:
                continue
            y_true.append(gold)
            y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred) * 100
    report = classification_report(y_true, y_pred, target_names=[id2label[i] for i in sorted(id2label)])
    print(f'Bhasha Eval Acc: {acc:.8f}')
    print(report)
    return acc, report


def main():
    processor = get_indic_processor()

    tokenizer = IndicSentencePieceTokenizer(vocab_size=64000)
    if not os.path.exists("indic_tokenizer.model"):
        tokenizer.train(["bhasha-abhijnaanam.csv", "phase1.csv", "phase2.csv"], sample_limit=200000)
    else:
        tokenizer.load("indic_tokenizer.model")

    # Phase 1: only NRE
    loader_nre = build_triplet_dataloaders('phase1.csv', processor, tokenizer, batch_size=128)

    model = TransformerEncoder(
        vocab_size=len(tokenizer),
        embed_dim=256,
        num_layers=6,
        num_heads=8,
        ff_dim=1024,
        phase='phase1'
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    train_phase1(model, optimizer, tokenizer, loader_nre, epochs=1)
    save_checkpoint(model, optimizer, 1, 'checkpoints/phase1_final.pt')

    # Phase 2
    model.phase = 'phase2'
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(256, 22)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    phase2_loader = build_phase2_dataloader('phase2.csv', processor, tokenizer, batch_size=128)
    train_phase2(model, optimizer, tokenizer, phase2_loader, epochs=2)
    save_checkpoint(model, optimizer, 5, 'checkpoints/phase2_final.pt')

    # Evaluation
    evaluate_on_bhasha(model, tokenizer, processor, bhasha_csv='bhasha-abhijnaanam.csv', label_map_path='label2id.json')


if __name__ == '__main__':
    main()
