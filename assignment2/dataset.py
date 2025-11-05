## file: dataset.py
# dataset.py
import os
import json
import unicodedata
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize
from typing import Optional

# language -> indicnlp code mapping (same as you provided)
lang_code_map = {
    'Assamese': 'as', 'Bodo': 'brx', 'Bangla': 'bn', 'Konkani': 'gom',
    'Gujarati': 'gu', 'Hindi': 'hi', 'Kannada': 'kn', 'Maithili': 'mai',
    'Malayalam': 'ml', 'Marathi': 'mr', 'Nepali': 'ne', 'Oriya': 'or',
    'Punjabi': 'pa', 'Sanskrit': 'sa', 'Sindhi': 'sd', 'Tamil': 'ta',
    'Telugu': 'te', 'Urdu': 'ur', 'Kashmiri': 'ks', 'Manipuri': 'mni',
    'Dogri': 'doi', 'Santali': 'sat'
}


def fallback_normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text)).split())


class IndicTextProcessor:
    def __init__(self):
        self.factory = IndicNormalizerFactory()
        self._cache = {}

    def process(self, text: str, label: Optional[str] = None):
        text = str(text).strip()
        if not text:
            return text
        code = lang_code_map.get(label, 'hi')
        try:
            if code not in self._cache:
                self._cache[code] = self.factory.get_normalizer(code)
            norm = self._cache[code].normalize(text)
            toks = indic_tokenize.trivial_tokenize(norm, code)
            return " ".join(toks)
        except Exception as e:
            # fallback
            return fallback_normalize(text)


class TripletDataset(Dataset):
    def __init__(self, csv_path, processor: IndicTextProcessor, dataset_type='native'):
        self.df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip', engine='python')
        self.processor = processor
        self.dataset_type = dataset_type
        if dataset_type == 'native':
            required = {'anchor', 'positive', 'negative'}
        else:
            required = {'native', 'roman', 'english'}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.dataset_type == 'native':
            a, p, n = row['anchor'], row['positive'], row['negative']
            label = None
        else:
            a, p, n = row['native'], row['roman'], row['english']
            label = None
        a = self.processor.process(a, label)
        p = self.processor.process(p, label)
        n = self.processor.process(n, label)
        return a, p, n, self.dataset_type


class Phase2Dataset(Dataset):
    def __init__(self, csv_path, processor: IndicTextProcessor, label_map_path='label2id.json'):
        self.df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip', engine='python')
        self.processor = processor
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r', encoding='utf-8') as f:
                self.label2id = json.load(f)
        else:
            langs = sorted(self.df['label'].unique())
            self.label2id = {lang: i for i, lang in enumerate(langs)}
            with open(label_map_path, 'w', encoding='utf-8') as f:
                json.dump(self.label2id, f, ensure_ascii=False, indent=2)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = self.processor.process(row['text'], row['label'])
        label_id = self.label2id[row['label']]
        return text, label_id


# Collate helpers that use a tokenizer with batch_encode(texts)
def collate_triplet_batch(batch_a, batch_b, tokenizer, max_len=256):
    batch = list(batch_a) + list(batch_b)
    anchors, poses, negs = [], [], []
    for a, p, n, _ in batch:
        anchors.append(a); poses.append(p); negs.append(n)
    a_enc = tokenizer.batch_encode(anchors, max_length=max_len)
    p_enc = tokenizer.batch_encode(poses, max_length=max_len)
    n_enc = tokenizer.batch_encode(negs, max_length=max_len)
    return {
        'anchor_ids': a_enc['input_ids'], 'anchor_mask': a_enc['attention_mask'],
        'pos_ids': p_enc['input_ids'], 'pos_mask': p_enc['attention_mask'],
        'neg_ids': n_enc['input_ids'], 'neg_mask': n_enc['attention_mask']
    }


def collate_phase2_batch(batch, tokenizer, max_len=256):
    texts, labels = zip(*batch)
    enc = tokenizer.batch_encode(list(texts), max_length=max_len)
    return {'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask'], 'labels': torch.tensor(labels, dtype=torch.long)}


# Builders
def build_triplet_dataloaders(native_csv, nre_csv, processor, tokenizer, batch_size=32, num_workers=4):
    ds_native = TripletDataset(native_csv, processor, dataset_type='native')
    ds_nre = TripletDataset(nre_csv, processor, dataset_type='nre')
    loader_native = DataLoader(ds_native, batch_size=batch_size//2, shuffle=True, num_workers=num_workers)
    loader_nre = DataLoader(ds_nre, batch_size=batch_size//2, shuffle=True, num_workers=num_workers)
    return loader_native, loader_nre


def build_phase2_dataloader(csv_path, processor, tokenizer, batch_size=64, num_workers=4):
    ds = Phase2Dataset(csv_path, processor)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader


def get_indic_processor():
    return IndicTextProcessor()
