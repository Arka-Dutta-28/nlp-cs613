# tokenizer.py
import os
import json
from collections import Counter
import torch


class SimpleTokenizer:
    """
    A lightweight whitespace-based tokenizer that builds its own vocabulary.

    Special tokens:
      [PAD] -> id 0
      [UNK] -> id 1
    """

    def __init__(self, vocab_path=None, min_freq=2):
        self.vocab_path = vocab_path
        self.min_freq = min_freq
        self.token2id = {"[PAD]": 0, "[UNK]": 1}
        self.id2token = {0: "[PAD]", 1: "[UNK]"}

        # Load vocab if provided
        if vocab_path and os.path.exists(vocab_path):
            self.load(vocab_path)

    def build_vocab(self, texts_iterator, vocab_path='vocab.json'):
        """
        Build a vocabulary from an iterable of preprocessed (space-tokenized) texts.
        Only tokens appearing at least min_freq times are included.
        """
        counter = Counter()
        for text in texts_iterator:
            for tok in text.split():
                counter[tok] += 1

        for tok, freq in counter.most_common():
            if freq < self.min_freq:
                break
            if tok not in self.token2id:
                self.token2id[tok] = len(self.token2id)

        self.id2token = {v: k for k, v in self.token2id.items()}
        self.vocab_path = vocab_path
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)

        print(f"Vocab built ({len(self.token2id)} tokens) -> {vocab_path}")

    def __len__(self):
        return len(self.token2id)

    def tokenize(self, text: str):
        """Split text by whitespace."""
        return text.strip().split()

    def encode(self, text: str, max_length=256):
        """Convert a single text string into a list of token IDs."""
        toks = self.tokenize(text)
        ids = [self.token2id.get(t, 1) for t in toks]  # 1 = [UNK]

        # Pad or truncate
        if len(ids) < max_length:
            ids = ids + [0] * (max_length - len(ids))  # pad with 0 = [PAD]
        else:
            ids = ids[:max_length]

        return ids

    def batch_encode(self, texts, max_length=256):
        """Batch encode a list of texts into tensors."""
        input_ids = [self.encode(t, max_length=max_length) for t in texts]
        attn_mask = [[1 if tok_id != 0 else 0 for tok_id in seq] for seq in input_ids]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attn_mask, dtype=torch.long)
        }

    def save(self, path=None):
        """Save vocabulary to a file."""
        path = path or self.vocab_path
        if not path:
            raise ValueError("No vocab path provided to save()")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)
        print(f"Saved vocab -> {path}")

    def load(self, path):
        """Load vocabulary from a file."""
        with open(path, 'r', encoding='utf-8') as f:
            self.token2id = json.load(f)
        # keys might be str in JSON, so ensure ints
        self.id2token = {int(v): k for k, v in self.token2id.items()}
        self.vocab_path = path
        print(f"Loaded vocab ({len(self.token2id)} tokens) from {path}")
