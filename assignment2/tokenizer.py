

## file: tokenizer.py
# tokenizer.py
import os
import json
from collections import Counter
import torch

class SimpleTokenizer:
    """Whitespace-based tokenizer with simple vocab build/save/load.

    Special tokens:
      [PAD] -> id 0
      [UNK] -> id 1
    """
    def __init__(self, vocab_path=None, min_freq=2):
        self.vocab_path = vocab_path
        self.min_freq = min_freq
        if vocab_path and os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.token2id = json.load(f)
            self.id2token = {int(v): k for k, v in self.token2id.items()}
        else:
            # token2id must map token->int
            self.token2id = {"[PAD]": 0, "[UNK]": 1}
            self.id2token = {0: "[PAD]", 1: "[UNK]"}

    def build_vocab(self, texts_iterator, vocab_path='vocab.json'):
        """Build vocab from an iterator of preprocessed (tokenized-with-spaces) texts.

        texts_iterator may be any iterable yielding strings. This function counts tokens
        and adds tokens with frequency >= min_freq.
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
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)
        self.vocab_path = vocab_path
        print(f"✅ Vocab built ({len(self.token2id)} tokens) -> {vocab_path}")

    def __len__(self):
        return len(self.token2id)

    def tokenize(self, text: str):
        return text.split()

    def encode(self, text: str, max_length=256):
        toks = self.tokenize(text)
        ids = [self.token2id.get(t, 1) for t in toks]
        if len(ids) < max_length:
            ids = ids + [0] * (max_length - len(ids))
        else:
            ids = ids[:max_length]
        return ids

    def batch_encode(self, texts, max_length=256):
        input_ids = [self.encode(t, max_length=max_length) for t in texts]
        attn_mask = [[1 if x != 0 else 0 for x in row] for row in input_ids]
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attn_mask, dtype=torch.long)
        }

    def save(self, path=None):
        path = path or self.vocab_path
        if path is None:
            raise ValueError('vocab path not provided')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)
        print(f"Saved vocab -> {path}")

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            self.token2id = json.load(f)
        self.id2token = {v: k for k, v in self.token2id.items()}
        self.vocab_path = path