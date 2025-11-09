import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel


# -------------------------------------------------------
# Unified BERT-like model supporting Phase1 & Phase2
# -------------------------------------------------------
class BertPhasedModel(nn.Module):
    def __init__(
        self,
        vocab_size=32000,
        num_langs=22,
        embed_dim=768,
        num_layers=6,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1,
        phase="phase1"
    ):
        super().__init__()
        self.phase = phase

        # 🧱 BERT Config (random initialization)
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=embed_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=ff_dim,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout
        )

        self.bert = BertModel(config)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # 🔹 Add classification head only for phase2
        if phase == "phase2":
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(embed_dim, num_langs)
            )

        self._init_weights()

    def _init_weights(self):
        """Reinitialize weights as in BERT (for from-scratch training)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def set_phase(self, phase, num_langs=None):
        """Switch between Phase1 and Phase2 dynamically."""
        self.phase = phase
        if phase == "phase2" and not hasattr(self, "classifier"):
            if num_langs is None:
                raise ValueError("num_langs must be provided when switching to phase2")
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.bert.config.hidden_size, num_langs)
            )

    def forward(self, input_ids, attention_mask=None, return_hidden=False):
        # Run BERT encoder
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [B, L, D]

        # Mean pooling
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).type_as(hidden)
            sum_hidden = (hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-6)
            sent_emb = sum_hidden / denom
        else:
            sent_emb = hidden.mean(dim=1)

        sent_emb = self.layer_norm(sent_emb)

        # -------------------------------
        # Phase 1: Embedding output
        # -------------------------------
        if self.phase == "phase1":
            sent_emb = F.normalize(sent_emb, p=2, dim=-1)
            if return_hidden:
                return sent_emb, hidden
            return sent_emb

        # -------------------------------
        # Phase 2: Classification output
        # -------------------------------
        elif self.phase == "phase2":
            logits = self.classifier(sent_emb)
            if return_hidden:
                return logits, sent_emb, hidden
            return logits, sent_emb


# -------------------------------------------------------
# Losses for Phase1 and Phase2
# -------------------------------------------------------
class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
            margin=margin
        )

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)


class ClassificationLoss(nn.Module):
    def __init__(self, loss_type="ce"):
        super().__init__()
        if loss_type == "ce":
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_type == "label_smoothing":
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        else:
            raise ValueError("Unsupported loss type")

    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)

# # phase1
# model = BertPhasedModel(phase="phase1", vocab_size=32000)
# triplet_loss = TripletLoss(margin=0.5)
#
# anchor_emb = model(anchor_ids, anchor_mask)
# pos_emb = model(pos_ids, pos_mask)
# neg_emb = model(neg_ids, neg_mask)
#
# loss = triplet_loss(anchor_emb, pos_emb, neg_emb)
# loss.backward()

# # phase2
# model.set_phase("phase2", num_langs=22)
# clf_loss = ClassificationLoss(loss_type="ce")
#
# logits, emb = model(input_ids, attention_mask)
# loss = clf_loss(logits, labels)
# loss.backward()
