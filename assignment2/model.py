## file: model.py


# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# TransformerEncoderBlock (Pre-Norm)
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (B, L, D), mask: (B, L) with 1 for real tokens
        key_padding_mask = None
        if mask is not None:
            # MultiheadAttention expects True for positions to be masked (PAD)
            key_padding_mask = (mask == 0)

        # Pre-norm attention
        norm_x = self.ln1(x)
        attn_out, attn_weights = self.attn(norm_x, norm_x, norm_x, key_padding_mask=key_padding_mask, need_weights=True)
        x = x + self.dropout(attn_out)

        # Pre-norm feed-forward
        norm_x = self.ln2(x)
        ff_out = self.ff(norm_x)
        x = x + self.dropout(ff_out)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        num_layers=6,
        num_heads=8,
        ff_dim=1024,
        dropout=0.1,
        max_len=256,
        num_langs=22,
        phase="phase1"
    ):
        super().__init__()
        self.phase = phase
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        if phase == "phase2":
            self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(embed_dim, num_langs))

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, input_ids, mask=None, return_attn=False):
        B, L = input_ids.shape
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.token_emb(input_ids) + self.pos_emb(pos_ids)
        x = self.dropout(x)

        attn_maps = []
        for layer in self.layers:
            # layer returns x, but we need attn weights from its internal attn
            # call ln1 + attn directly to capture weights
            norm_x = layer.ln1(x)
            attn_out, attn_w = layer.attn(norm_x, norm_x, norm_x, key_padding_mask=(mask==0) if mask is not None else None, need_weights=True)
            x = x + layer.dropout(attn_out)
            # feed-forward
            norm_x = layer.ln2(x)
            ff_out = layer.ff(norm_x)
            x = x + layer.dropout(ff_out)
            if return_attn:
                attn_maps.append(attn_w.detach().cpu())

        # Masked mean pooling
        if mask is not None:
            valid_mask = mask.unsqueeze(-1).type_as(x)
            x_sum = (x * valid_mask).sum(dim=1)
            lengths = valid_mask.sum(dim=1).clamp(min=1)
            sent_emb = x_sum / lengths
        else:
            sent_emb = x.mean(dim=1)

        if self.phase == "phase1":
            sent_emb = F.normalize(sent_emb, p=2, dim=-1)
            return (sent_emb, attn_maps) if return_attn else sent_emb
        else:
            logits = self.classifier(sent_emb)
            return (logits, sent_emb, attn_maps) if return_attn else (logits, sent_emb)


class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
            margin=margin
        )

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)