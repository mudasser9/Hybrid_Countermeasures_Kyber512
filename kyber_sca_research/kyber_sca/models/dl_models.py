"""
Deep Learning models for Side-Channel Analysis on Kyber512.

Three architectures:
  1. MLP         — baseline fully-connected
  2. LSTM_SCA    — sequence model for trace dependencies
  3. Transformer_SCA — attention-based for long-range correlations

All models output: logits over HW classes (0..16 for 12-bit values).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


HW_CLASSES = 17   # Hamming weight of 12-bit value ∈ [0, 16]


# -------------------------------------------------------
# 1. Baseline MLP
# -------------------------------------------------------
class MLP_SCA(nn.Module):
    """
    Simple MLP baseline.
    Input:  (B, trace_len)
    Output: (B, HW_CLASSES)
    """
    def __init__(self, trace_len: int = 1024, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(trace_len, hidden),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, HW_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------------
# 2. LSTM-SCA
# -------------------------------------------------------
class LSTM_SCA(nn.Module):
    """
    Bidirectional LSTM for side-channel trace analysis.
    Treats the trace as a time-series — captures temporal leakage patterns.

    Input:  (B, trace_len)  → reshaped to (B, T, 1) as a 1D time series
    Output: (B, HW_CLASSES)
    """
    def __init__(self,
                 trace_len: int = 1024,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.trace_len = trace_len

        # Initial projection: 1 → d_model
        self.input_proj = nn.Linear(1, hidden_size)

        self.lstm = nn.LSTM(
            input_size  = hidden_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            bidirectional = True,
            dropout = dropout if num_layers > 1 else 0.0,
        )

        # Attention pooling over time
        self.attn = nn.Linear(hidden_size * 2, 1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, HW_CLASSES),
        )

    def forward(self, x):
        # x: (B, T)
        B, T = x.shape
        x = x.unsqueeze(-1)             # (B, T, 1)
        x = self.input_proj(x)          # (B, T, H)
        h, _ = self.lstm(x)             # (B, T, 2H)

        # Attention pooling
        attn_w = torch.softmax(self.attn(h), dim=1)   # (B, T, 1)
        ctx = (attn_w * h).sum(dim=1)                  # (B, 2H)

        return self.classifier(ctx)


# -------------------------------------------------------
# 3. Transformer-SCA
# -------------------------------------------------------
class TransformerSCA(nn.Module):
    """
    Transformer encoder for side-channel analysis.
    Multi-head self-attention captures long-range leakage dependencies.

    Architecture:
        Patch embedding → Positional encoding →
        N × TransformerEncoderLayer → CLS token → Classifier

    Input:  (B, trace_len)
    Output: (B, HW_CLASSES)
    """
    def __init__(self,
                 trace_len:  int = 1024,
                 patch_size: int = 16,
                 d_model:    int = 64,
                 n_heads:    int = 4,
                 n_layers:   int = 3,
                 dropout:    float = 0.2):
        super().__init__()
        assert trace_len % patch_size == 0
        self.patch_size  = patch_size
        self.n_patches   = trace_len // patch_size
        self.d_model     = d_model

        # Patch embedding: (B, T) → (B, n_patches, d_model)
        self.patch_embed = nn.Linear(patch_size, d_model)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding (learned)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_patches + 1, d_model)
        )

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_model * 4,
            dropout         = dropout,
            activation      = 'gelu',
            batch_first     = True,
            norm_first      = True,   # Pre-LN (more stable)
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, HW_CLASSES),
        )

    def forward(self, x):
        B, T = x.shape
        # Patchify
        x = x.view(B, self.n_patches, self.patch_size)   # (B, P, patch)
        x = self.patch_embed(x)                            # (B, P, d)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)            # (B, 1, d)
        x   = torch.cat([cls, x], dim=1)                  # (B, P+1, d)
        x   = x + self.pos_embed

        # Encode
        x = self.encoder(x)                               # (B, P+1, d)
        x = self.norm(x[:, 0])                            # CLS token → (B, d)

        return self.classifier(x)


# -------------------------------------------------------
# Factory function
# -------------------------------------------------------
def build_model(arch: str, trace_len: int = 1024) -> nn.Module:
    arch = arch.lower()
    if arch == 'mlp':
        return MLP_SCA(trace_len=trace_len)
    elif arch == 'lstm':
        return LSTM_SCA(trace_len=trace_len)
    elif arch == 'transformer':
        return TransformerSCA(trace_len=trace_len)
    else:
        raise ValueError(f"Unknown architecture: {arch}. Choose mlp|lstm|transformer")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
