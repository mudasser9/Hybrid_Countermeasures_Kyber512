"""Profiled-attack utilities. Reuses the repository's Trainer and MLP / Transformer models unchanged."""
import contextlib, io, os, sys
import numpy as np
import torch
import torch.nn as nn

# Path to kyber_sca_research/kyber_sca of the cloned repository (override with the KYBER_SCA_REPO environment variable)
REPO = os.environ.get("KYBER_SCA_REPO", "/home/claude/work/repo/kyber_sca_research/kyber_sca")
sys.path.insert(0, REPO)
from models.dl_models import MLP_SCA, TransformerSCA          # noqa: E402  (repo code, unchanged)
from attacks.trainer import Trainer, TraceDataset              # noqa: E402  (repo code, unchanged)

N_CLASSES = 17   # same output size as the repository's models (HW of a 16-bit word)


class CNN_SCA(nn.Module):
    """Compact 1-D CNN (added here; the repository has no CNN). Three conv blocks with average pooling."""
    def __init__(self, trace_len, n_classes=N_CLASSES):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv1d(i, o, 11, padding=5), nn.BatchNorm1d(o), nn.ReLU(), nn.AvgPool1d(4))
        self.f = nn.Sequential(blk(1, 16), blk(16, 32), blk(32, 64))
        flat = 64 * (trace_len // 64)
        self.h = nn.Sequential(nn.Flatten(), nn.Linear(flat, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.h(self.f(x.unsqueeze(1)))


def build(arch, trace_len):
    return {"mlp": lambda: MLP_SCA(trace_len=trace_len),
            "cnn": lambda: CNN_SCA(trace_len),
            "transformer": lambda: TransformerSCA(trace_len=trace_len)}[arch]()


class Standardiser:
    def fit(self, X):
        self.m = X.mean(0, keepdims=True)
        self.s = X.std(0, keepdims=True)
        self.s[self.s < 1e-6] = 1.0
        return self

    def __call__(self, X):
        return ((X - self.m) / self.s).astype(np.float32)


def train(arch, Xtr, ytr, epochs, seed, batch_size=256, lr=1e-3):
    """Train with the repository Trainer (AdamW, cosine schedule, best-val checkpoint). Returns model, history."""
    torch.manual_seed(seed); np.random.seed(seed)
    ds = TraceDataset(Xtr, ytr[:, None].astype(np.int16), coeff_idx=0)
    model = build(arch, Xtr.shape[1])
    tr = Trainer(model, epochs=epochs, batch_size=batch_size, lr=lr)
    with contextlib.redirect_stdout(io.StringIO()):
        hist = tr.fit(ds)
    return model.eval(), hist


@torch.no_grad()
def log_probs(model, X, bs=1024):
    out = []
    for i in range(0, len(X), bs):
        out.append(torch.log_softmax(model(torch.from_numpy(X[i:i + bs])), -1).numpy())
    return np.concatenate(out)


class TemplateAttack:
    """Classical profiled template attack (positive control): top-k SNR points of interest + pooled-covariance LDA."""
    def __init__(self, k=10):
        self.k = k

    def fit(self, X, y):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        import metrics
        snr, _ = metrics.snr_nicv(X, y)
        self.poi = np.argsort(snr)[-self.k:]
        self.m = LinearDiscriminantAnalysis().fit(X[:, self.poi], y)
        return self

    def log_probs(self, X):
        lp = self.m.predict_log_proba(X[:, self.poi])
        full = np.full((len(X), N_CLASSES), -50.0)
        full[:, self.m.classes_] = np.maximum(lp, -50.0)
        return full
