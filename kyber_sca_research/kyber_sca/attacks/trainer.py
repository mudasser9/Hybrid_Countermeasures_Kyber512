"""
Training and evaluation engine for DL-SCA models.

Supports:
  - Profiled attack training (fixed key, HW label regression)
  - Guessing entropy (GE) and success rate computation
  - Key rank evaluation
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from typing import Tuple, Dict, List
import time


# -------------------------------------------------------
# Dataset helper
# -------------------------------------------------------
class TraceDataset:
    """Wraps traces and HW labels into PyTorch tensors."""

    def __init__(self, traces: np.ndarray, labels: np.ndarray,
                 coeff_idx: int = 0):
        """
        Args:
            traces:    (N, T) float32
            labels:    (N, 256) int16 — HW per coefficient
            coeff_idx: which coefficient index to predict (default 0)
        """
        self.X = torch.tensor(traces,    dtype=torch.float32)
        # Use per-coefficient HW as class label
        self.y = torch.tensor(labels[:, coeff_idx], dtype=torch.long)

    def split(self, val_frac: float = 0.1):
        n     = len(self.X)
        n_val = int(n * val_frac)
        ds    = TensorDataset(self.X, self.y)
        return random_split(ds, [n - n_val, n_val])


# -------------------------------------------------------
# Trainer
# -------------------------------------------------------
class Trainer:

    def __init__(self,
                 model:      nn.Module,
                 device:     str = 'cpu',
                 lr:         float = 1e-3,
                 batch_size: int = 256,
                 epochs:     int = 30):
        self.model      = model.to(device)
        self.device     = device
        self.batch_size = batch_size
        self.epochs     = epochs
        self.optim      = torch.optim.AdamW(model.parameters(), lr=lr,
                                            weight_decay=1e-4)
        self.sched      = torch.optim.lr_scheduler.CosineAnnealingLR(
                              self.optim, T_max=epochs)
        self.loss_fn    = nn.CrossEntropyLoss()
        self.history    = {'train_loss': [], 'val_loss': [],
                           'train_acc':  [], 'val_acc':  []}

    def _epoch(self, loader: DataLoader, train: bool) -> Tuple[float, float]:
        self.model.train(train)
        total_loss, total_correct, total_n = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                loss   = self.loss_fn(logits, y)
                if train:
                    self.optim.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optim.step()
                total_loss    += loss.item() * len(y)
                total_correct += (logits.argmax(1) == y).sum().item()
                total_n       += len(y)
        return total_loss / total_n, total_correct / total_n

    def fit(self, dataset: TraceDataset, val_frac: float = 0.1) -> Dict:
        train_ds, val_ds = dataset.split(val_frac)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size,
                              shuffle=True,  num_workers=0)
        val_dl   = DataLoader(val_ds,   batch_size=self.batch_size,
                              shuffle=False, num_workers=0)

        best_val_acc  = 0.0
        best_state    = None

        print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
              f"{'Val Loss':>8} | {'Val Acc':>8}")
        print("-" * 60)

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            tl, ta = self._epoch(train_dl, train=True)
            vl, va = self._epoch(val_dl,   train=False)
            self.sched.step()

            self.history['train_loss'].append(tl)
            self.history['val_loss'].append(vl)
            self.history['train_acc'].append(ta)
            self.history['val_acc'].append(va)

            if va > best_val_acc:
                best_val_acc = va
                best_state   = {k: v.cpu().clone()
                                for k, v in self.model.state_dict().items()}

            print(f"{epoch:>6} | {tl:>10.4f} | {ta:>9.4f} | "
                  f"{vl:>8.4f} | {va:>8.4f}  [{time.time()-t0:.1f}s]")

        # Restore best weights
        if best_state:
            self.model.load_state_dict(best_state)

        print(f"\nBest val accuracy: {best_val_acc:.4f}")
        return self.history


# -------------------------------------------------------
# Attack evaluator — Guessing Entropy & Key Rank
# -------------------------------------------------------
class AttackEvaluator:
    """
    Computes standard SCA metrics:
      - Key rank (lower = easier attack)
      - Guessing entropy (average number of guesses)
      - Success rate (fraction of experiments recovering key)
      - Minimum traces-to-disclosure (TTD)
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model  = model.to(device)
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def get_log_probs(self, traces: np.ndarray) -> np.ndarray:
        """Run model inference, return log-softmax probabilities."""
        X  = torch.tensor(traces, dtype=torch.float32).to(self.device)
        bs = 512
        out = []
        for i in range(0, len(X), bs):
            logits = self.model(X[i:i+bs])
            out.append(torch.log_softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0)

    def key_rank_curve(self,
                       traces:     np.ndarray,
                       true_labels: np.ndarray,
                       n_experiments: int = 100,
                       max_traces:    int = None) -> np.ndarray:
        """
        Compute average key rank as a function of number of traces used.

        Args:
            traces:         (N, T) attack traces
            true_labels:    (N,)  ground-truth HW values
            n_experiments:  number of random sub-experiments to average
            max_traces:     limit trace count (default: all)

        Returns:
            ranks: (max_traces,) array of average key rank at each trace count
        """
        N = len(traces)
        if max_traces is None:
            max_traces = N

        log_probs = self.get_log_probs(traces)    # (N, 17)
        true_hw   = true_labels                    # (N,)

        ranks = np.zeros(max_traces)

        for _ in range(n_experiments):
            idx  = np.random.permutation(N)[:max_traces]
            lp   = log_probs[idx]
            hw   = true_hw[idx]

            # Accumulate log-probabilities for each HW hypothesis
            accumulated = np.zeros(17)
            for t in range(max_traces):
                accumulated += lp[t]
                # Rank of the true key (lower is better)
                rank_t = (accumulated > accumulated[hw[t]]).sum()
                ranks[t] += rank_t

        return ranks / n_experiments

    def attack_accuracy(self,
                        traces:      np.ndarray,
                        true_labels: np.ndarray) -> Dict:
        """Compute top-1 and top-3 accuracy of the attack."""
        log_probs = self.get_log_probs(traces)
        pred      = log_probs.argmax(axis=1)
        top1      = (pred == true_labels).mean()
        top3      = sum(
            true_labels[i] in log_probs[i].argsort()[-3:]
            for i in range(len(true_labels))
        ) / len(true_labels)
        return {'top1': float(top1), 'top3': float(top3)}

    def traces_to_disclosure(self,
                             rank_curve: np.ndarray,
                             threshold: int = 0) -> int:
        """
        Find minimum number of traces for rank to drop to `threshold`.
        Returns len(rank_curve) if never reached.
        """
        idxs = np.where(rank_curve <= threshold)[0]
        return int(idxs[0]) + 1 if len(idxs) > 0 else len(rank_curve)
