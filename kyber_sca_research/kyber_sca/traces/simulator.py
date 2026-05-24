"""
Side-Channel Leakage Trace Simulator for Kyber512 Decapsulation

Simulates realistic power trace differences between:
  - UNMASKED: direct HW leakage on secret s (high SNR, easy to attack)
  - MASKED:   only share-level HW leakage (lower SNR, harder to attack)
  - HYBRID:   masked + shuffled + dummy ops + ASCON noise (lowest SNR)

The trace structure maps to the sequence of operations in
the decapsulation inner product: s^T·u using NTT.
"""

import numpy as np
from core.kyber import KyberKEM
from core.polynomial import Polynomial
from core.params import N, Q, K


TRACE_LEN = 1024


def _hw(x: int) -> int:
    return bin(int(x) & 0xFFFF).count('1')


class LeakageSimulator:
    """
    Power trace simulator for Kyber512 decapsulation.

    Produces (trace_len,) float32 arrays. The signal component of each
    trace is derived from HW of values processed during the decapsulation
    inner product s^T·u. Gaussian noise is added on top.

    Mode comparison:
        unmasked: SNR ~high  → easy profiled attack
        masked:   SNR ~mid   → harder (need more traces)
        hybrid:   SNR ~low   → hardest (shuffling + dummies break templates)
    """

    def __init__(self, mode='unmasked', noise_std=1.0, trace_len=TRACE_LEN, seed=None):
        assert mode in ('unmasked', 'masked', 'hybrid')
        self.mode      = mode
        self.noise_std = noise_std
        self.trace_len = trace_len
        self.rng       = np.random.default_rng(seed)

    # -------------------------------------------------------
    def _sig_unmasked(self, s, u, su, w) -> list:
        """
        Direct HW leakage on secret polynomial coefficients.
        Attacker observes: HW(s[i]) at each time step.
        High SNR — straightforward profiled attack.
        """
        sig = []
        for poly in s:
            # Full HW leakage on each secret coefficient
            for c in poly._c:
                sig.append(_hw(c))
        # NTT(u) processing leakage (not secret, but adds structure)
        for poly in u:
            u_ntt = poly.ntt()
            for c in u_ntt._c[:N//4]:
                sig.append(_hw(c))
        # Inner product accumulation
        for c in su._c[:N//4]:
            sig.append(_hw(c))
        # Decode stage
        for c in w._c[:N//8]:
            sig.append(_hw(c))
        return sig

    def _sig_masked(self, s, u, su, w) -> list:
        """
        First-order masked leakage.
        Secret s is split: s = s0 + s1 (mod Q).
        Attacker only sees HW(s0) and HW(s1) separately.
        Each share individually looks uniform → reduced SNR.
        Needs second-order attack or many more traces to succeed.
        """
        sig = []
        for poly in s:
            for c in poly._c:
                # Additive masking: split into two uniform-looking shares
                mask = int(self.rng.integers(0, Q))
                s0   = (c - mask) % Q
                s1   = mask
                sig.append(_hw(s0))   # share 0 leakage
                sig.append(_hw(s1))   # share 1 leakage (looks uniform)

        # NTT(u) still reveals structure (public, not secret)
        for poly in u:
            u_ntt = poly.ntt()
            for c in u_ntt._c[:N//8]:
                sig.append(_hw(c))

        # Inner product result in masked domain — masked intermediate
        for c in su._c[:N//8]:
            # Attacker sees noise-like value (masked inner product)
            masked_su = (c + int(self.rng.integers(0, Q))) % Q
            sig.append(_hw(masked_su))
        return sig

    def _sig_hybrid(self, s, u, su, w) -> list:
        """
        Hybrid countermeasure leakage.
        Layers:
          (1) Additive masking — shares only
          (2) Shuffling — random access order breaks timing templates
          (3) Dummy ops — random HW insertions at unpredictable positions
          (4) ASCON protection — su intermediate encrypted before use
        Net effect: SNR further reduced; trace length varies; no fixed POI.
        """
        sig = []

        # (1+2) Masked + shuffled coefficient processing
        for poly in s:
            order = self.rng.permutation(N//4).tolist()   # shuffled
            for idx in order:
                c    = poly._c[idx]
                mask = int(self.rng.integers(0, Q))
                s0   = (c - mask) % Q
                sig.append(_hw(s0))
                # (3) Dummy op with ~25% probability
                if self.rng.random() < 0.25:
                    dummy = int(self.rng.integers(0, Q))
                    sig.append(_hw(dummy))   # random, no correlation with c

        # (4) ASCON permutation leakage on protected su
        # ASCON processes state in 5×64-bit lanes; leakage is from permutation,
        # not from su directly → appears uncorrelated with s
        ascon_samples = int(self.rng.integers(N//8, N//4))
        ascon_leak    = [int(self.rng.integers(0, 16)) for _ in range(ascon_samples)]
        sig.extend(ascon_leak)

        # Very weak residual leakage from re-encode check (no secret here)
        for c in w._c[:N//16]:
            sig.append(_hw(c) // 4)

        return sig

    # -------------------------------------------------------
    def _build_trace(self, intermediates: dict) -> np.ndarray:
        s, u, su, w = (intermediates['s'], intermediates['u'],
                       intermediates['su'], intermediates['w'])
        if self.mode == 'unmasked':
            sig = self._sig_unmasked(s, u, su, w)
        elif self.mode == 'masked':
            sig = self._sig_masked(s, u, su, w)
        else:
            sig = self._sig_hybrid(s, u, su, w)

        arr = np.array(sig[:self.trace_len], dtype=np.float32)
        if len(arr) < self.trace_len:
            arr = np.pad(arr, (0, self.trace_len - len(arr)))

        # Normalise signal to [0,1]
        lo, hi = arr.min(), arr.max()
        if hi - lo > 0:
            arr = (arr - lo) / (hi - lo)
        return arr

    # -------------------------------------------------------
    def generate_traces(self, n_traces: int, fixed_key: bool = False):
        """
        Generate n_traces power traces.

        Returns:
            traces (N, T) float32,
            labels (N, 256) int16   — HW(s_poly[0]) per coeff,
            metas  list of dicts
        """
        print(f"[Sim] {n_traces} traces | mode={self.mode} | noise={self.noise_std:.2f}")
        pk, sk, _ = KyberKEM.keygen()

        traces = np.zeros((n_traces, self.trace_len), dtype=np.float32)
        labels = np.zeros((n_traces, N), dtype=np.int16)
        metas  = []

        for i in range(n_traces):
            if not fixed_key:
                pk, sk, _ = KyberKEM.keygen()
            ct, _      = KyberKEM.encaps(pk)
            _, ints    = KyberKEM.decaps(sk, ct)

            sig   = self._build_trace(ints)
            noise = self.rng.normal(0, self.noise_std, self.trace_len).astype(np.float32)
            traces[i] = sig + noise
            labels[i] = np.array([_hw(c) for c in ints['s'][0]._c], dtype=np.int16)
            metas.append({'idx': i, 'mode': self.mode})

            if (i + 1) % max(1, n_traces // 10) == 0:
                print(f"  [{i+1}/{n_traces}]", end='\r')

        print(f"\n[Sim] Done. shape={traces.shape}")
        return traces, labels, metas

    # -------------------------------------------------------
    def snr(self, traces: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Per-sample SNR across HW classes for coefficient 0."""
        snr = np.zeros(self.trace_len)
        for t in range(self.trace_len):
            col = traces[:, t]
            class_means, class_vars = [], []
            for hw in range(17):
                m = (labels[:, 0] == hw)
                if m.sum() < 2: continue
                class_means.append(col[m].mean())
                class_vars.append(col[m].var())
            if len(class_means) > 1:
                snr[t] = np.var(class_means) / (np.mean(class_vars) + 1e-9)
        return snr

    def nicv(self, traces: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Normalised Inter-Class Variance per sample."""
        nicv = np.zeros(self.trace_len)
        for t in range(self.trace_len):
            col = traces[:, t]
            var_total = col.var()
            if var_total < 1e-9: continue
            means = [col[labels[:, 0] == hw].mean()
                     for hw in range(17)
                     if (labels[:, 0] == hw).sum() > 0]
            nicv[t] = np.var(means) / var_total
        return nicv
