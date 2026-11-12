
import numpy as np

Q = 3329
NC = 512            
T_LEN = 1536       
HW_LUT = np.array([bin(i).count("1") for i in range(4096)], dtype=np.uint8)
SCALE = 12.0
MODES = ("C0", "C1", "C2")


def sample_secret(rng, n):
    """(n, 512) secret coefficients, centred binomial eta=3 (Kyber512 eta1), mod q."""
    a = rng.integers(0, 2, (n, NC, 3)).sum(-1)
    b = rng.integers(0, 2, (n, NC, 3)).sum(-1)
    return ((a - b) % Q).astype(np.int64)


def label_hw(s):
    """Attack target: Hamming weight of s[0][0] (first coefficient of the first polynomial)."""
    return HW_LUT[s[:, 0]].astype(np.int64)


def gen_traces(mode, s, rng, sigma, T=T_LEN, p_dummy=0.25, n_ascon=64):
    """Return (traces float32 (n,T), aux dict). `s` is (n, 512). Fresh masks/permutation/dummies per trace."""
    assert mode in MODES
    n = s.shape[0]
    X = np.zeros((n, T), dtype=np.float32)
    aux = {}
    if mode == "C0":
        X[:, :NC] = HW_LUT[s] / SCALE
    else:
        r = rng.integers(0, Q, (n, NC))
        s0 = (s - r) % Q
        s1 = r
        h0 = HW_LUT[s0] / SCALE
        h1 = HW_LUT[s1] / SCALE
        aux["share0_hw0"] = HW_LUT[s0[:, 0]].astype(np.int64)   # HW of share 0 of coefficient 0 (profiling-only knowledge)
        aux["share1_hw0"] = HW_LUT[s1[:, 0]].astype(np.int64)
        if mode == "C1":
            X[:, 0:2 * NC:2] = h0
            X[:, 1:2 * NC:2] = h1
        else:  # C2: shuffle + dummies + ASCON
            rows = np.arange(n)[:, None]
            order = np.argsort(rng.random((n, NC)), axis=1)             # fresh permutation per trace
            flags = rng.random((n, NC)) < p_dummy                       # dummy op before slot k
            cf = np.cumsum(flags, axis=1)
            real = 2 * np.arange(NC)[None, :] + 2 * cf                  # start of the real op in slot k
            ok = real + 1 < T
            X[rows.repeat(NC, 1)[ok], real[ok]] = np.take_along_axis(h0, order, 1)[ok]
            X[rows.repeat(NC, 1)[ok], real[ok] + 1] = np.take_along_axis(h1, order, 1)[ok]
            ii, kk = np.nonzero(flags)
            dpos = real[ii, kk] - 2
            m = dpos + 1 < T
            ii, dpos = ii[m], dpos[m]
            X[ii, dpos] = HW_LUT[rng.integers(0, Q, ii.size)] / SCALE
            X[ii, dpos + 1] = HW_LUT[rng.integers(0, Q, ii.size)] / SCALE
            base = 2 * NC + 2 * cf[:, -1]
            idx = base[:, None] + np.arange(n_ascon)[None, :]
            ok = idx < T
            vals = HW_LUT[rng.integers(0, 4096, (n, n_ascon))] / SCALE
            X[rows.repeat(n_ascon, 1)[ok], idx[ok]] = vals[ok]
            # position (in time) at which share pair of coefficient 0 sits -- diagnostics only
            inv = np.argsort(order, axis=1)[:, 0]
            aux["pos_coeff0"] = real[np.arange(n), inv]
            aux["n_dummies"] = cf[:, -1]
    X += rng.normal(0.0, sigma, X.shape).astype(np.float32)
    return X, aux
