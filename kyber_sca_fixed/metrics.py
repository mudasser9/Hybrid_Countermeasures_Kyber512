"""Leakage-assessment metrics: Welch-t TVLA (1st and 2nd order), SNR/NICV, rank/GE/SR curves."""
import numpy as np

T_THRESH = 4.5


def welch_t(a, b):
    """Per-column Welch t-statistic between sample sets a (n_a,T) and b (n_b,T)."""
    ma, mb = a.mean(0), b.mean(0)
    va, vb = a.var(0, ddof=1), b.var(0, ddof=1)
    den = np.sqrt(va / a.shape[0] + vb / b.shape[0])
    den[den == 0] = np.inf
    return (ma - mb) / den


def second_order_t(a, b, offset=1):
    """Second-order (bivariate) Welch t: per-class centred product of samples (j, j+offset)."""
    def prep(x):
        c = x - x.mean(0, keepdims=True)
        return c[:, :-offset] * c[:, offset:]
    return welch_t(prep(a), prep(b))


def tvla_summary(t, thresh=T_THRESH):
    at = np.abs(t)
    return {"max_abs_t": float(at.max()), "argmax": int(at.argmax()),
            "n_leaking": int((at > thresh).sum()), "n_samples": int(at.size)}


def snr_nicv(X, y, classes=None):
    """Per-sample SNR and NICV for labels y (n,). Classes with <2 traces are ignored."""
    classes = np.unique(y) if classes is None else classes
    means, varis, cnts = [], [], []
    for c in classes:
        m = y == c
        if m.sum() < 2:
            continue
        means.append(X[m].mean(0)); varis.append(X[m].var(0, ddof=1)); cnts.append(m.sum())
    means, varis, cnts = np.array(means), np.array(varis), np.array(cnts, float)
    w = cnts / cnts.sum()
    gm = (w[:, None] * means).sum(0)
    var_between = (w[:, None] * (means - gm) ** 2).sum(0)
    var_within = (w[:, None] * varis).sum(0)
    snr = var_between / (var_within + 1e-12)
    nicv = var_between / (var_between + var_within + 1e-12)
    return snr, nicv


def rank_curves(logp, y_true_per_key, log_prior, n_max, valid=None, map_prior=True):
    """
    logp: (R, n_att, C) log-probabilities for R target keys, each with n_att traces of ONE fixed key.
    y_true_per_key: (R,) true class of each target key. Posterior-ratio accumulation:
        score_c(n) = log prior(c) + sum_{i<=n} [ log p(c | x_i) - log prior(c) ]      (MAP; map_prior=True)
    With zero information the curve stays at the prior-only rank, the correct no-information reference.
    Returns rank (R, n_max): number of classes with strictly higher score than the true class.
    """
    s = np.cumsum(logp - log_prior[None, None, :], axis=1)[:, :n_max, :]
    if map_prior:
        s = s + log_prior[None, None, :]
    if valid is not None:                      # only classes that can occur are hypotheses
        s = np.where(valid[None, None, :], s, -np.inf)
    true = np.take_along_axis(s, y_true_per_key[:, None, None].repeat(n_max, 1), axis=2)[:, :, 0]
    return (s > true[:, :, None]).sum(2)


def ge_sr(ranks):
    """Guessing entropy (mean rank) and success rate (rank == 0) vs. number of traces."""
    return ranks.mean(0), (ranks == 0).mean(0)


def traces_for_sr(sr, target=0.9):
    idx = np.where(sr >= target)[0]
    return int(idx[0]) + 1 if idx.size else None


def binom_ci(k, n, z=1.96):
    """Wilson interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))
