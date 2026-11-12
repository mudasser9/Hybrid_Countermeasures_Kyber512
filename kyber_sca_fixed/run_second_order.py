"""
Second-order profiled template attack (LDA on [x_p, x_{p+1}, centred product]) against C1 and C2.

  C1            : attacker knows the share pair of coefficient 0 sits at samples (0,1)       -> realistic strong attacker
  C2 blind      : attacker uses the same fixed samples (0,1) although the order is shuffled   -> realistic
  C2 oracle     : attacker is GIVEN the time position of coefficient 0 in every trace         -> upper bound
                  (isolates what shuffling/dummies hide from what masking hides)
Features are extracted per chunk so memory stays small. Ranking uses the calibrated-MAP scoring of rescore.py.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, metrics

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SEED = 7
N_TRAIN, R_KEYS, N_ATT, CHUNK = 40000, 100, 2000, 5000
SIGMAS = [0.1, 0.5]
VARIANTS = [("C1", "C1"), ("C2", "C2 blind"), ("C2", "C2 oracle")]


def feats(X, aux, variant):
    n = X.shape[0]
    if variant == "C2 oracle":
        p = aux["pos_coeff0"]
        a = X[np.arange(n), p]; b = X[np.arange(n), p + 1]
    else:
        a, b = X[:, 0], X[:, 1]
    return np.stack([a, b], 1)


def add_product(F, mu):
    return np.column_stack([F, (F[:, 0] - mu[0]) * (F[:, 1] - mu[1])])


def collect(mode, variant, secrets, rng, sigma, chunk=CHUNK):
    out = []
    for i in range(0, len(secrets), chunk):
        X, aux = sim.gen_traces(mode, secrets[i:i + chunk], rng, sigma)
        out.append(feats(X, aux, variant))
    return np.concatenate(out)


def main():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    ss = np.random.SeedSequence(SEED)
    res = []
    curves = {}
    for sigma in SIGMAS:
        for mode, variant in VARIANTS:
            rng = np.random.default_rng(ss.spawn(1)[0])
            s_tr = sim.sample_secret(rng, N_TRAIN); y_tr = sim.label_hw(s_tr)
            F_tr = collect(mode, variant, s_tr, rng, sigma)
            mu = F_tr.mean(0)
            lda = LinearDiscriminantAnalysis().fit(add_product(F_tr, mu), y_tr)
            counts = np.bincount(y_tr, minlength=17).astype(float)
            valid = counts > 0
            log_prior = np.log(np.maximum(counts, 1) / counts.sum())
            s_keys = sim.sample_secret(rng, R_KEYS); y_keys = sim.label_hw(s_keys)
            LP = np.zeros((R_KEYS, N_ATT, 17))
            for k in range(R_KEYS):
                F = collect(mode, variant, np.repeat(s_keys[k:k + 1], N_ATT, 0), rng, sigma, chunk=N_ATT)
                lp = lda.predict_log_proba(add_product(F, mu))
                full = np.full((N_ATT, 17), -50.0); full[:, lda.classes_] = np.maximum(lp, -50.0)
                LP[k] = full
            m = np.exp(LP).reshape(-1, 17).mean(0); log_m = np.log(np.maximum(m, 1e-12))
            sc = log_prior[None, None, :] + np.cumsum(LP - log_m[None, None, :], axis=1)
            sc = np.where(valid[None, None, :], sc, -np.inf)
            true = np.take_along_axis(sc, y_keys[:, None, None].repeat(N_ATT, 1), axis=2)[:, :, 0]
            ranks = (sc > true[:, :, None]).sum(2)
            ge, sr = metrics.ge_sr(ranks)
            lpv = np.where(valid, log_prior, -np.inf)
            prior_ge = float((lpv[None, :] > lpv[y_keys][:, None]).sum(1).mean())
            row = {"sigma": sigma, "variant": variant, "ge_at_1": float(ge[0]), "ge_at_nmax": float(ge[-1]), "sr_at_nmax": float(sr[-1]),
                   "traces_sr50": metrics.traces_for_sr(sr, 0.5), "traces_sr90": metrics.traces_for_sr(sr, 0.9),
                   "prior_only_ge": prior_ge, "n_train": N_TRAIN, "r_keys": R_KEYS, "n_att": N_ATT}
            res.append(row); curves[f"ge_{variant}_{sigma}"] = ge; curves[f"sr_{variant}_{sigma}"] = sr
            print(row, flush=True)
    json.dump({"seed": SEED, "rows": res}, open(os.path.join(OUT, "second_order.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, "second_order_curves.npz"), **curves)


if __name__ == "__main__":
    main()
