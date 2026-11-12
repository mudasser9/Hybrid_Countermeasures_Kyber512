"""
Calibrated MAP rescoring of the saved attack log-probabilities (no retraining).

score_c(n) = log prior_c + sum_{i<=n} [ log p(c|x_i) - log m_c ]
  prior_c : class frequency in the profiling set (secret distribution is public)
  m_c     : the model's own mean predicted distribution over unlabelled attack traces (removes the
            miscalibration bias of models that did not learn, which would otherwise be amplified n times)
Hypotheses are restricted to classes that occur in profiling.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    raw = np.load(os.path.join(R, "attack_logp.npz"))
    out, curves = {}, {}
    for key in raw.files:
        if not key.startswith("lp_"):
            continue
        _, mode, sigma, arch = key.split("_")
        lp = raw[key].astype(np.float64)                                   # (R, N, 17)
        y = raw[f"y_{mode}_{sigma}"]; counts = raw[f"counts_{mode}_{sigma}"].astype(float)
        valid = counts > 0
        log_prior = np.log(np.maximum(counts, 1.0) / counts.sum())
        m = np.exp(lp).reshape(-1, lp.shape[-1]).mean(0)
        log_m = np.log(np.maximum(m, 1e-12))
        n = lp.shape[1]
        s = log_prior[None, None, :] + np.cumsum(lp - log_m[None, None, :], axis=1)
        s = np.where(valid[None, None, :], s, -np.inf)
        true = np.take_along_axis(s, y[:, None, None].repeat(n, 1), axis=2)[:, :, 0]
        ranks = (s > true[:, :, None]).sum(2)
        ge, sr = metrics.ge_sr(ranks)
        out[f"{mode}@{sigma}|{arch}"] = {"ge_at_nmax": float(ge[-1]), "sr_at_nmax": float(sr[-1]),
                                         "ge_at_1": float(ge[0]), "traces_for_sr90": metrics.traces_for_sr(sr, 0.9),
                                         "traces_for_sr50": metrics.traces_for_sr(sr, 0.5)}
        curves[f"ge_{mode}_{sigma}_{arch}"] = ge; curves[f"sr_{mode}_{sigma}_{arch}"] = sr
    json.dump(out, open(os.path.join(R, "attacks_rescored.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(R, "attack_curves_rescored.npz"), **curves)
    for k, v in out.items():
        print(f"{k:28s} GE@N={v['ge_at_nmax']:.2f} SR@N={v['sr_at_nmax']:.2f} n50={v['traces_for_sr50']} n90={v['traces_for_sr90']}")


if __name__ == "__main__":
    main()
