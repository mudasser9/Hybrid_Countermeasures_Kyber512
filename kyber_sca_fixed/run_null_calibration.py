"""Empirical false-alarm rate of the TVLA pipeline: repeat fixed-vs-fixed tests many times."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, metrics

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
REPS, N = 40, 5000
SEED = 4242

def run():
    ss = np.random.SeedSequence(SEED)
    out = []
    for sigma in (0.1, 0.5):
        for mode in sim.MODES:
            m1, m2 = [], []
            for _ in range(REPS):
                rng = np.random.default_rng(ss.spawn(1)[0])
                s_f = np.repeat(sim.sample_secret(rng, 1), N, 0)
                A, _ = sim.gen_traces(mode, s_f, rng, sigma)
                C, _ = sim.gen_traces(mode, s_f, rng, sigma)
                m1.append(np.abs(metrics.welch_t(A, C)).max())
                m2.append(np.abs(metrics.second_order_t(A, C, 1)).max())
            m1, m2 = np.array(m1), np.array(m2)
            row = {"sigma": sigma, "mode": mode, "reps": REPS,
                   "o1_false_alarms": int((m1 > metrics.T_THRESH).sum()), "o1_max_of_max": float(m1.max()), "o1_median_max": float(np.median(m1)),
                   "o2_false_alarms": int((m2 > metrics.T_THRESH).sum()), "o2_max_of_max": float(m2.max()), "o2_median_max": float(np.median(m2))}
            out.append(row)
            print(row, flush=True)
    json.dump({"seed": SEED, "n_per_class": N, "rows": out}, open(os.path.join(OUT, "null_calibration.json"), "w"), indent=1)

if __name__ == "__main__":
    run()
