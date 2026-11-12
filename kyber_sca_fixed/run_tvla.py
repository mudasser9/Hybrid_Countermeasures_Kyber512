"""TVLA (fixed-vs-random secret key), 1st and 2nd order, with a fixed-vs-fixed null control."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, metrics

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

SEED = 2026
N_PER_CLASS = 5000
SIGMAS = [0.1, 0.5]
N_FIXED_KEYS = 3          # repeat with 3 different fixed secrets; report the worst case
OFFSET = 1                # 2nd order: centred product of adjacent samples (share pair is emitted adjacently)


def run():
    res = {"seed": SEED, "n_per_class": N_PER_CLASS, "sigmas": SIGMAS, "t_threshold": metrics.T_THRESH,
           "n_fixed_keys": N_FIXED_KEYS, "second_order_offset": OFFSET, "results": []}
    figdata = {}
    ss = np.random.SeedSequence(SEED)
    for sigma in SIGMAS:
        for mode in sim.MODES:
            for kidx in range(N_FIXED_KEYS):
                rng = np.random.default_rng(ss.spawn(1)[0])
                s_fixed = sim.sample_secret(rng, 1)
                t0 = time.time()
                A, _ = sim.gen_traces(mode, np.repeat(s_fixed, N_PER_CLASS, 0), rng, sigma)           # fixed
                B, _ = sim.gen_traces(mode, sim.sample_secret(rng, N_PER_CLASS), rng, sigma)          # random
                C, _ = sim.gen_traces(mode, np.repeat(s_fixed, N_PER_CLASS, 0), rng, sigma)           # fixed (null)
                t1 = metrics.welch_t(A, B)
                t2 = metrics.second_order_t(A, B, OFFSET)
                n1 = metrics.welch_t(A, C)
                n2 = metrics.second_order_t(A, C, OFFSET)
                row = {"sigma": sigma, "mode": mode, "fixed_key_idx": kidx,
                       "order1": metrics.tvla_summary(t1), "order2": metrics.tvla_summary(t2),
                       "null_order1": metrics.tvla_summary(n1), "null_order2": metrics.tvla_summary(n2),
                       "sec": round(time.time() - t0, 1)}
                res["results"].append(row)
                if kidx == 0:
                    figdata[f"t1_{mode}_{sigma}"] = t1.astype(np.float32)
                    figdata[f"t2_{mode}_{sigma}"] = t2.astype(np.float32)
                print(f"sigma={sigma} {mode} key{kidx}: o1 max|t|={row['order1']['max_abs_t']:.2f} "
                      f"({row['order1']['n_leaking']} leaking) | o2 max|t|={row['order2']['max_abs_t']:.2f} "
                      f"({row['order2']['n_leaking']}) | null o1={row['null_order1']['max_abs_t']:.2f} "
                      f"o2={row['null_order2']['max_abs_t']:.2f}  [{row['sec']}s]", flush=True)
    with open(os.path.join(OUT, "tvla.json"), "w") as f:
        json.dump(res, f, indent=1)
    np.savez_compressed(os.path.join(OUT, "tvla_curves.npz"), **figdata)
    return res


if __name__ == "__main__":
    run()
