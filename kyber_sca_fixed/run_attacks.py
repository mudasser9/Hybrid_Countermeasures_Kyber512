"""
Profiled-attack evaluation. Profiling/validation/test traces use RANDOM keys (one key per trace);
the rank/GE evaluation uses R target keys, each with n_att traces of that ONE fixed key.
Usage: python3 -I run_attacks.py [--quick]
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, metrics, attack

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
QUICK = "--quick" in sys.argv

SEED = 2026
N_TRAIN, N_TEST = (1500, 500) if QUICK else (6000, 2000)
R_KEYS, N_ATT = (20, 100) if QUICK else (100, 300)
EPOCHS = 2 if QUICK else 12
# (sigma, architectures) -- Transformer only at the headline sigma (CPU time)
PLAN = [(0.5, ["lda", "mlp", "cnn", "transformer"]), (0.1, ["lda", "mlp", "cnn", "transformer"])]
SHARE_DIAG = [(0.5, "cnn"), (0.1, "cnn")]   # mask-aware profiling diagnostic on C1, C2


def datasets(mode, sigma, ss):
    rng = np.random.default_rng(ss.spawn(1)[0])
    s_tr = sim.sample_secret(rng, N_TRAIN); Xtr, atr = sim.gen_traces(mode, s_tr, rng, sigma)
    s_te = sim.sample_secret(rng, N_TEST);  Xte, ate = sim.gen_traces(mode, s_te, rng, sigma)
    s_keys = sim.sample_secret(rng, R_KEYS)
    Xat, _ = sim.gen_traces(mode, np.repeat(s_keys, N_ATT, 0), rng, sigma)       # R keys x N_ATT traces, same key within a group
    return dict(Xtr=Xtr, ytr=sim.label_hw(s_tr), atr=atr, Xte=Xte, yte=sim.label_hw(s_te), ate=ate,
                Xat=Xat.reshape(R_KEYS, N_ATT, -1), yat=sim.label_hw(s_keys))


def evaluate(arch, d, seed):
    t0 = time.time()
    if arch == "lda":
        model = attack.TemplateAttack(k=10).fit(d["Xtr"], d["ytr"])
        st = lambda X: X
        predict = model.log_probs
        hist = {"val_acc": [float("nan")], "train_acc": [float((model.log_probs(d["Xtr"]).argmax(1) == d["ytr"]).mean())]}
    else:
        st = attack.Standardiser().fit(d["Xtr"])
        model, hist = attack.train(arch, st(d["Xtr"]), d["ytr"], EPOCHS, seed)
        predict = lambda X: attack.log_probs(model, X)
    sec = time.time() - t0
    lp = predict(st(d["Xte"]))
    pred = lp.argmax(1)
    top1 = float((pred == d["yte"]).mean())
    top3 = float(np.mean([y in np.argsort(l)[-3:] for l, y in zip(lp, d["yte"])]))
    maj = float(np.bincount(d["yte"]).max() / len(d["yte"]))
    lo, hi = metrics.binom_ci(int((pred == d["yte"]).sum()), len(d["yte"]))
    # rank / GE / SR on fixed-key attack groups; log-prior removed so evidence accumulates correctly
    counts = np.bincount(d["ytr"], minlength=attack.N_CLASSES).astype(float)
    valid = counts > 0                                   # attacker knows which HW classes can occur (secret distribution is public)
    log_prior = np.log(np.maximum(counts, 1.0) / counts.sum())
    lpa = predict(st(d["Xat"].reshape(-1, d["Xat"].shape[-1]))).reshape(R_KEYS, N_ATT, -1)
    ranks = metrics.rank_curves(lpa, d["yat"], log_prior, N_ATT, valid)
    ge, sr = metrics.ge_sr(ranks)
    # target-key majority baseline: rank of the true class under the prior alone (n=0 reference)
    lpv = np.where(valid, log_prior, -np.inf)
    prior_rank = float((lpv[None, :] > lpv[d["yat"]][:, None]).sum(1).mean())
    return {"arch": arch, "top1": top1, "top1_ci95": [lo, hi], "top3": top3, "majority_baseline": maj,
            "ge_at_1": float(ge[0]), "ge_at_nmax": float(ge[-1]), "sr_at_nmax": float(sr[-1]),
            "traces_for_sr90": metrics.traces_for_sr(sr, 0.9), "prior_only_rank": prior_rank,
            "best_val_acc": float(max(hist["val_acc"])), "final_train_acc": float(hist["train_acc"][-1]),
            "train_sec": round(sec, 1)}, ge, sr, lpa


def share_diag(arch, d, seed):
    """Mask-aware profiling: attacker knows the mask during profiling and learns HW(share0 of coeff 0)."""
    st = attack.Standardiser().fit(d["Xtr"])
    y_tr, y_te = d["atr"]["share0_hw0"], d["ate"]["share0_hw0"]
    model, _ = attack.train(arch, st(d["Xtr"]), y_tr, EPOCHS, seed)
    pred = attack.log_probs(model, st(d["Xte"])).argmax(1)
    acc = float((pred == y_te).mean()); maj = float(np.bincount(y_te).max() / len(y_te))
    lo, hi = metrics.binom_ci(int((pred == y_te).sum()), len(y_te))
    return {"arch": arch, "share0_top1": acc, "ci95": [lo, hi], "majority_baseline": maj}


def main():
    ss = np.random.SeedSequence(SEED)
    res = {"seed": SEED, "quick": QUICK, "n_train": N_TRAIN, "n_test": N_TEST, "r_keys": R_KEYS, "n_att": N_ATT,
           "epochs": EPOCHS, "plan": PLAN, "results": {}, "share_diag": []}
    curves = {}
    raw = {}
    path = os.path.join(OUT, "attacks" + ("_quick" if QUICK else "") + ".json")
    for sigma, archs in PLAN:
        for mode in sim.MODES:
            d = datasets(mode, sigma, ss)
            snr, nicv = metrics.snr_nicv(d["Xtr"], d["ytr"])
            res["results"].setdefault(f"{mode}@{sigma}", {"snr_peak": float(snr.max()), "snr_argmax": int(snr.argmax()),
                                                          "nicv_peak": float(nicv.max()), "models": []})
            for arch in archs:
                r, ge, sr, lpa = evaluate(arch, d, SEED)
                raw[f"lp_{mode}_{sigma}_{arch}"] = lpa.astype(np.float32)
                raw[f"y_{mode}_{sigma}"] = d["yat"]
                raw[f"counts_{mode}_{sigma}"] = np.bincount(d["ytr"], minlength=attack.N_CLASSES)
                res["results"][f"{mode}@{sigma}"]["models"].append(r)
                curves[f"ge_{mode}_{sigma}_{arch}"] = ge; curves[f"sr_{mode}_{sigma}_{arch}"] = sr
                print(f"[{mode} s={sigma} {arch}] top1={r['top1']:.3f} (maj {r['majority_baseline']:.3f}) "
                      f"GE@{N_ATT}={r['ge_at_nmax']:.2f} SR={r['sr_at_nmax']:.2f} n90={r['traces_for_sr90']} "
                      f"trainacc={r['final_train_acc']:.2f} [{r['train_sec']}s]", flush=True)
                json.dump(res, open(path, "w"), indent=1)
                np.savez_compressed(os.path.join(OUT, "attack_logp" + ("_quick" if QUICK else "") + ".npz"), **raw)
            if mode in ("C1", "C2") and any(sg == sigma for sg, _ in SHARE_DIAG):
                for sg, arch in SHARE_DIAG:
                    if sg == sigma:
                        sd = share_diag(arch, d, SEED); sd.update({"mode": mode, "sigma": sigma}); res["share_diag"].append(sd)
                        print(f"[share-diag {mode} s={sigma}] share0 HW top1={sd['share0_top1']:.3f} (maj {sd['majority_baseline']:.3f})", flush=True)
                        json.dump(res, open(path, "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, "attack_curves" + ("_quick" if QUICK else "") + ".npz"), **curves)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
