#!/usr/bin/env python3
"""
Master Experiment Runner — Kyber512 SCA Research
"""

import sys, os, argparse, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traces.simulator       import LeakageSimulator, TRACE_LEN
from models.dl_models       import build_model, count_params
from attacks.trainer        import TraceDataset, Trainer, AttackEvaluator
from countermeasures.hybrid import HybridKyberKEM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n_traces', type=int, default=3000)
    p.add_argument('--epochs',   type=int, default=20)
    p.add_argument('--arch',     type=str, default='lstm',
                   choices=['mlp', 'lstm', 'transformer'])
    p.add_argument('--quick',    action='store_true')
    p.add_argument('--out_dir',  type=str, default='results/')
    return p.parse_args()


def generate_dataset(mode, n, noise, seed):
    sim = LeakageSimulator(mode=mode, noise_std=noise, seed=seed)
    traces, labels, _ = sim.generate_traces(n, fixed_key=True)
    return traces, labels, sim


def train_model(arch, traces, labels, epochs):
    ds      = TraceDataset(traces, labels, coeff_idx=0)
    model   = build_model(arch, trace_len=TRACE_LEN)
    trainer = Trainer(model, epochs=epochs, batch_size=256)
    history = trainer.fit(ds)
    return model, history


def evaluate_model(model, traces, labels, n_exp=20):
    evaluator  = AttackEvaluator(model)
    acc        = evaluator.attack_accuracy(traces, labels[:, 0])
    rank_curve = evaluator.key_rank_curve(
        traces, labels[:, 0], n_experiments=n_exp,
        max_traces=min(400, len(traces)))
    ttd = evaluator.traces_to_disclosure(rank_curve, threshold=0)
    return acc, rank_curve, ttd


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.quick:
        args.n_traces = 500
        args.epochs   = 5
        print("[QUICK MODE]")

    noise_cfg = {'unmasked': 0.5, 'masked': 1.0, 'hybrid': 2.0}
    results   = {}
    curves    = {}

    print(f"\n{'='*60}")
    print(f"  KYBER512 DL-SCA  |  arch={args.arch}  |  traces={args.n_traces}")
    print(f"{'='*60}")

    # Verify hybrid KEM
    print("\n[0] Hybrid KEM correctness check...")
    hybrid = HybridKyberKEM()
    for i in range(3):
        pk, sk, _ = hybrid.keygen()
        ct, ss1   = hybrid.encaps(pk)
        ss2, _    = hybrid.decaps(sk, ct)
        print(f"    Trial {i+1}: {'OK' if ss1==ss2 else 'FAIL'}")

    for mode in ['unmasked', 'masked', 'hybrid']:
        print(f"\n{'='*60}")
        print(f"  MODE: {mode.upper()}")
        print(f"{'='*60}")

        noise = noise_cfg[mode]

        traces, labels, sim = generate_dataset(mode, args.n_traces, noise, 42)

        snr_vals  = sim.snr(traces, labels)
        nicv_vals = sim.nicv(traces, labels)

        n_train = int(0.8 * len(traces))
        tr_train, lb_train = traces[:n_train], labels[:n_train]
        tr_test,  lb_test  = traces[n_train:], labels[n_train:]

        print(f"\n[Train] {args.arch.upper()}, epochs={args.epochs}")
        t0 = time.time()
        model, history = train_model(args.arch, tr_train, lb_train, args.epochs)
        elapsed = time.time() - t0

        acc, rank_curve, ttd = evaluate_model(model, tr_test, lb_test)

        results[mode] = {
            'top1_acc':  round(acc['top1'], 4),
            'top3_acc':  round(acc['top3'], 4),
            'ttd':       ttd,
            'max_snr':   round(float(snr_vals.max()), 4),
            'mean_nicv': round(float(nicv_vals.mean()), 6),
            'train_sec': round(elapsed, 1),
            'n_params':  count_params(model),
        }
        curves[mode] = rank_curve.tolist()

        print(f"\n  → top1={acc['top1']:.4f}  top3={acc['top3']:.4f}  "
              f"TTD={ttd}  maxSNR={snr_vals.max():.4f}")

    # Summary table
    print(f"\n{'='*60}")
    print("  FINAL COMPARISON")
    print(f"{'='*60}")
    labels_map = {'unmasked':'Unmasked (base)', 'masked':'Masked Kyber', 'hybrid':'Hybrid (ours)'}
    print(f"{'Design':<18} | {'Top-1':>6} | {'Top-3':>6} | {'TTD':>6} | {'MaxSNR':>7}")
    print("-"*55)
    for m, r in results.items():
        print(f"{labels_map[m]:<18} | {r['top1_acc']:>6.4f} | {r['top3_acc']:>6.4f} | "
              f"{r['ttd']:>6} | {r['max_snr']:>7.4f}")

    out = os.path.join(args.out_dir, 'results.json')
    json.dump({'config': vars(args), 'results': results, 'rank_curves': curves},
              open(out,'w'), indent=2)
    print(f"\n[Saved] {out}")
    return results


if __name__ == '__main__':
    main()
