# Reproducing the experiments (Sections 4–5 of the revised paper)

Everything in `revised_experiments_v3.tex` that is a measured number comes from the scripts in `kyber_sca_fixed/`.
All random draws are seeded. Only the hardware subsection (5.4) and the literature comparison (5.6) are open
(`\todo` in red): the repository does not contain a reproducible source for those numbers.

## What these results are (and are not)

* **Simulated** power traces under a Hamming-weight leakage model, not measurements from an FPGA.
* Target: `HW(s[0][0])`, the Hamming weight of one secret coefficient — not full key recovery.
* Chance level for accuracy is the majority-class rate (**32.8 %**), not 1/17, because the secret
  (centred binomial, η₁ = 3) makes the label take only the values {0, 1, 2, 3, 9, 10}.
* Model limit: one share's Hamming weight carries 0 bits about the secret; the *pair* of shares carries
  0.097 bits of the 2.33 bits of a coefficient (noise-free). So C1/C2 are secure at first order by
  construction, and only second-order tests/attacks are informative (`scripts/theory.py` computes this).

## Environment

Python 3.13, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, PyTorch 2.14.1 (CPU), Matplotlib 3.11.2.

`kyber_sca_fixed/attack.py` reuses the repository's `MLP_SCA`, `TransformerSCA` and `Trainer` unchanged. Clone
<https://github.com/mudasser9/Hybrid_Countermeasures_Kyber512> and point to it:

    export KYBER_SCA_REPO=/path/to/Hybrid_Countermeasures_Kyber512/kyber_sca_research/kyber_sca

## Run order (from `kyber_sca_fixed/`)

| Step | Command | Seed | Writes (in `results/`) |
|---|---|---|---|
| 1. TVLA, 1st and 2nd order, fixed-vs-random secret, 3 fixed secrets, with fixed-vs-fixed null control | `python3 -I run_tvla.py` | 2026 | `tvla.json`, `tvla_curves.npz` |
| 2. False-alarm calibration (40 repeated fixed-vs-fixed tests per config/σ/order) | `python3 -I run_null_calibration.py` | 4242 | `null_calibration.json` |
| 3. Profiled attacks (template/LDA, MLP, CNN, Transformer; σ = 0.5 and 0.1) | `python3 -I run_attacks.py` (`--quick` for a smoke test) | 2026 | `attacks.json`, `attack_logp.npz` (30 MB, not shipped), `attack_curves.npz` |
| 4. Calibrated MAP rescoring of step 3 (no retraining) | `python3 -I rescore.py` | – | `attacks_rescored.json`, `attack_curves_rescored.npz` |
| 5. Second-order template attack (C1, C2 blind, C2 oracle) | `python3 -I run_second_order.py` | 7 | `second_order.json`, `second_order_curves.npz` |
| 6. Share-recovery diagnostic (mask-aware profiling) | `python3 -I run_share_diag_lda.py` | 11 | `share_diag_lda.json`, `tables_sharediag.tex` |
| 7. Tables and figures | `python3 -I make_report.py && python3 -I make_report_so.py` | – | `tables_*.tex`, `fig_*.png` |

Step 7 was checked to regenerate `tables_attacks`, `tables_second_order`, `tables_snr` and `tables_tvla`
byte-for-byte from the stored JSON/NPZ. `scripts/wrap_tables.py` then wraps each `tabular` in a
shrink-only `\resizebox` (used for the files in `paper_experiments/results/`).

## Protocol summary

* Profiling/validation/test traces: random secret per trace (6 000 train incl. 10 % validation, 2 000 test).
* Rank/GE/SR: 100 random target keys, each with up to 300 traces of that one key (2 000 for the second-order attack,
  profiled on 40 000 traces). Scoring: `log prior + Σ [log p(c|x) − log m_c]` with `m_c` the model's mean prediction on
  unlabelled attack traces; ranks over the six feasible classes; no-information reference GE ≈ 1.2, SR ≈ 0.33.
* TVLA: N = 5 000 per class, |t| > 4.5, second order = per-class centred product of adjacent samples; reported as
  leaking only if the threshold is exceeded for all three fixed secrets.
* Same noise σ for all configurations (σ = 0.1 and 0.5), trace length 1 536.


## Not in this package / still open

* Hardware numbers: the committed Vivado reports describe the 63-line placeholder `kyber_core` (14 LUTs, 18 FFs,
  0.098 W, setup slack −3.545 ns); none of the 8 465-LUT, 0.214 W or 4 280–10 365-cycle figures has a source in the
  repository. Re-synthesise C0, C1 and C2 with one flow and fill the table in Section 5.4.
* The repository's BiLSTM was not run (CPU time).
