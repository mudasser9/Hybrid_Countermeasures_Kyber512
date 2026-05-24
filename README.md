# Masked Kyber512 + Hybrid Countermeasure Research Implementation

## Overview
Complete DL-SCA research pipeline for CRYSTALS-Kyber512, implementing the base paper
and a proposed hybrid countermeasure at the decapsulation phase.

## Structure
```
core/           Kyber512 KEM (KeyGen/Encaps/Decaps) + verified NTT
traces/         Power trace simulator (HW model, three leakage modes)
models/         MLP / LSTM / Transformer attack models
attacks/        Training engine + key rank / GE evaluator
countermeasures/  Hybrid: masking + shuffling + dummies + ASCON
verilog/        FPGA hardware stub (Vivado/Artix-7)
```

## Quick Start
```bash
pip install numpy torch scipy scikit-learn
python run_experiment.py --quick --arch mlp
python run_experiment.py --n_traces 1000 --epochs 10 --arch lstm
```

## Hybrid Countermeasure
Four layers applied at decapsulation inner product s^T.u:
  1. First-order additive masking: s = s0+s1 (mod Q)
  2. Instruction shuffling: randomised coefficient access order
  3. Dummy operations: 25% probability random insertions
  4. ASCON-128: intermediate su encrypted before use

## Key Implementation Notes
- NTT uses zeta[i] = 17^brv7(i) mod 3329 (verified against ref impl)
- Masking is ADDITIVE (not XOR) — required for Z_q arithmetic
- Leakage model: Hamming Weight + Gaussian noise
- Attack metric: key rank, guessing entropy, traces-to-disclosure (TTD)
