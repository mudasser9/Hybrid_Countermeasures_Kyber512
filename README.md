# Hybrid Side-Channel Countermeasure for CRYSTALS-Kyber (ML-KEM-512) Decapsulation

Code accompanying the paper **"A Side-Channel-Resistant Hybrid Framework for CRYSTALS-Kyber"**
(M. H. Wasti, S. Ullah, S. Khan, T. Ahmad, A. Buriro, A. Bisegna), submitted to *Scientific Reports*.

## What the paper evaluates

The countermeasure protects the inner product **s**<sup>T</sup>**u** of Kyber-512 decapsulation with three layers:

1. **First-order arithmetic masking**: each secret coefficient is split into two shares, `s = s0 + s1 (mod q)`, with fresh masks in every execution.
2. **Shuffling**: a fresh Fisher–Yates permutation of all 512 coefficient operations in every execution.
3. **Dummy operations**: a masked multiplication on random operands is inserted before each real operation with probability 0.25.

The randomness for the shuffling and the dummy operations is drawn from **ASCON-XOF128** (NIST SP 800-232). ASCON is used as a **randomness source only**; it does not encrypt intermediate values. ASCON-128 authenticated encryption of sensor data belongs to the wider V2X architecture described in the paper and is not evaluated.

**All leakage and attack results are obtained from simulated Hamming-weight power traces, not from measurements on an FPGA.**

## Repository layout

| Path | Content |
|---|---|
| `kyber_sca_fixed/` | **Evaluation used in the paper**: trace simulator, TVLA, profiled attacks, second-order and aggregate attacks, report generation. Outputs are in `kyber_sca_fixed/results/`. |
| `algorithm1.py` | Python reference implementation of Algorithm 1 (HybridDecaps); see the status notes below. |
| `kyber_sca_research/kyber_sca/` | Kyber-512 core (KEM, NTT), MLP/LSTM/Transformer models and the training engine. The evaluation scripts import the models and trainer from here. `kyber_sca_research.tar.gz` is an archive of the same folder. |
| `kyber512_decap_hybrid.v`, `kyber_imp_viv/`, `*_report.txt`, `vivado*.log` / `vivado*.jou` | FPGA design (Xilinx Artix-7, Vivado 2019.1), utilisation, timing and power reports, and Vivado logs. |
| `scripts/` | `theory.py` (class probabilities and information limits of the leakage model), `wrap_tables.py` (table formatting). |
| `hybrid_countermeasure.py`, `run_experiment.py`, `kyber.py`, `polynomial.py`, `tv_test.py` | **Earlier prototype, not used for the results in the paper.** Its ASCON layer encrypts the intermediate value **s**<sup>T</sup>**u**, which is not the design evaluated in the paper. |

## Requirements

Python 3.13 with NumPy 2.5.3, SciPy 1.18, scikit-learn 1.9.1, PyTorch 2.14 (CPU) and Matplotlib.

```bash
pip install numpy scipy scikit-learn torch matplotlib
```

## Reproducing the results

Every random draw is seeded, so each script regenerates the same numbers.

```bash
git clone https://github.com/mudasser9/Hybrid_Countermeasures_Kyber512
cd Hybrid_Countermeasures_Kyber512
export KYBER_SCA_REPO="$PWD/kyber_sca_research/kyber_sca"   # location of the models and trainer
cd kyber_sca_fixed

python run_tvla.py               # first- and second-order TVLA (Table 1, Fig. 2)
python run_null_calibration.py   # fixed-vs-fixed null control (TVLA false-alarm rate)
python run_attacks.py            # template, MLP, CNN and Transformer attacks (add --quick for a short test run)
python rescore.py                # calibrated MAP scoring of the saved attack outputs (Table 2, Fig. 3)
python run_second_order.py       # second-order template attacks (Table 3, Fig. 4)
python run_share_diag_lda.py     # mask-aware share diagnostic (Supplementary Table S3)
(cd results && python ../run_aggregate_attack.py 0.1 && python ../run_aggregate_attack.py 0.5)
                                 # aggregate attacker without location (Table 3, rows marked with a dagger)
python make_report.py            # LaTeX tables and figures in results/
python make_report_so.py         # second-order table and figure
```

The `KYBER_SCA_REPO` variable is required: the default path in `attack.py` and `algorithm1.py` points to the authors' machine.

## Leakage model (summary)

- A trace has T = 1536 samples. A processed value v contributes HW(v)/12 plus Gaussian noise N(0, σ²), with σ ∈ {0.1, 0.5}.
- **C0** (unmasked): HW(s<sub>i</sub>) of each of the 512 secret coefficients.
- **C1** (masked): HW(s<sub>0,i</sub>) and HW(s<sub>1,i</sub>) at adjacent samples.
- **C2** (hybrid): the C1 share pairs in a random order, a dummy pair before each real pair with probability 0.25, and 64 secret-independent ASCON samples.
- Attack target: HW(s[0][0]), which takes six values for Kyber-512 (η₁ = 3).

## Reference implementation of Algorithm 1

```bash
export KYBER_SCA_REPO="$PWD/kyber_sca_research/kyber_sca"
python algorithm1.py 5    # five decapsulations, compared with the reference KEM
```

| Status | Steps |
|---|---|
| Implemented | Masking, both Fisher–Yates shuffles, dummy operations, share-wise products in the NTT domain, share-wise inverse NTT, shuffled decoding loop |
| Placeholder (not side-channel protected) | `MaskedShiftRegDecode`, `MaskedG`, `MaskedEncrypt`, `MaskedConstTimeCompare`: functionally correct, but they recombine the shares internally |
| Stand-in | The randomness comes from SHAKE-128 instead of ASCON-XOF128 |
| Note | The products use the element-wise multiplication of the repository's NTT core, not the FIPS 203 base multiplication on coefficient pairs |

`algorithm1.py` computes the algorithm; it does not generate leakage. All leakage results come from `kyber_sca_fixed/sim.py`.

## Hardware

The FPGA design targets a Xilinx Artix-7 (xc7a100tcsg324-1) and was synthesised with Vivado 2019.1. It implements the NTT unit with three Barrett multipliers and the NTT-domain arithmetic of the inner product. It does **not** implement `MaskedG`, `MaskedEncrypt` or the comparison step. Post-implementation utilisation: 8,465 LUTs, 3,363 flip-flops, no Block RAM, nine DSP48E1 slices. No unprotected baseline was synthesised with the same flow, so no hardware overhead is claimed.

## Citation

If you use this code, please cite the paper (citation details will be added after publication).
