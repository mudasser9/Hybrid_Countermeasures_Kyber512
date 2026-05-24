"""
Hybrid Countermeasure for Kyber512 Decapsulation Phase
=======================================================
Implements and combines:
  1. First-order ADDITIVE masking (correct for Z_q arithmetic)
  2. Instruction shuffling (randomised coefficient order)
  3. Dummy operations (random noise insertions)
  4. ASCON-style intermediate value protection

Additive masking: s = s0 + s1 (mod Q)  →  s^T·u = s0^T·u + s1^T·u (mod Q)
This is the correct masking scheme for Kyber's modular-arithmetic operations,
unlike XOR masking which only works for binary/GF(2) circuits.
"""

import os
import numpy as np
from core.polynomial import Polynomial
from core.params import N, Q, K, DU, DV
from core.sampling import G, H, KDF
from core.kyber import KyberPKE, KyberKEM, _polyvec_from_bytes, _ntt_vec, \
                       _dot_ntt, _inv_ntt_vec


# -------------------------------------------------------
# Simulated ASCON-128 permutation
# -------------------------------------------------------

class AsconSim:
    """
    Simulated ASCON-128 for protecting intermediate values.
    In a real hardware implementation this is a Verilog module (see verilog/).
    In simulation we model its effect on the trace profile.
    """
    RC = [0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b]

    def __init__(self, key: bytes = None):
        self.key = key or os.urandom(16)
        # Derive key words
        self._kw = [int.from_bytes(self.key[i:i+2], 'little') for i in range(0, 16, 2)]

    def _permute(self, state: list, rounds: int = 6) -> list:
        s = state[:]
        for r in range(rounds):
            s[2] ^= self.RC[r]
            # Nonlinear layer (simplified from spec)
            for i in range(len(s)):
                s[i] = (s[i] ^ (s[(i+1) % len(s)]) ^ (s[(i+2) % len(s)] & s[(i+3) % len(s)])) & 0xFFFF
            # Linear diffusion
            s = [((s[i] >> 3) ^ (s[(i+1) % len(s)] << 1)) & 0xFFFF for i in range(len(s))]
        return s

    def encrypt_coeff(self, val: int) -> int:
        """
        Encrypt a 12-bit polynomial coefficient.
        Result is used for leakage modelling; correctness maintained via decrypt.
        """
        state = [val & 0xFFF, self._kw[0], self._kw[1], 0xA5A5, 0x3C3C]
        enc   = self._permute(state, rounds=6)
        return enc[0] & 0xFFF

    def decrypt_coeff(self, enc_val: int, original: int) -> int:
        """Transparent in simulation — returns the original value."""
        return original


# -------------------------------------------------------
# Additive Masking (correct for Z_q)
# -------------------------------------------------------

class AdditiveMasking:
    """
    First-order additive Boolean masking over Z_Q.
    s = s0 + s1 (mod Q)  where s0, s1 are random shares.

    Property: s^T·u = s0^T·u + s1^T·u (mod Q)
    This preserves the linearity of the NTT inner product.
    """

    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()

    def mask_poly(self, poly: Polynomial) -> tuple:
        """
        Split poly into two additive shares.
        Returns (share0, share1) such that share0 + share1 ≡ poly (mod Q).
        """
        r  = Polynomial([int(self.rng.integers(0, Q)) for _ in range(N)])
        s0 = Polynomial([(poly._c[i] - r._c[i]) % Q for i in range(N)])
        s1 = r
        return s0, s1

    def recombine(self, s0: Polynomial, s1: Polynomial) -> Polynomial:
        return Polynomial([(s0._c[i] + s1._c[i]) % Q for i in range(N)])


# -------------------------------------------------------
# Instruction Shuffler
# -------------------------------------------------------

class InstructionShuffler:
    """Randomises polynomial processing order (independent of correctness)."""

    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()

    def shuffled_order(self, n: int) -> list:
        return self.rng.permutation(n).tolist()

    def coeff_access_order(self, n: int = N) -> list:
        return self.rng.permutation(n).tolist()


# -------------------------------------------------------
# Dummy Operation Generator
# -------------------------------------------------------

class DummyOps:
    """Inserts discarded random computations to obscure trace patterns."""

    def __init__(self, prob: float = 0.25, rng=None):
        self.prob = prob
        self.rng  = rng or np.random.default_rng()

    def maybe_exec(self) -> bool:
        return self.rng.random() < self.prob

    def dummy_add(self) -> int:
        return (int(self.rng.integers(0, Q)) + int(self.rng.integers(0, Q))) % Q

    def dummy_mul(self) -> int:
        return (int(self.rng.integers(0, Q)) * int(self.rng.integers(0, Q))) % Q


# -------------------------------------------------------
# Hybrid Decapsulation Engine
# -------------------------------------------------------

class HybridDecapsulation:
    """
    Protected Kyber512 PKE decryption with all four countermeasure layers.

    Correctness: verified against standard KyberPKE.decrypt().
    """

    def __init__(self, ascon_key: bytes = None, seed: int = None):
        rng = np.random.default_rng(seed)
        self.masking  = AdditiveMasking(rng=rng)
        self.shuffler = InstructionShuffler(rng=rng)
        self.dummy    = DummyOps(prob=0.25, rng=rng)
        self.ascon    = AsconSim(key=ascon_key)

    def decrypt(self, sk: bytes, ct: bytes):
        """
        Hybrid-protected decryption.  Same output as KyberPKE.decrypt().
        Returns: (m: bytes, intermediates: dict)
        """
        # --- Parse ciphertext ---
        u_sz = N * DU // 8
        v_sz = N * DV // 8
        u = [Polynomial.decompress_from_bytes(ct[i*u_sz:(i+1)*u_sz], DU)
             for i in range(K)]
        v = Polynomial.decompress_from_bytes(ct[K*u_sz : K*u_sz+v_sz], DV)

        # --- Parse secret key ---
        s = _polyvec_from_bytes(sk[:K*384], K)

        # ====================================================
        # LAYER 1: Additive masking of secret key s
        # ====================================================
        shares = [self.masking.mask_poly(poly) for poly in s]
        # shares[i] = (s0_i, s1_i) with s0_i + s1_i = s_i (mod Q)

        # ====================================================
        # LAYER 2: Shuffled NTT computation of u
        # LAYER 3: Dummy ops during accumulation
        # ====================================================
        u_hat = _ntt_vec(u)

        # Process K polynomial pairs in random order
        poly_order = self.shuffler.shuffled_order(K)

        acc0 = Polynomial()  # accumulates share0 inner product
        acc1 = Polynomial()  # accumulates share1 inner product
        _sink = 0            # absorb dummy results (prevents dead-code elimination)

        for idx in poly_order:
            s0_i, s1_i = shares[idx]
            u_i        = u_hat[idx]

            # Dummy op before (Layer 3)
            if self.dummy.maybe_exec():
                _sink += self.dummy.dummy_add()

            # NTT of each share independently (critical: no unmasked intermediate)
            s0_hat = s0_i.ntt()
            s1_hat = s1_i.ntt()

            # Point-wise multiply each share with u
            acc0 = acc0 + s0_hat.pw_mul(u_i)
            acc1 = acc1 + s1_hat.pw_mul(u_i)

            # Dummy op after
            if self.dummy.maybe_exec():
                _sink += self.dummy.dummy_mul()

        # InvNTT of each share
        su0 = acc0.inv_ntt()
        su1 = acc1.inv_ntt()

        # ====================================================
        # LAYER 4: ASCON-protect the intermediate su before recombination
        # ====================================================
        # Encrypt each coefficient of su0 (the dominant share)
        su0_protected = Polynomial([self.ascon.encrypt_coeff(su0._c[i]) for i in range(N)])
        # Decrypt transparently for computation
        su0_real = Polynomial([self.ascon.decrypt_coeff(su0_protected._c[i], su0._c[i]) for i in range(N)])

        # Recombine shares: su = su0 + su1 (mod Q)
        su = self.masking.recombine(su0_real, su1)

        # Final decode: w = v - su
        w = v - su
        m = w.encode_msg()

        intermediates = {
            'u': u, 'v': v, 's': s, 'su': su, 'w': w,
            'su_share0': su0_protected,  # protected version (for leakage model)
            'su_share1': su1,
            's_shares':  shares,
        }
        return m, intermediates


# -------------------------------------------------------
# Hybrid KEM
# -------------------------------------------------------

class HybridKyberKEM:
    """
    Full Kyber512 KEM with hybrid-protected decapsulation.
    Functionally identical to KyberKEM; side-channel hardened.
    """

    def __init__(self, ascon_key: bytes = None, seed: int = None):
        self._engine = HybridDecapsulation(ascon_key=ascon_key, seed=seed)

    def keygen(self):
        return KyberKEM.keygen()

    def encaps(self, pk: bytes):
        return KyberKEM.encaps(pk)

    def decaps(self, sk: bytes, ct: bytes):
        sk_pke = sk[:K*384]
        pk     = sk[K*384 : K*384 + 800]
        hpk    = sk[K*384+800 : K*384+832]
        z      = sk[K*384+832 : K*384+864]

        # Protected decryption
        m_prime, dec_int = self._engine.decrypt(sk_pke, ct)

        # FO re-encrypt + compare
        K_bar_prime, r_prime = G(m_prime + hpk)
        ct_prime = KyberPKE.encrypt(pk, m_prime, r_prime)

        ct_match = (ct == ct_prime)
        ss = KDF(K_bar_prime + H(ct)) if ct_match else KDF(z + H(ct))

        return ss, {**dec_int, 'ct_match': ct_match}
