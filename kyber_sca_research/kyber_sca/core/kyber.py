"""
CRYSTALS-Kyber512 KEM — complete implementation
Algorithms from CRYSTALS-Kyber v3.02 specification.

Exposes intermediate values at Decapsulation for SCA.
"""

import os
from .params import N, Q, K, ETA1, ETA2, DU, DV
from .polynomial import Polynomial
from .sampling import G, H, KDF, PRF, gen_matrix, gen_noise_vec, sample_noise


# -------------------------------------------------------
# Module-level helpers (operate on lists of Polynomials)
# -------------------------------------------------------

def _ntt_vec(v):      return [p.ntt() for p in v]
def _inv_ntt_vec(v):  return [p.inv_ntt() for p in v]

def _mat_vec_ntt(A, v):
    """Matrix-vector multiply; A and v both in NTT domain."""
    k = len(v)
    result = []
    for i in range(k):
        acc = Polynomial()
        for j in range(k):
            acc = acc + A[i][j].pw_mul(v[j])
        result.append(acc)
    return result

def _dot_ntt(a, b):
    """Inner product of two NTT-domain vectors."""
    acc = Polynomial()
    for i in range(len(a)):
        acc = acc + a[i].pw_mul(b[i])
    return acc

def _vec_add(a, b):
    return [a[i] + b[i] for i in range(len(a))]

def _polyvec_to_bytes(v):
    return b''.join(p.to_bytes12() for p in v)

def _polyvec_from_bytes(data, k):
    return [Polynomial.from_bytes12(data[i*384:(i+1)*384]) for i in range(k)]


# -------------------------------------------------------
# Kyber-PKE (IND-CPA layer)
# -------------------------------------------------------

class KyberPKE:

    @staticmethod
    def keygen(seed: bytes = None):
        """Algorithm 4: Kyber.CPAPKE.KeyGen"""
        if seed is None:
            seed = os.urandom(32)
        rho, sigma = G(seed)

        A = gen_matrix(rho, K, transposed=False)   # A in NTT domain

        nonce = 0
        s_raw = gen_noise_vec(sigma, nonce,     ETA1, K); nonce += K
        e_raw = gen_noise_vec(sigma, nonce,     ETA1, K); nonce += K

        s_hat = _ntt_vec(s_raw)
        e_hat = _ntt_vec(e_raw)

        # t̂ = Â·ŝ + ê   (in NTT domain, then convert back for storage)
        As_hat = _mat_vec_ntt(A, s_hat)
        t_hat  = _vec_add(As_hat, e_hat)
        t_raw  = _inv_ntt_vec(t_hat)

        pk = _polyvec_to_bytes(t_raw) + rho
        sk = _polyvec_to_bytes(s_raw)

        internals = {
            'rho': rho, 'sigma': sigma,
            's': s_raw, 'e': e_raw, 't': t_raw,
        }
        return pk, sk, internals

    @staticmethod
    def encrypt(pk: bytes, m: bytes, coins: bytes) -> bytes:
        """Algorithm 5: Kyber.CPAPKE.Enc"""
        t_raw = _polyvec_from_bytes(pk[:K*384], K)
        rho   = pk[K*384: K*384+32]

        AT = gen_matrix(rho, K, transposed=True)

        nonce = 0
        r_raw  = gen_noise_vec(coins, nonce, ETA1, K); nonce += K
        e1_raw = gen_noise_vec(coins, nonce, ETA2, K); nonce += K
        e2_raw = sample_noise(coins, nonce, ETA2)

        r_hat  = _ntt_vec(r_raw)
        t_hat  = _ntt_vec(t_raw)

        # u = A^T·r + e1
        ATr    = _mat_vec_ntt(AT, r_hat)
        u      = _vec_add(_inv_ntt_vec(ATr), e1_raw)

        # v = t^T·r + e2 + Decompress_q(m,1)
        tr     = _dot_ntt(t_hat, r_hat).inv_ntt()
        m_poly = Polynomial.decode_msg(m)
        v      = tr + e2_raw + m_poly

        # Compress and serialize
        ct = b''
        for poly in u:
            ct += poly.compress_to_bytes(DU)
        ct += v.compress_to_bytes(DV)
        return ct

    @staticmethod
    def decrypt(sk: bytes, ct: bytes):
        """
        Algorithm 6: Kyber.CPAPKE.Dec
        PRIMARY TARGET for side-channel attacks.
        Returns (m_bytes, intermediates_dict).
        """
        # Parse ciphertext
        u_bytes_each = N * DU // 8   # 320 bytes per poly
        v_bytes_each = N * DV // 8   # 128 bytes

        u = [Polynomial.decompress_from_bytes(
                ct[i*u_bytes_each:(i+1)*u_bytes_each], DU)
             for i in range(K)]
        v = Polynomial.decompress_from_bytes(
                ct[K*u_bytes_each: K*u_bytes_each + v_bytes_each], DV)

        # Parse secret key
        s  = _polyvec_from_bytes(sk[:K*384], K)
        s_hat = _ntt_vec(s)
        u_hat = _ntt_vec(u)

        # w = v - s^T·u   (the sensitive computation)
        su = _dot_ntt(s_hat, u_hat).inv_ntt()
        w  = v - su

        m = w.encode_msg()

        intermediates = {
            'u': u, 'v': v, 's': s,
            'su': su, 'w': w,
        }
        return m, intermediates


# -------------------------------------------------------
# Kyber-KEM (IND-CCA2 via Fujisaki-Okamoto transform)
# -------------------------------------------------------

class KyberKEM:

    @staticmethod
    def keygen():
        """Full KEM key generation."""
        z  = os.urandom(32)
        pk, sk_pke, internals = KyberPKE.keygen()
        hpk = H(pk)
        # sk = sk_pke ∥ pk ∥ H(pk) ∥ z
        sk = sk_pke + pk + hpk + z
        return pk, sk, internals

    @staticmethod
    def encaps(pk: bytes):
        """Encapsulation — generates (ciphertext, shared_secret)."""
        m   = os.urandom(32)
        hpk = H(pk)
        K_bar, r = G(m + hpk)
        ct  = KyberPKE.encrypt(pk, m, r)
        ss  = KDF(K_bar + H(ct))
        return ct, ss

    @staticmethod
    def decaps(sk: bytes, ct: bytes):
        """
        Decapsulation — main target for side-channel analysis.
        Implements implicit rejection (Targhi-Unruh FO transform).

        Returns: (shared_secret: bytes, intermediates: dict)

        The intermediates dict exposes all sensitive values for
        side-channel leakage simulation.
        """
        sk_pke = sk[:K*384]
        pk     = sk[K*384 : K*384 + 800]
        hpk    = sk[K*384+800 : K*384+832]
        z      = sk[K*384+832 : K*384+864]

        # ---- Step 1: Decrypt (most leakage here) ----------------
        m_prime, dec_int = KyberPKE.decrypt(sk_pke, ct)

        # ---- Step 2: Re-encrypt (FO check) ----------------------
        K_bar_prime, r_prime = G(m_prime + hpk)
        ct_prime = KyberPKE.encrypt(pk, m_prime, r_prime)

        # ---- Step 3: Implicit rejection (constant-time in HW) ---
        ct_match = (ct == ct_prime)
        if ct_match:
            ss = KDF(K_bar_prime + H(ct))
        else:
            ss = KDF(z + H(ct))

        intermediates = {
            'm_prime':      m_prime,
            'K_bar_prime':  K_bar_prime,
            'ct_match':     ct_match,
            'dec':          dec_int,
            # Key SCA targets:
            'w':            dec_int['w'],   # message decoding polynomial
            'su':           dec_int['su'],  # s^T·u (leaks secret s)
            's':            dec_int['s'],   # secret key polynomials
            'u':            dec_int['u'],   # ciphertext u
        }
        return ss, intermediates
