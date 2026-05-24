"""
Kyber512 Parameters — CRYSTALS-Kyber NIST PQC Standard
Reference: CRYSTALS-Kyber Algorithm Specification v3.0
"""

# Core parameters for Kyber512
N   = 256          # Polynomial degree
Q   = 3329         # Modulus (prime)
K   = 2            # Module rank (k=2 for Kyber512)
ETA1 = 3           # Noise distribution parameter (keygen)
ETA2 = 2           # Noise distribution parameter (enc)
DU   = 10          # Compression bits for u
DV   = 4           # Compression bits for v

# NTT parameters (for Z_q[X]/(X^N + 1))
ZETA = 17          # Primitive 256th root of unity mod Q
GAMMA = 17         # Generator used in NTT layer

# Derived
N_BYTES       = N * 12 // 8     # Bytes for one polynomial
POLY_BYTES    = 384
POLY_VEC_BYTES = K * POLY_BYTES
SEED_BYTES    = 32
SYM_BYTES     = 32

# Shared secret and key sizes
SS_BYTES      = 32
SK_BYTES      = 1632   # Secret key size in bytes
PK_BYTES      = 800    # Public key size
CT_BYTES      = 768    # Ciphertext size

# -------------------------------------------------------
# Montgomery / Barrett reduction constants (for hardware)
# -------------------------------------------------------
MONT_R   = 1 << 16
MONT_R2  = (MONT_R * MONT_R) % Q
QINV     = 62209   # Q^{-1} mod 2^16  (used in Montgomery)
MONT_INV = pow(MONT_R, Q - 2, Q)   # R^{-1} mod Q

def barrett_reduce(a: int) -> int:
    """Barrett reduction: computes a mod Q efficiently."""
    v = ((1 << 26) + Q // 2) // Q
    t = v * a >> 26
    return a - t * Q

def montgomery_reduce(a: int) -> int:
    """Montgomery reduction: computes aR^{-1} mod Q."""
    u = (a * QINV) & 0xFFFF
    t = (a - u * Q) >> 16
    if t < 0:
        t += Q
    return t

def freeze(a: int) -> int:
    """Fully reduce a mod Q into [0, Q-1]."""
    r = barrett_reduce(a)
    r -= Q
    r += (r >> 31) & Q
    return r
