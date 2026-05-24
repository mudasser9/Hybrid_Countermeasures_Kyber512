"""
Sampling routines for Kyber512:
  - Hash functions (G, H, KDF, PRF using Python hashlib)
  - Uniform polynomial sampling (XOF / rejection sampling)
  - Centered Binomial Distribution noise
"""

import hashlib
import numpy as np
from .params import N, Q, ETA1, ETA2
from .polynomial import Polynomial


# -------------------------------------------------------
# Hash functions
# -------------------------------------------------------

def sha3_256(data: bytes) -> bytes:
    return hashlib.sha3_256(data).digest()

def sha3_512(data: bytes) -> bytes:
    return hashlib.sha3_512(data).digest()

def shake_128(data: bytes, length: int) -> bytes:
    h = hashlib.shake_128(); h.update(data)
    return h.digest(length)

def shake_256(data: bytes, length: int) -> bytes:
    h = hashlib.shake_256(); h.update(data)
    return h.digest(length)

def G(seed: bytes):
    """G: B* → B^32 × B^32 (SHA3-512, split in half)."""
    h = sha3_512(seed)
    return h[:32], h[32:]

def H(data: bytes) -> bytes:
    return sha3_256(data)

def KDF(data: bytes) -> bytes:
    return shake_256(data, 32)

def PRF(seed: bytes, nonce: int, length: int) -> bytes:
    """PRF(seed, nonce) = SHAKE-256(seed ∥ nonce)."""
    return shake_256(seed + bytes([nonce]), length)


# -------------------------------------------------------
# Uniform polynomial sampling (Algorithm 1 in spec)
# -------------------------------------------------------

def sample_uniform(rho: bytes, i: int, j: int) -> Polynomial:
    """
    Parse a uniform polynomial from SHAKE-128(rho ∥ j ∥ i).
    Note: spec uses (j, i) order for A[i][j].
    """
    xof_seed = rho + bytes([j, i])
    stream = shake_128(xof_seed, 3 * N)   # generous initial pull
    coeffs = []
    pos = 0
    while len(coeffs) < N:
        if pos + 3 > len(stream):
            stream += shake_128(xof_seed, 3 * N)
        d1 = stream[pos] | ((stream[pos+1] & 0xF) << 8)
        d2 = (stream[pos+1] >> 4) | (stream[pos+2] << 4)
        if d1 < Q:
            coeffs.append(d1)
        if d2 < Q and len(coeffs) < N:
            coeffs.append(d2)
        pos += 3
    return Polynomial(coeffs)


# -------------------------------------------------------
# CBD noise sampling
# -------------------------------------------------------

def cbd(eta: int, buf: bytes) -> Polynomial:
    """
    Centered Binomial Distribution CBD_η.
    Needs exactly N * 2 * eta bits = N * eta // 4 bytes (for N=256).
    """
    needed = N * eta // 4
    assert len(buf) >= needed, f"CBD needs {needed} bytes, got {len(buf)}"
    coeffs = []
    for i in range(N):
        a, b = 0, 0
        for j in range(eta):
            bit_idx_a = 2 * i * eta + j
            bit_idx_b = 2 * i * eta + eta + j
            a += (buf[bit_idx_a >> 3] >> (bit_idx_a & 7)) & 1
            b += (buf[bit_idx_b >> 3] >> (bit_idx_b & 7)) & 1
        coeffs.append((a - b) % Q)
    return Polynomial(coeffs)


def sample_noise(seed: bytes, nonce: int, eta: int) -> Polynomial:
    needed = N * eta // 4
    return cbd(eta, PRF(seed, nonce, needed))


# -------------------------------------------------------
# Generate matrix A and noise vectors
# -------------------------------------------------------

def gen_matrix(rho: bytes, k: int, transposed: bool = False) -> list:
    """
    Generate k×k matrix of NTT-domain polynomials.
    A[i][j] = sample_uniform(rho, i, j)  (or transposed)
    """
    A = []
    for i in range(k):
        row = []
        for j in range(k):
            if transposed:
                poly = sample_uniform(rho, j, i)
            else:
                poly = sample_uniform(rho, i, j)
            row.append(poly.ntt())
        A.append(row)
    return A


def gen_noise_vec(sigma: bytes, nonce_start: int, eta: int, k: int) -> list:
    result = []
    for i in range(k):
        result.append(sample_noise(sigma, nonce_start + i, eta))
    return result
