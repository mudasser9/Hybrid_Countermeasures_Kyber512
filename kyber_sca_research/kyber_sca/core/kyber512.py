"""
kyber512.py
===========
Pure-Python reference implementation of Kyber512 (CRYSTALS-Kyber).
Covers:
  - Number Theoretic Transform (NTT) and inverse NTT
  - Polynomial arithmetic mod q=3329
  - Key generation, encapsulation, decapsulation
  - First-order boolean masking of decapsulation (base paper model)

All arithmetic is done in plain Python integers so leakage traces
can be captured at the instruction level for the SCA simulation.
"""

import os
import struct
import hashlib
import hmac
from typing import Tuple, List, Optional

# ── Kyber512 parameters ───────────────────────────────────────────────────────
Q    = 3329          # prime modulus
N    = 256           # polynomial degree
K    = 2             # k=2 for Kyber512
ETA1 = 3             # noise parameter η1
ETA2 = 2             # noise parameter η2
DU   = 10            # compression parameter du
DV   = 4             # compression parameter dv

# Pre-computed NTT root of unity (ζ = 17, a primitive 256th root of 1 mod 3329)
ZETA = 17

# Bit-reversal permutation for NTT
def _brv(x: int, bits: int = 8) -> int:
    result = 0
    for _ in range(bits):
        result = (result << 1) | (x & 1)
        x >>= 1
    return result

# Pre-compute zeta powers in bit-reversal order (need 192 entries: 128 for NTT + 64 for basemul)
ZETAS = [pow(ZETA, _brv(i, 7), Q) for i in range(128)]
# Extend with additional zeta powers for basemul (zeta^(2*br(i)+1))
ZETAS_BASEMUL = [pow(ZETA, 2*_brv(i, 7) + 1, Q) for i in range(64)]

# ── Modular arithmetic helpers ────────────────────────────────────────────────

def barrett_reduce(a: int) -> int:
    """Barrett reduction: reduce a mod Q to (-Q/2, Q/2]."""
    v = ((1 << 26) + Q // 2) // Q
    t = v * a >> 26
    a -= t * Q
    return a

def mont_reduce(a: int) -> int:
    """Montgomery reduction (simplified, mod Q)."""
    return a % Q

def to_mont(a: int) -> int:
    return (a * 2285) % Q  # 2285 = 2^16 mod Q (Montgomery constant)

def from_mont(a: int) -> int:
    return (a * pow(2285, -1, Q)) % Q

# ── NTT Implementation ────────────────────────────────────────────────────────

def ntt(f: List[int]) -> List[int]:
    """
    Number Theoretic Transform of polynomial f in Z_Q[X]/(X^256+1).
    Returns NTT representation.  Cooley-Tukey butterfly.
    """
    f = list(f)
    k = 1
    length = 128
    while length >= 2:
        start = 0
        while start < 256:
            zeta = ZETAS[k]
            k += 1
            for j in range(start, start + length):
                t = (zeta * f[j + length]) % Q
                f[j + length] = (f[j] - t) % Q
                f[j]          = (f[j] + t) % Q
            start += 2 * length
        length //= 2
    return f

def intt(f: List[int]) -> List[int]:
    """Inverse NTT. Gentleman-Sande butterfly."""
    f = list(f)
    k = 127
    length = 2
    while length <= 128:
        start = 0
        while start < 256:
            zeta = ZETAS[k]
            k -= 1
            for j in range(start, start + length):
                t       = f[j]
                f[j]    = (t + f[j + length]) % Q
                f[j + length] = (zeta * (f[j + length] - t)) % Q
            start += 2 * length
        length *= 2
    # Multiply by N^{-1} mod Q = 3303
    inv_n = pow(256, -1, Q)
    f = [(coeff * inv_n) % Q for coeff in f]
    return f

def poly_basemul(a: List[int], b: List[int]) -> List[int]:
    """
    Pointwise multiplication of two NTT-domain polynomials.
    Uses the degree-2 base multiplication for the incomplete NTT.
    """
    result = [0] * 256
    for i in range(64):
        # Schoolbook over degree-1 field elements paired with zeta
        a0, a1 = a[2*i],   a[2*i+1]
        b0, b1 = b[2*i],   b[2*i+1]
        zeta   = ZETAS_BASEMUL[i]
        result[2*i]   = (a0*b0 + zeta*a1*b1) % Q
        result[2*i+1] = (a0*b1 + a1*b0)      % Q
    # Second half with negated zeta (conjugate)
    for i in range(64, 128):
        a0, a1 = a[2*i],   a[2*i+1]
        b0, b1 = b[2*i],   b[2*i+1]
        zeta   = (Q - ZETAS_BASEMUL[i - 64]) % Q  # negated zeta
        result[2*i]   = (a0*b0 + zeta*a1*b1) % Q
        result[2*i+1] = (a0*b1 + a1*b0)      % Q
    return result

def poly_add(a: List[int], b: List[int]) -> List[int]:
    return [(x + y) % Q for x, y in zip(a, b)]

def poly_sub(a: List[int], b: List[int]) -> List[int]:
    return [(x - y) % Q for x, y in zip(a, b)]

def polyvec_add(A, B):
    return [poly_add(a, b) for a, b in zip(A, B)]

def polyvec_sub(A, B):
    return [poly_sub(a, b) for a, b in zip(A, B)]

# ── Encoding / Decoding ───────────────────────────────────────────────────────

def compress(x: int, d: int) -> int:
    """Compress an integer modulo Q to d bits."""
    return round(x * (1 << d) / Q) % (1 << d)

def decompress(x: int, d: int) -> int:
    """Decompress d-bit integer back to Z_Q."""
    return round(x * Q / (1 << d)) % Q

def poly_compress(p: List[int], d: int) -> bytes:
    coeffs = [compress(c, d) for c in p]
    # Pack coefficients into bytes
    out = bytearray()
    if d == 4:
        for i in range(0, 256, 2):
            out.append(coeffs[i] | (coeffs[i+1] << 4))
    elif d == 10:
        for i in range(0, 256, 4):
            val = coeffs[i] | (coeffs[i+1] << 10) | (coeffs[i+2] << 20) | (coeffs[i+3] << 30)
            out += val.to_bytes(5, 'little')
    elif d == 11:
        for i in range(0, 256, 8):
            val = sum(coeffs[i+j] << (11*j) for j in range(8))
            out += val.to_bytes(11, 'little')
    return bytes(out)

def poly_decompress(b: bytes, d: int) -> List[int]:
    coeffs = []
    if d == 4:
        for byte in b:
            coeffs.append(decompress(byte & 0xF, d))
            coeffs.append(decompress(byte >> 4, d))
    elif d == 10:
        for i in range(0, len(b), 5):
            val = int.from_bytes(b[i:i+5], 'little')
            for j in range(4):
                coeffs.append(decompress((val >> (10*j)) & 0x3FF, d))
    elif d == 11:
        for i in range(0, len(b), 11):
            val = int.from_bytes(b[i:i+11], 'little')
            for j in range(8):
                coeffs.append(decompress((val >> (11*j)) & 0x7FF, d))
    return coeffs[:256]

def encode_poly(p: List[int], d: int) -> bytes:
    """Encode polynomial to byte array (d bits per coeff)."""
    out = bytearray()
    if d == 12:
        for i in range(0, 256, 2):
            v = (p[i] & 0xFFF) | ((p[i+1] & 0xFFF) << 12)
            out += v.to_bytes(3, 'little')
    else:
        # Generic: pack d bits per coefficient
        bits = 0
        accum = 0
        for c in p:
            accum |= (c & ((1 << d) - 1)) << bits
            bits += d
            while bits >= 8:
                out.append(accum & 0xFF)
                accum >>= 8
                bits -= 8
        if bits > 0:
            out.append(accum & 0xFF)
    return bytes(out)

def decode_poly(b: bytes, d: int) -> List[int]:
    """Decode byte array to polynomial."""
    coeffs = []
    accum = 0
    bits = 0
    mask = (1 << d) - 1
    for byte in b:
        accum |= byte << bits
        bits += 8
        while bits >= d:
            coeffs.append(accum & mask)
            accum >>= d
            bits -= d
    return coeffs[:256]

# ── Sampling ──────────────────────────────────────────────────────────────────

def cbd(b: bytes, eta: int) -> List[int]:
    """
    Centered binomial distribution sampling.
    Sample polynomial from CBD_eta using byte array b.
    """
    assert len(b) == 64 * eta
    f = [0] * 256
    for i in range(256):
        byte_idx = (i * eta) // 8
        bit_pos  = (i * eta) % 8
        a, bb = 0, 0
        for j in range(eta):
            a  += (b[byte_idx + (bit_pos + j    ) // 8] >> ((bit_pos + j    ) % 8)) & 1
            bb += (b[byte_idx + (bit_pos + j + eta) // 8] >> ((bit_pos + j + eta) % 8)) & 1
        f[i] = (a - bb) % Q
    return f

def sample_ntt(seed: bytes, i: int, j: int) -> List[int]:
    """
    Sample matrix element A[i][j] in NTT domain via rejection sampling.
    Uses SHAKE-128 XOF.
    """
    import hashlib
    xof = hashlib.shake_128(seed + bytes([i, j]))
    coeffs = []
    buf = xof.digest(504)   # sample extra bytes
    pos = 0
    while len(coeffs) < 256:
        if pos + 2 >= len(buf):
            buf = xof.digest(len(buf) + 168)
        d1 = buf[pos] | ((buf[pos+1] & 0x0F) << 8)
        d2 = (buf[pos+1] >> 4) | (buf[pos+2] << 4)
        if d1 < Q:
            coeffs.append(d1)
        if d2 < Q and len(coeffs) < 256:
            coeffs.append(d2)
        pos += 3
    return coeffs[:256]

def gen_matrix(rho: bytes, transposed: bool = False) -> List[List[List[int]]]:
    """Generate matrix A (or A^T) in NTT domain."""
    A = []
    for i in range(K):
        row = []
        for j in range(K):
            if transposed:
                row.append(sample_ntt(rho, j, i))
            else:
                row.append(sample_ntt(rho, i, j))
        A.append(row)
    return A

# ── Hash / PRF helpers ────────────────────────────────────────────────────────

def H(b: bytes) -> bytes:
    return hashlib.sha3_256(b).digest()

def G(b: bytes) -> Tuple[bytes, bytes]:
    h = hashlib.sha3_512(b).digest()
    return h[:32], h[32:]

def KDF(b: bytes) -> bytes:
    return hashlib.shake_256(b).digest(32)

def PRF(s: bytes, b: int, length: int) -> bytes:
    return hashlib.shake_256(s + bytes([b])).digest(length)

# ── Key Generation ────────────────────────────────────────────────────────────

def keygen() -> Tuple[bytes, bytes]:
    """
    Kyber512 key generation.
    Returns (public_key, secret_key) as bytes.
    """
    d = os.urandom(32)
    rho, sigma = G(d)

    # Sample matrix A
    A = gen_matrix(rho, transposed=False)

    # Sample secret s and error e
    s_vec, e_vec = [], []
    for i in range(K):
        prf_out = PRF(sigma, i,       64 * ETA1)
        s_vec.append(ntt(cbd(prf_out, ETA1)))
    for i in range(K):
        prf_out = PRF(sigma, K + i,   64 * ETA1)
        e_vec.append(ntt(cbd(prf_out, ETA1)))

    # t = A*s + e
    t_vec = []
    for i in range(K):
        acc = [0] * 256
        for j in range(K):
            acc = poly_add(acc, poly_basemul(A[i][j], s_vec[j]))
        t_vec.append(poly_add(acc, e_vec[i]))

    # Compress t for public key (12 bits per coefficient)
    pk_bytes = b''.join(encode_poly(t, 12) for t in t_vec) + rho
    sk_bytes = b''.join(encode_poly(s, 12) for s in s_vec)

    return pk_bytes, sk_bytes + pk_bytes + H(pk_bytes) + os.urandom(32)

# ── Encapsulation ─────────────────────────────────────────────────────────────

def encaps(pk: bytes) -> Tuple[bytes, bytes]:
    """
    Kyber512 encapsulation.
    Returns (ciphertext, shared_secret).
    """
    m = os.urandom(32)
    m_hash = H(m)

    # Decode public key
    t_vec = [decode_poly(pk[i*384:(i+1)*384], 12) for i in range(K)]
    rho   = pk[K*384:K*384+32]

    A_T = gen_matrix(rho, transposed=True)

    K_bar, r = G(m_hash + H(pk))

    # Sample r, e1, e2
    r_vec, e1_vec = [], []
    for i in range(K):
        r_vec.append(ntt(cbd(PRF(r, i,     64 * ETA1), ETA1)))
    for i in range(K):
        e1_vec.append(    cbd(PRF(r, K + i, 64 * ETA2), ETA2))

    e2 = cbd(PRF(r, 2*K, 64 * ETA2), ETA2)

    # u = A^T * r + e1 (in NTT domain, then INTT)
    u_vec = []
    for i in range(K):
        acc = [0] * 256
        for j in range(K):
            acc = poly_add(acc, poly_basemul(A_T[i][j], r_vec[j]))
        u_vec.append(poly_add(intt(acc), e1_vec[i]))

    # v = t^T * r + e2 + encode(m)
    acc = [0] * 256
    for j in range(K):
        acc = poly_add(acc, poly_basemul(t_vec[j], r_vec[j]))
    v = poly_add(intt(acc), poly_add(e2, decode_poly(m_hash, 1)))

    # Compress ciphertext
    ct  = b''.join(poly_compress(u, DU) for u in u_vec)
    ct += poly_compress(v, DV)

    return ct, KDF(K_bar + H(ct))

# ── Decapsulation ─────────────────────────────────────────────────────────────

def decaps(sk_full: bytes, ct: bytes) -> bytes:
    """
    Kyber512 decapsulation (UNMASKED reference).
    This is the primary attack target.

    Flow:
      1. Decode ciphertext (u, v)
      2. Recover message: m' = v - s^T*u
      3. Re-encapsulate to verify
      4. Return shared secret (or random on failure)
    """
    # Parse secret key
    s_bytes  = sk_full[:K*384]
    pk_bytes = sk_full[K*384:K*384 + K*384 + 32]
    h_pk     = sk_full[K*384 + K*384 + 32:K*384 + K*384 + 64]
    z        = sk_full[K*384 + K*384 + 64:K*384 + K*384 + 96]

    s_vec = [decode_poly(s_bytes[i*384:(i+1)*384], 12) for i in range(K)]

    # Decompress ciphertext
    u_len    = K * (DU * N // 8)
    u_vec    = [poly_decompress(ct[i*(DU*N//8):(i+1)*(DU*N//8)], DU) for i in range(K)]
    v        = poly_decompress(ct[u_len:], DV)

    # ── KEY LEAKAGE POINT: s^T * u ────────────────────────────────────────────
    # This inner product is the primary side-channel leakage target
    s_ntt_vec = [ntt(s) for s in s_vec]
    u_ntt_vec = [ntt(u) for u in u_vec]

    acc = [0] * 256
    for j in range(K):
        # LEAKAGE: secret s_vec[j] multiplied by ciphertext u_vec[j]
        acc = poly_add(acc, poly_basemul(s_ntt_vec[j], u_ntt_vec[j]))

    mp = poly_sub(v, intt(acc))   # m' = v - s^T*u

    # Encode/decode recovered message
    m_prime_bytes = encode_poly([compress(c, 1) for c in mp], 1)

    # Re-encapsulate and compare
    K_bar, r = G(m_prime_bytes + h_pk)
    ct_prime, _ = _reencaps_inner(pk_bytes, m_prime_bytes, r, K_bar)

    # Implicit rejection: use z if ct != ct'
    ok = hmac.compare_digest(ct, ct_prime)
    if ok:
        return KDF(K_bar + H(ct))
    else:
        return KDF(z + H(ct))

def _reencaps_inner(pk: bytes, m: bytes, r: bytes, K_bar: bytes) -> Tuple[bytes, bytes]:
    """Internal re-encapsulation for Fujisaki-Okamoto transform."""
    t_vec = [decode_poly(pk[i*384:(i+1)*384], 12) for i in range(K)]
    rho   = pk[K*384:]

    A_T = gen_matrix(rho, transposed=True)

    r_vec, e1_vec = [], []
    for i in range(K):
        r_vec.append(ntt(cbd(PRF(r, i,     64*ETA1), ETA1)))
    for i in range(K):
        e1_vec.append(    cbd(PRF(r, K+i,  64*ETA2), ETA2))

    e2 = cbd(PRF(r, 2*K, 64*ETA2), ETA2)

    u_vec = []
    for i in range(K):
        acc = [0] * 256
        for j in range(K):
            acc = poly_add(acc, poly_basemul(A_T[i][j], r_vec[j]))
        u_vec.append(poly_add(intt(acc), e1_vec[i]))

    acc = [0] * 256
    for j in range(K):
        acc = poly_add(acc, poly_basemul(t_vec[j], r_vec[j]))
    v = poly_add(intt(acc), poly_add(e2, decode_poly(m, 1)))

    ct  = b''.join(poly_compress(u, DU) for u in u_vec)
    ct += poly_compress(v, DV)
    return ct, KDF(K_bar + H(ct))


# ── Masked Decapsulation (Base Paper) ─────────────────────────────────────────

def masked_decaps(sk_full: bytes, ct: bytes) -> bytes:
    """
    First-order masked Kyber512 decapsulation.
    Implements the base paper's boolean masking scheme.

    Key idea: secret s is split into two shares (s0, s1) such that
    s0 XOR s1 = s (boolean masking in coefficient representation).
    All operations on s are performed on shares.
    """
    # Parse secret key
    s_bytes  = sk_full[:K*384]
    pk_bytes = sk_full[K*384:K*384 + K*384 + 32]
    h_pk     = sk_full[K*384 + K*384 + 32:K*384 + K*384 + 64]
    z        = sk_full[K*384 + K*384 + 64:K*384 + K*384 + 96]

    s_vec = [decode_poly(s_bytes[i*384:(i+1)*384], 12) for i in range(K)]

    # ── BOOLEAN MASKING: split s into shares ──────────────────────────────────
    mask_vec = []
    s0_vec   = []   # share 0 (random)
    s1_vec   = []   # share 1 = s XOR s0

    for s in s_vec:
        mask = [os.urandom(1)[0] % Q for _ in range(N)]  # random mask
        s0   = mask
        s1   = [(si ^ m) % Q for si, m in zip(s, mask)]  # XOR in coeff domain
        mask_vec.append(mask)
        s0_vec.append(s0)
        s1_vec.append(s1)

    # Decompress ciphertext
    u_len = K * (DU * N // 8)
    u_vec = [poly_decompress(ct[i*(DU*N//8):(i+1)*(DU*N//8)], DU) for i in range(K)]
    v     = poly_decompress(ct[u_len:], DV)

    # ── MASKED INNER PRODUCT: (s0 + s1)^T * u = s0^T*u + s1^T*u ─────────────
    # Compute share 0 contribution
    s0_ntt  = [ntt(s0) for s0 in s0_vec]
    u_ntt   = [ntt(u)  for u  in u_vec]

    acc0 = [0] * 256
    for j in range(K):
        acc0 = poly_add(acc0, poly_basemul(s0_ntt[j], u_ntt[j]))

    # Compute share 1 contribution
    s1_ntt = [ntt(s1) for s1 in s1_vec]

    acc1 = [0] * 256
    for j in range(K):
        acc1 = poly_add(acc1, poly_basemul(s1_ntt[j], u_ntt[j]))

    # Combine shares (unmask)
    acc = poly_add(intt(acc0), intt(acc1))
    # Subtract random mask contribution (acc = s^T*u since s0+s1 ~ s under XOR)
    # Note: in proper boolean masking, recombination uses XOR; for arithmetic
    # schemes the approach is: compute each share's product, then combine.

    mp = poly_sub(v, acc)

    m_prime_bytes = encode_poly([compress(c, 1) for c in mp], 1)

    K_bar, r = G(m_prime_bytes + h_pk)
    ct_prime, _ = _reencaps_inner(pk_bytes, m_prime_bytes, r, K_bar)

    ok = hmac.compare_digest(ct, ct_prime)
    if ok:
        return KDF(K_bar + H(ct))
    else:
        return KDF(z + H(ct))


# ── Sanity test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing Kyber512 implementation...")
    pk, sk = keygen()
    ct, ss_enc = encaps(pk)
    ss_dec = decaps(sk, ct)
    print(f"  Shared secrets match (unmasked): {ss_enc == ss_dec}")

    ss_masked = masked_decaps(sk, ct)
    print(f"  Shared secrets match (masked):   {ss_enc == ss_masked}")
    print(f"  Public key size:  {len(pk)} bytes  (expected 800)")
    print(f"  Ciphertext size:  {len(ct)} bytes  (expected 768)")
