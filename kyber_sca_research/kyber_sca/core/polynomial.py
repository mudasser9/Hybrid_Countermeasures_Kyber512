"""
Polynomial arithmetic over Z_Q[X]/(X^N+1) — Kyber512
Verified NTT/InvNTT following CRYSTALS-Kyber reference implementation.
"""

import numpy as np
from .params import N, Q

# -------------------------------------------------------
# Official Kyber zeta table: zetas[i] = 17^brv7(i) mod 3329
# -------------------------------------------------------
def _brv7(n):
    r = 0
    for _ in range(7):
        r = (r << 1) | (n & 1)
        n >>= 1
    return r

ZETAS = [pow(17, _brv7(i), Q) for i in range(128)]
INV_F = pow(128, Q - 2, Q)   # 128^{-1} mod 3329 = 1441


class Polynomial:
    """
    Represents a polynomial in Z_Q[X]/(X^N+1).
    Coefficients stored as Python list for correctness.
    """

    def __init__(self, coeffs=None):
        if coeffs is None:
            self._c = [0] * N
        elif isinstance(coeffs, list):
            assert len(coeffs) == N, f"Expected {N}, got {len(coeffs)}"
            self._c = [int(x) % Q for x in coeffs]
        else:
            arr = list(np.asarray(coeffs, dtype=np.int64).flat)
            assert len(arr) == N
            self._c = [int(x) % Q for x in arr]

    @property
    def coeffs(self):
        return np.array(self._c, dtype=np.int64)

    # -------------------------------------------------------
    # Arithmetic
    # -------------------------------------------------------
    def __add__(self, other):
        return Polynomial([(self._c[i] + other._c[i]) % Q for i in range(N)])

    def __sub__(self, other):
        return Polynomial([(self._c[i] - other._c[i]) % Q for i in range(N)])

    def __eq__(self, other):
        return all(self._c[i] == other._c[i] for i in range(N))

    def copy(self):
        p = Polynomial()
        p._c = self._c[:]
        return p

    # -------------------------------------------------------
    # NTT — Cooley-Tukey butterflies (matches reference C impl)
    # -------------------------------------------------------
    def ntt(self):
        f = self._c[:]
        k, llen = 1, 128
        while llen >= 2:
            start = 0
            while start < N:
                zeta = ZETAS[k]; k += 1
                for j in range(start, start + llen):
                    t = zeta * f[j + llen] % Q
                    f[j + llen] = (f[j] - t) % Q
                    f[j]        = (f[j] + t) % Q
                start += 2 * llen
            llen >>= 1
        p = Polynomial(); p._c = f
        return p

    # -------------------------------------------------------
    # InvNTT — Gentleman-Sande butterflies
    # -------------------------------------------------------
    def inv_ntt(self):
        f = self._c[:]
        k, llen = 127, 2
        while llen <= 128:
            start = 0
            while start < N:
                zeta = ZETAS[k]; k -= 1
                for j in range(start, start + llen):
                    t = f[j]
                    f[j]        = (t + f[j + llen]) % Q
                    f[j + llen] = zeta * (f[j + llen] - t) % Q
                start += 2 * llen
            llen <<= 1
        f = [x * INV_F % Q for x in f]
        p = Polynomial(); p._c = f
        return p

    # -------------------------------------------------------
    # Point-wise multiply (both in NTT domain)
    # -------------------------------------------------------
    def pw_mul(self, other):
        return Polynomial([self._c[i] * other._c[i] % Q for i in range(N)])

    # -------------------------------------------------------
    # Compress / Decompress
    # -------------------------------------------------------
    def compress(self, d: int):
        scale = 1 << d
        c = [int(round(x * scale / Q)) % scale for x in self._c]
        return Polynomial(c)

    def decompress(self, d: int):
        scale = 1 << d
        c = [int(round(x * Q / scale)) % Q for x in self._c]
        return Polynomial(c)

    # -------------------------------------------------------
    # 12-bit packed serialization (for sk/pk polynomials)
    # -------------------------------------------------------
    def to_bytes12(self) -> bytes:
        out = bytearray(384)
        for i in range(128):
            t0 = self._c[2*i]   % Q
            t1 = self._c[2*i+1] % Q
            out[3*i]   = t0 & 0xFF
            out[3*i+1] = (t0 >> 8) | ((t1 & 0xF) << 4)
            out[3*i+2] = t1 >> 4
        return bytes(out)

    @classmethod
    def from_bytes12(cls, data: bytes):
        assert len(data) >= 384
        coeffs = []
        for i in range(128):
            b0, b1, b2 = data[3*i], data[3*i+1], data[3*i+2]
            coeffs.append(b0 | ((b1 & 0xF) << 8))
            coeffs.append((b1 >> 4) | (b2 << 4))
        return cls(coeffs)

    # -------------------------------------------------------
    # Bit-packed serialization (for compressed ciphertext)
    # -------------------------------------------------------
    def compress_to_bytes(self, d: int) -> bytes:
        c = self.compress(d)
        mask = (1 << d) - 1
        buf, nbits = 0, 0
        out = bytearray()
        for coef in c._c:
            buf |= (coef & mask) << nbits
            nbits += d
            while nbits >= 8:
                out.append(buf & 0xFF)
                buf >>= 8; nbits -= 8
        if nbits:
            out.append(buf & 0xFF)
        return bytes(out)

    @classmethod
    def decompress_from_bytes(cls, data: bytes, d: int):
        mask = (1 << d) - 1
        coeffs, buf, nbits, idx = [], 0, 0, 0
        while len(coeffs) < N:
            while nbits < d and idx < len(data):
                buf |= data[idx] << nbits
                nbits += 8; idx += 1
            coeffs.append(buf & mask)
            buf >>= d; nbits -= d
        return cls(coeffs).decompress(d)

    # -------------------------------------------------------
    # Message encode / decode (32-byte ↔ polynomial)
    # -------------------------------------------------------
    def encode_msg(self) -> bytes:
        """Decode poly coefficients to nearest bit → 32-byte message."""
        msg = bytearray(32)
        for i in range(N):
            if self._c[i] > Q // 4 and self._c[i] < 3 * Q // 4:
                msg[i >> 3] |= 1 << (i & 7)
        return bytes(msg)

    @classmethod
    def decode_msg(cls, msg: bytes):
        """Encode 32-byte message as polynomial ∈ {0, ⌈Q/2⌉}^N."""
        assert len(msg) == 32
        coeffs = [((Q + 1) // 2 if (msg[i >> 3] >> (i & 7)) & 1 else 0) for i in range(N)]
        return cls(coeffs)

    # -------------------------------------------------------
    def hamming_weight(self) -> int:
        return sum(bin(c).count('1') for c in self._c)

    def __repr__(self):
        return f"Polynomial({self._c[:6]}...)"
