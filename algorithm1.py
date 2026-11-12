"""Reference implementation of Algorithm 1 (HybridDecaps) of the paper, in Python.

Runs on top of the repository's core (kyber_sca_research/kyber_sca/core). Set KYBER_SCA_REPO to that folder.
Status of each step (be explicit in the paper):
  IMPLEMENTED   steps 1-22: 512 additive shares, Fisher-Yates shuffle, dummy ops (prob 1/4 from two random bits),
                BaseMulMasked loop (see NOTE 1), w^ accumulation, share-wise inverse NTT, shuffled decode loop (steps 1-26).
  PLACEHOLDER   MaskedShiftRegDecode, MaskedG, MaskedEncrypt, MaskedConstTimeCompare: functionally correct but they
                RECOMBINE the shares internally (marked  # PLACEHOLDER).  They are NOT side-channel protected.
  STAND-IN      Rand(.) and the bits of R come from SHAKE-128, not ASCON-XOF128 (not implemented here).
  NOT MODELLED  This file computes the algorithm; it does not generate leakage. Leakage results come from sim.py.
NOTE 1: BaseMulMasked.  The repository's core (polynomial.py: pw_mul) multiplies NTT-domain values ELEMENT-WISE, not with
        the FIPS 203 base multiplication on coefficient pairs. This file therefore computes t_b[k] = s^_b[j][c]*u^[j][c]
        (k = 256*j + c). The structure (share-wise, one operation per k) is that of BaseMulMasked; the arithmetic is the
        repository's. w^_b[i] = t_b[i] + t_b[256+i], then the linear inverse NTT per share (Algorithm 1, steps 15-16).
"""
import os, sys, hashlib
REPO = os.environ.get("KYBER_SCA_REPO", "/home/claude/work/repo/kyber_sca_research/kyber_sca")
sys.path.insert(0, REPO)
from core.polynomial import Polynomial            # noqa: E402
from core.params import N, Q, K, DU, DV           # noqa: E402
from core.sampling import G, H, KDF               # noqa: E402
from core.kyber import KyberPKE, KyberKEM, _polyvec_from_bytes   # noqa: E402


class RandomSource:
    """R <- XOF(seed).  STAND-IN: SHAKE-128 (the paper specifies ASCON-XOF128)."""
    def __init__(self, seed: bytes):
        self._xof = hashlib.shake_128(seed); self._buf = b""; self._n = 0; self._bits = 0; self._nb = 0
    def _bytes(self, n):
        self._n += n; self._buf = self._xof.digest(self._n); return self._buf[self._n - n:]
    def bit(self):
        if self._nb == 0: self._bits = self._bytes(1)[0]; self._nb = 8
        b = self._bits & 1; self._bits >>= 1; self._nb -= 1; return b
    def rand(self, m):                           # uniform in [0, m) by rejection sampling
        nbytes = (m.bit_length() + 7) // 8 + 1
        lim = (256 ** nbytes // m) * m
        while True:
            v = int.from_bytes(self._bytes(nbytes), "little")
            if v < lim: return v % m


def _fisher_yates(n, R):                          # steps 6-9 / second shuffle
    idx = list(range(n))
    for i in range(n - 1, 0, -1):
        j = R.rand(i + 1); idx[i], idx[j] = idx[j], idx[i]
    return idx


def _dummy_mul(R, sink):                              # DummyButterfly(): discarded random arithmetic
    sink[0] = (sink[0] + R.rand(Q) * R.rand(Q)) % Q


def hybrid_decaps(sk: bytes, ct: bytes, seed: bytes, trace=None):
    """Algorithm 1. `seed` plays the role of TRNG(); `trace` (dict) optionally records shuffle orders/dummy counts."""
    sk_pke, pk = sk[:K * 384], sk[K * 384: K * 384 + 800]
    hpk, z = sk[K * 384 + 800: K * 384 + 832], sk[K * 384 + 832: K * 384 + 864]
    u_sz, v_sz = N * DU // 8, N * DV // 8
    u = [Polynomial.decompress_from_bytes(ct[i * u_sz:(i + 1) * u_sz], DU) for i in range(K)]
    v = Polynomial.decompress_from_bytes(ct[K * u_sz: K * u_sz + v_sz], DV)
    s = _polyvec_from_bytes(sk_pke, K)
    u_hat = [p.ntt() for p in u]

    R = RandomSource(seed)                                                       # step 1
    s_hat = [p.ntt()._c for p in s]                                              # sk stores s; Algorithm 1 takes s^ = NTT(s)
    sh0 = [[R.rand(Q) for _ in range(N)] for _ in range(K)]                      # steps 2-5: shares of s^ in the NTT domain
    sh1 = [[(s_hat[j][i] - sh0[j][i]) % Q for i in range(N)] for j in range(K)]
    sh = [sh0, sh1]
    order = _fisher_yates(K * N, R)                                              # steps 6-9
    t1, t2, sink, ndummy = [0] * (K * N), [0] * (K * N), [0], 0
    for t in range(K * N):                                                       # steps 10-
        k = order[t]; j, c = divmod(k, N)
        if R.bit() & R.bit(): _dummy_mul(R, sink); ndummy += 1
        t1[k] = sh[0][j][c] * u_hat[j]._c[c] % Q                                    # PolyMulMasked (share 1)
        t2[k] = sh[1][j][c] * u_hat[j]._c[c] % Q                                    # PolyMulMasked (share 2)
    w1 = Polynomial([(t1[i] + t1[N + i]) % Q for i in range(N)]).inv_ntt()       # share-wise inverse NTT (NOTE 1)
    w2 = Polynomial([(t2[i] + t2[N + i]) % Q for i in range(N)]).inv_ntt()
    pi = _fisher_yates(N, R)                                                     # second shuffle
    m1, m2 = [0] * N, [0] * N
    for t in range(N):
        i = pi[t]
        if R.bit() & R.bit(): _dummy_mul(R, sink); ndummy += 1
        x1 = (v._c[i] - w1._c[i]) % Q; x2 = (-w2._c[i]) % Q
        xs = (x1 + x2) % Q                                                       # PLACEHOLDER: MaskedShiftRegDecode recombines
        mbit = 1 if (Q // 4) < xs < (3 * Q // 4 + 1) else 0                      # = round(2x/q) mod 2 for x in Z_q
        m1[i] = R.bit(); m2[i] = m1[i] ^ mbit
    m = bytearray(32)                                                            # PLACEHOLDER: MaskedG/MaskedEncrypt/Compare
    for i in range(N): m[i // 8] |= (m1[i] ^ m2[i]) << (i % 8)                   #   recombine m1,m2 before G and re-encryption
    Kbar, r = G(bytes(m) + hpk)
    ct2 = KyberPKE.encrypt(pk, bytes(m), r)
    c = (ct == ct2)
    if trace is not None: trace.update(order=order, pi=pi, dummies=ndummy)
    return KDF(Kbar + H(ct)) if c else KDF(z + H(ct))


if __name__ == "__main__":
    import random
    ok = bad = 0
    for it in range(int(sys.argv[1]) if len(sys.argv) > 1 else 5):
        pk, sk, _ = KyberKEM.keygen(); ct, ss = KyberKEM.encaps(pk)
        ref, _ = KyberKEM.decaps(sk, ct)
        out = hybrid_decaps(sk, ct, os.urandom(32))
        bad_ct = bytearray(ct); bad_ct[3] ^= 1
        ref2, _ = KyberKEM.decaps(sk, bytes(bad_ct)); out2 = hybrid_decaps(sk, bytes(bad_ct), os.urandom(32))
        ok += (out == ref == ss) and (out2 == ref2); bad += not ((out == ref == ss) and (out2 == ref2))
    print("matches reference decaps (valid and tampered ct):", ok, "/", ok + bad)
