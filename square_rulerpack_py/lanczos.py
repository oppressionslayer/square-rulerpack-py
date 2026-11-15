"""Block Lanczos over GF(2) (Montgomery, 1995) for the dependencies of a large sparse relation
matrix, where dense elimination stops fitting. A port of the Rust engine's ``rulerpack::lanczos``
(Lars Rocha / Claude, written from the paper; nothing here is ported from any GPL siever) into
numpy ``uint64`` blocks, with numba kernels for the three hot loops and numpy fallbacks.

The matrix is ``rows`` relations over ``cols`` columns, each relation a sorted list of the
columns with an odd exponent. A dependency is a set of relations whose columns cancel: a vector
z with Bᵀz = 0. The iteration works with the symmetric A = B·Bᵀ applied to blocks of 64 vectors
at once (an n×64 block is a ``uint64`` array of length n, one word a row), solves A·X = A·Y for a
random Y so that X − Y lies in the kernel of A, and a final small elimination turns kernel
vectors of A into kernel vectors of Bᵀ. Every dependency is checked exactly before it is returned.
"""
from __future__ import annotations

import numpy as np

try:
    import numba as _nb
    HAVE_NUMBA = True
except ImportError:                                   # pragma: no cover
    _nb = None
    HAVE_NUMBA = False

__all__ = ["dependencies", "HAVE_NUMBA"]

MASK64 = (1 << 64) - 1


# ----------------------------------------------------------------------------- the sparse matrix

class Sparse:
    """B as CSR; ``bt`` is Bᵀ·X (cols × 64), ``b`` is B·Y (rows × 64), ``a`` is B·Bᵀ·X."""

    def __init__(self, rows: list[list[int]], cols: int, use_numba: bool = True):
        self.n = len(rows)
        self.cols = cols
        counts = np.fromiter((len(r) for r in rows), dtype=np.int64, count=self.n)
        self.indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        self.indices = np.fromiter((c for r in rows for c in r), dtype=np.int32, count=int(counts.sum()))
        self.row_of = np.repeat(np.arange(self.n, dtype=np.int32), counts)
        self.fast = HAVE_NUMBA and use_numba

    def bt(self, x: np.ndarray) -> np.ndarray:
        if self.fast:
            return _bt_kernel(x, self.indptr, self.indices, self.cols)
        out = np.zeros(self.cols, dtype=np.uint64)
        np.bitwise_xor.at(out, self.indices, x[self.row_of])
        return out

    def b(self, y: np.ndarray) -> np.ndarray:
        if self.fast:
            return _b_kernel(y, self.indptr, self.indices, self.n)
        out = np.zeros(self.n, dtype=np.uint64)
        np.bitwise_xor.at(out, self.row_of, y[self.indices])
        return out

    def a(self, x: np.ndarray) -> np.ndarray:
        return self.b(self.bt(x))


# ----------------------------------------------------------------------------- block arithmetic

if HAVE_NUMBA:
    @_nb.njit(cache=True, nogil=True)
    def _bt_kernel(x, indptr, indices, cols):          # pragma: no cover (compiled)
        out = np.zeros(cols, dtype=np.uint64)
        for r in range(indptr.shape[0] - 1):
            w = x[r]
            if w != 0:
                for k in range(indptr[r], indptr[r + 1]):
                    out[indices[k]] ^= w
        return out

    @_nb.njit(cache=True, nogil=True)
    def _b_kernel(y, indptr, indices, n):              # pragma: no cover (compiled)
        out = np.zeros(n, dtype=np.uint64)
        for r in range(n):
            acc = np.uint64(0)
            for k in range(indptr[r], indptr[r + 1]):
                acc ^= y[indices[k]]
            out[r] = acc
        return out

    @_nb.njit(cache=True, nogil=True)
    def _inner_kernel(v, w):                           # pragma: no cover (compiled)
        out = np.zeros(64, dtype=np.uint64)
        for r in range(v.shape[0]):
            a = v[r]
            if a == 0:
                continue
            b = w[r]
            for i in range(64):
                if (a >> np.uint64(i)) & np.uint64(1):
                    out[i] ^= b
        return out

    @_nb.njit(cache=True, nogil=True)
    def _mul_small_kernel(v, m):                       # pragma: no cover (compiled)
        # eight 256-entry tables, one per byte of the input word
        tables = np.zeros((8, 256), dtype=np.uint64)
        for k in range(8):
            for byte in range(1, 256):
                low = byte & (byte - 1)
                bit = byte ^ low
                idx = 0
                while (bit >> idx) != 1:
                    idx += 1
                tables[k, byte] = tables[k, low] ^ m[k * 8 + idx]
        out = np.empty(v.shape[0], dtype=np.uint64)
        for r in range(v.shape[0]):
            w = v[r]
            acc = np.uint64(0)
            for k in range(8):
                acc ^= tables[k, (w >> np.uint64(8 * k)) & np.uint64(0xFF)]
            out[r] = acc
        return out


def inner(v: np.ndarray, w: np.ndarray, fast: bool) -> np.ndarray:
    """Vᵀ·W as 64×64: row i collects the W rows where V has bit i."""
    if fast:
        return _inner_kernel(v, w)
    out = np.zeros(64, dtype=np.uint64)
    for i in range(64):
        sel = ((v >> np.uint64(i)) & np.uint64(1)).astype(bool)
        if sel.any():
            out[i] = np.bitwise_xor.reduce(w[sel])
    return out


def mul_small(v: np.ndarray, m: np.ndarray, fast: bool) -> np.ndarray:
    """V·M: row r becomes the XOR of M's rows picked by the bits of V[r]."""
    if fast:
        return _mul_small_kernel(v, m)
    out = np.zeros(v.shape[0], dtype=np.uint64)
    for i in range(64):
        sel = ((v >> np.uint64(i)) & np.uint64(1)).astype(bool)
        out[sel] ^= m[i]
    return out


def small_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(64, dtype=np.uint64)
    for i in range(64):
        bits = int(a[i])
        while bits:
            j = (bits & -bits).bit_length() - 1
            out[i] ^= b[j]
            bits &= bits - 1
    return out


def identity() -> np.ndarray:
    return (np.uint64(1) << np.arange(64, dtype=np.uint64))


def mask_cols(a: np.ndarray, mask: int) -> np.ndarray:
    return a & np.uint64(mask)


def choose(t: np.ndarray, prev_mask: int) -> tuple[np.ndarray, int]:
    """Montgomery's choice of S_i and W_i⁻¹ from T = Vᵢᵀ·A·Vᵢ, preferring the columns left out of
    the previous S. Returns (W_inv with zero rows and columns outside S, the mask of S)."""
    left = [int(x) for x in t]
    right = [1 << i for i in range(64)]
    order = [c for c in range(64) if not (prev_mask >> c) & 1] + [c for c in range(64) if (prev_mask >> c) & 1]
    s_mask = 0
    for j, cj in enumerate(order):
        k = next((k for k in range(j, 64) if (left[order[k]] >> cj) & 1), None)
        if k is not None:
            ck = order[k]
            left[cj], left[ck] = left[ck], left[cj]
            right[cj], right[ck] = right[ck], right[cj]
            for r in range(64):
                if r != cj and (left[r] >> cj) & 1:
                    left[r] ^= left[cj]
                    right[r] ^= right[cj]
            s_mask |= 1 << cj
        else:
            k = next((k for k in range(j, 64) if (right[order[k]] >> cj) & 1), None)
            if k is not None:
                ck = order[k]
                left[cj], left[ck] = left[ck], left[cj]
                right[cj], right[ck] = right[ck], right[cj]
                for r in range(64):
                    if r != cj and (right[r] >> cj) & 1:
                        left[r] ^= left[cj]
                        right[r] ^= right[cj]
                left[cj] = 0
                right[cj] = 0
    winv = np.zeros(64, dtype=np.uint64)
    for r in range(64):
        if (s_mask >> r) & 1:
            winv[r] = np.uint64(right[r] & s_mask)
    return winv, s_mask


# ----------------------------------------------------------------------------- the iteration

def _xorshift_block(n: int, seed: int) -> np.ndarray:
    rng = (seed ^ 0x9E3779B97F4A7C15) & MASK64
    out = np.empty(n, dtype=np.uint64)
    for i in range(n):
        rng ^= (rng << 13) & MASK64
        rng ^= rng >> 7
        rng ^= (rng << 17) & MASK64
        out[i] = rng
    return out


def dependencies(rows: list[list[int]], cols: int, seed: int = 1, use_numba: bool = True,
                 stats: dict | None = None) -> list[list[int]]:
    """Up to 64 dependencies: sets of row indices whose columns cancel. ``seed`` picks the random
    start; another seed gives other dependencies. ``stats`` (if given) receives the iteration
    count and the number of kernel vectors that checked."""
    m = Sparse(rows, cols, use_numba)
    n, fast = m.n, m.fast
    if n == 0:
        return []
    y = _xorshift_block(n, seed)
    v0 = m.a(y)
    x = np.zeros(n, dtype=np.uint64)
    v_prev = np.zeros(n, dtype=np.uint64)
    v_prev2 = np.zeros(n, dtype=np.uint64)
    v = v0.copy()
    ident = identity()
    winv_prev = np.zeros(64, dtype=np.uint64)
    winv_prev2 = np.zeros(64, dtype=np.uint64)
    vav_prev = np.zeros(64, dtype=np.uint64)
    vaav_prev = np.zeros(64, dtype=np.uint64)
    s_prev = MASK64
    iterations = 0
    while True:
        av = m.a(v)
        vav = inner(v, av, fast)
        vaav = inner(av, av, fast)
        if not vav.any():
            break
        winv, s = choose(vav, s_prev)
        # X += V · Winv · Vᵀ V0
        vtv0 = inner(v, v0, fast)
        x ^= mul_small(v, small_mul(winv, vtv0), fast)
        # V_{i+1} = A V S Sᵀ + V D + V_{i-1} E + V_{i-2} F
        vaav_s = mask_cols(vaav, s)
        d = ident ^ small_mul(winv, vaav_s ^ vav)
        e = small_mul(winv_prev, mask_cols(vav, s))
        inner_f = ident ^ small_mul(vav_prev, winv_prev)
        f = small_mul(winv_prev2, mask_cols(small_mul(inner_f, mask_cols(vaav_prev, s_prev) ^ vav_prev), s))
        v_next = (av & np.uint64(s)) ^ mul_small(v, d, fast) ^ mul_small(v_prev, e, fast) ^ mul_small(v_prev2, f, fast)
        v_prev2, v_prev, v = v_prev, v, v_next
        winv_prev2, winv_prev = winv_prev, winv
        vav_prev, vaav_prev, s_prev = vav, vaav, s
        iterations += 1
        if iterations > n // 32 + 100:          # did not converge: give back what there is
            break
    # kernel of A: columns of Z = [X ⊕ Y | V_last]; make kernel vectors of Bᵀ from them
    z0 = x ^ y
    z1 = v
    u0, u1 = m.bt(z0), m.bt(z1)
    # the 128 columns of [U0 | U1] as bit vectors over cols (Python ints); their linear dependencies
    pivots: dict[int, tuple[int, int]] = {}        # top bit -> (vector, combination)
    combos: list[int] = []
    for j in range(128):
        src, bit = (u0, j) if j < 64 else (u1, j - 64)
        col_bits = ((src >> np.uint64(bit)) & np.uint64(1)).astype(np.uint8)
        vcol = int.from_bytes(np.packbits(col_bits, bitorder="little").tobytes(), "little")
        combo = 1 << j
        while vcol:
            top = vcol.bit_length() - 1
            piv = pivots.get(top)
            if piv is None:
                pivots[top] = (vcol, combo)
                break
            vcol ^= piv[0]
            combo ^= piv[1]
        else:
            combos.append(combo)
    out: list[list[int]] = []
    for combo in combos:
        lo = np.uint64(combo & MASK64)
        hi = np.uint64(combo >> 64)
        parity = (np.bitwise_count(z0 & lo) + np.bitwise_count(z1 & hi)) & 1
        dep = np.flatnonzero(parity)
        if len(dep) == 0:
            continue
        # check Bᵀ z = 0 exactly before handing it on
        col_sum = np.zeros(cols, dtype=np.uint8)
        for r in dep.tolist():
            col_sum[rows[r]] ^= 1
        if not col_sum.any():
            out.append(dep.tolist())
    if stats is not None:
        stats["iterations"] = iterations
        stats["kernel_vectors"] = len(combos)
        stats["dependencies"] = len(out)
    return out
