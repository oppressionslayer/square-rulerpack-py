"""64-bit cofactor arithmetic compiled with numba: Miller–Rabin and Pollard–Brent on machine
integers for the double-large-prime cofactors (all below 2^52 here), instead of Python ``pow`` on
arbitrary-precision ints.  Falls back to ``arith`` when numba is missing or the number is larger.

The modular product uses the double-precision quotient estimate: for a, b < m < 2^52 the float
quotient of a·b/m is off by at most one, and a·b − q·m computed in wrapping 64-bit arithmetic is
the exact remainder up to one correction.  Miller–Rabin with the twelve prime bases up to 37 is
deterministic far past 2^64 (Jaeschke; Sorenson–Webster), the same bases ``arith`` and the Rust
engine use.  Pollard's rho with Brent's cycle detection and batched gcds is Brent 1980.

Measured 2026-11-14 on a 60-digit number, one worker: cofactor classification fell from 48 s to
under 2 s of a 145 s run (see TIMINGS.md).  Written by Claude for Lars Rocha's square-rulerpack.
"""
from __future__ import annotations

import math

from .arith import is_probable_prime, pollard_brent

try:
    import numba as _nb
    HAVE_NUMBA = True
except ImportError:                                   # pragma: no cover
    _nb = None
    HAVE_NUMBA = False

FAST_LIMIT = 1 << 52

__all__ = ["HAVE_NUMBA", "FAST_LIMIT", "is_prime_cofactor", "split_cofactor"]

if HAVE_NUMBA:
    @_nb.njit(cache=True, nogil=True, inline="always")
    def _mulmod(a, b, m):
        q = _nb.int64(_nb.float64(a) * _nb.float64(b) / _nb.float64(m))
        r = a * b - q * m                               # exact modulo 2^64 (wrapping int64)
        if r < 0:
            r += m
        elif r >= m:
            r -= m
        if r < 0:                                       # quotient estimate two off (never seen, kept safe)
            r += m
        elif r >= m:
            r -= m
        return r

    @_nb.njit(cache=True, nogil=True)
    def _powmod(a, e, m):
        r = _nb.int64(1)
        a %= m
        while e > 0:
            if e & 1:
                r = _mulmod(r, a, m)
            a = _mulmod(a, a, m)
            e >>= 1
        return r

    @_nb.njit(cache=True, nogil=True)
    def _is_prime_u52(n):
        if n < 2:
            return False
        for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if n % p == 0:
                return n == p
        d = n - 1
        s = 0
        while d & 1 == 0:
            d >>= 1
            s += 1
        for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            x = _powmod(_nb.int64(a), d, n)
            if x == 1 or x == n - 1:
                continue
            composite = True
            for _ in range(s - 1):
                x = _mulmod(x, x, n)
                if x == n - 1:
                    composite = False
                    break
            if composite:
                return False
        return True

    @_nb.njit(cache=True, nogil=True)
    def _gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    @_nb.njit(cache=True, nogil=True)
    def _brent_u52(n, seed, max_iter):
        """A nontrivial factor of the odd composite n < 2^52, or 0 if the budget runs out."""
        if n % 2 == 0:
            return _nb.int64(2)
        state = seed ^ n
        for _attempt in range(8):
            state = (state * 6364136223846793005 + 1442695040888963407) & 0x7FFFFFFFFFFFFFFF
            y = state % (n - 1) + 1
            state = (state * 6364136223846793005 + 1442695040888963407) & 0x7FFFFFFFFFFFFFFF
            c = state % (n - 1) + 1
            m = 128
            g = _nb.int64(1); r = _nb.int64(1); q = _nb.int64(1)
            it = 0
            x = y; ys = y
            while g == 1:
                x = y
                for _ in range(r):
                    y = (_mulmod(y, y, n) + c) % n
                k = 0
                while k < r and g == 1:
                    ys = y
                    lim = min(m, r - k)
                    for _ in range(lim):
                        y = (_mulmod(y, y, n) + c) % n
                        q = _mulmod(q, abs(x - y), n)
                    g = _gcd(q, n)
                    k += m
                r *= 2
                it += r
                if it > max_iter:
                    break
            if g == n:
                g = _nb.int64(1)
                while g == 1:
                    ys = (_mulmod(ys, ys, n) + c) % n
                    g = _gcd(abs(x - ys), n)
            if 1 < g < n:
                return g
        return _nb.int64(0)


def is_prime_cofactor(v: int) -> bool:
    """Primality of a cofactor: compiled Miller–Rabin below 2^52, ``arith`` above."""
    if HAVE_NUMBA and v < FAST_LIMIT:
        return bool(_is_prime_u52(v))
    return is_probable_prime(v)


def split_cofactor(v: int, seed: int) -> int | None:
    """A nontrivial factor of the odd composite v: compiled Brent below 2^52, ``arith`` above."""
    if HAVE_NUMBA and v < FAST_LIMIT:
        f = int(_brent_u52(v, seed & 0x7FFFFFFF, 1 << 22))
        return f if f else None
    return pollard_brent(v, seed=seed)
