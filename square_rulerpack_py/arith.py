"""Integer arithmetic the sieve needs: primes, Legendre symbols, modular square roots,
probable-prime tests and a Pollard–Brent splitter for double-large-prime cofactors.

Everything here is plain Python integers (arbitrary precision) plus numpy for the prime sieve,
so each function is short enough to read in one sitting. The Rust engine (square-rulerpack 0.2.0)
has the same functions in ``siqs.rs``, ``arith64.rs`` and ``uecm.rs``; the Python versions keep
the same names where it helps to read the two side by side.
"""
from __future__ import annotations

import math
import random

import numpy as np

__all__ = [
    "primes_upto", "legendre", "sqrt_mod", "modinv", "is_probable_prime", "is_square",
    "pollard_brent", "XorShift64", "log2_int",
]


def primes_upto(n: int) -> np.ndarray:
    """All primes ≤ n as an int64 array (sieve of Eratosthenes on a boolean array)."""
    if n < 2:
        return np.zeros(0, dtype=np.int64)
    flags = np.ones(n + 1, dtype=bool)
    flags[:2] = False
    for p in range(2, math.isqrt(n) + 1):
        if flags[p]:
            flags[p * p::p] = False
    return np.flatnonzero(flags).astype(np.int64)


def legendre(a: int, p: int) -> int:
    """The Legendre symbol (a/p) for an odd prime p: 1 if a is a nonzero square mod p,
    -1 if not, 0 if p divides a. Euler's criterion: a^((p-1)/2) mod p."""
    a %= p
    if a == 0:
        return 0
    r = pow(a, (p - 1) // 2, p)
    return -1 if r == p - 1 else r


def sqrt_mod(a: int, p: int) -> int:
    """A square root of a modulo the odd prime p (a must be a quadratic residue), by
    Tonelli–Shanks. Returns the root in [0, p); the other root is p − root."""
    a %= p
    if a == 0:
        return 0
    if p == 2:
        return a
    if p % 4 == 3:
        return pow(a, (p + 1) // 4, p)
    # write p − 1 = q·2^s with q odd
    q, s = p - 1, 0
    while q % 2 == 0:
        q //= 2
        s += 1
    z = 2
    while legendre(z, p) != -1:
        z += 1
    m, c, t, r = s, pow(z, q, p), pow(a, q, p), pow(a, (q + 1) // 2, p)
    while t != 1:
        i, t2 = 0, t
        while t2 != 1:
            t2 = t2 * t2 % p
            i += 1
        b = pow(c, 1 << (m - i - 1), p)
        m, c, t, r = i, b * b % p, t * b * b % p, r * b % p
    return r


def modinv(a: int, m: int) -> int:
    """a⁻¹ mod m (a and m coprime)."""
    return pow(a, -1, m)


_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_probable_prime(n: int, rounds: int = 16) -> bool:
    """Miller–Rabin. Deterministic with the first twelve prime bases below 3.3·10^24 (the
    same bases the Rust ``is_prime_u64`` uses), plus ``rounds`` random bases above that."""
    if n < 2:
        return False
    for p in _MR_BASES:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    bases = list(_MR_BASES)
    if n >= 3_317_044_064_679_887_385_961_981:
        rng = random.Random(n & 0xFFFFFFFF)
        bases += [rng.randrange(2, n - 1) for _ in range(rounds)]
    for a in bases:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def is_square(n: int) -> bool:
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n


def pollard_brent(n: int, seed: int = 1, max_iter: int = 1 << 22) -> int | None:
    """A nontrivial factor of the odd composite n by Pollard's rho with Brent's cycle
    detection and batched gcds, or None if the budget runs out. This is what the Rust engine's
    tiny ECM (``uecm.rs``) does for the cofactor of a double-large-prime candidate: the
    cofactor is below 2^63 and is known to be composite and not a square."""
    if n % 2 == 0:
        return 2
    rng = random.Random(seed ^ n)
    for _attempt in range(8):
        y, c, m = rng.randrange(1, n), rng.randrange(1, n), 128
        g, r, q = 1, 1, 1
        it = 0
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
            it += r
            if it > max_iter:
                break
        if g == n:
            # the batch overshot: step one at a time from the saved point
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
        if 1 < g < n:
            return g
    return None


class XorShift64:
    """The 64-bit xorshift generator ``choose_a`` in the Rust engine uses, seeded the same
    way, so that family number i picks the same primes for a here and there."""

    MASK = (1 << 64) - 1

    def __init__(self, seed: int):
        self.state = ((seed * 0x9E3779B97F4A7C15) + 0xD1B54A32D192ED03) & self.MASK

    def next(self) -> int:
        x = self.state
        x ^= (x << 13) & self.MASK
        x ^= x >> 7
        x ^= (x << 17) & self.MASK
        self.state = x
        return x


def log2_int(n: int) -> float:
    """log2 of a positive integer of any size (``math.log2`` handles big ints exactly enough)."""
    return math.log2(n)
