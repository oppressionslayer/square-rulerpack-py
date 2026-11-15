"""Lars Rocha's XOR build system: the prefix-XOR Cayley table of a sequence, and the odd-square
rulers it is read against.

Given an index sequence (here: the doubled primes), its *anchor row* is the exclusive prefix XOR

    a[0] = 0,  a[k] = index[0] ^ index[1] ^ ... ^ index[k-1]

and the *build system* is the table  T[i, j] = a[i] ^ a[j].  T is symmetric with a zero diagonal,
every row is row 0 XORed with one constant, and the whole table rebuilds from its row 0; the
index comes back from any row as  row[:-1] ^ row[1:].

A *ruler* XORed across the columns (``table_with_ruler``) puts the ruler on the diagonal, since
a[i] ^ a[i] = 0, and leaves column 0 untouched.  With the odd squares as the ruler each row is a
running XOR cipher: the squares are the key, the primes the message, so ``decode_row`` strips
the squares and returns the doubled primes exactly.  Column 0 is the pure walk  1, 5, 3, 9, 7,
17, 11, 41, ...  where consecutive entries XOR to 2·p; it stays within 8 bits for the first 26
primes while their sum passes 2000.

The lattice  (1 + 4k)² − anchor  is the value set of the quadratic-sieve polynomial  x² − n  at
n = anchor over the odd x (every odd number is ±(1 + 4k)); its odd prime divisors are exactly the
primes p with  (anchor / p) = 1  or  p | anchor, which is the sieve's factor-base rule.

Measured facts (2026-11-14, this module's tests): the first 26 primes XOR-walk to 41 twice,
because 19 ^ 23 ^ 29 ^ 31 ^ 37 ^ 41 ^ 43 ^ 47 ^ 53 ^ 59 = 0.  The walk revisits earlier values
far more often than random odd numbers of the same size do (2,733 revisits against 478 for the
first 10,000 primes), but random odd numbers that grow as slowly as the primes revisit just as
often (2,574), and shuffling the primes drops it to 362: the effect is the slow monotone growth,
not primality.  The raw adjacent XORs of row 0 land on 2·prime at the same rate as random odd
numbers do (1,029 against 1,058 of 5,000).

Credit: the table, the anchor walk and the rulers are Lars Rocha's (``numtoolz.buildsystem``);
Claude measured the facts above.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "xor_prefix", "buildsystem", "recover_index", "lattice", "odd_squares",
    "table_with_ruler", "decode_row", "anchor_walk", "lattice_divisors", "first_primes",
]


def xor_prefix(index) -> np.ndarray:
    """Exclusive prefix XOR of ``index``, with a leading 0 (the anchor row)."""
    index = np.asarray(index, dtype=np.int64)
    return np.concatenate(([0], np.bitwise_xor.accumulate(index)))


def buildsystem(index) -> np.ndarray:
    """T[i, j] = a[i] ^ a[j] with a = xor_prefix(index): symmetric, zero diagonal."""
    a = xor_prefix(index)
    return a[:, None] ^ a[None, :]


def recover_index(a) -> np.ndarray:
    """The index back from the anchor row (or from a table, whose row 0 is the anchor)."""
    a = np.asarray(a, dtype=np.int64)
    if a.ndim == 2:
        a = a[0]
    return a[:-1] ^ a[1:]


def lattice(anchor: int = 0, k_from: int = -127, k_to: int = 127) -> np.ndarray:
    """sorted((1 + 4k)² − anchor for k in range(k_from, k_to)).  anchor=0 gives the odd squares."""
    k = np.arange(k_from, k_to, dtype=np.int64)
    return np.sort((1 + 4 * k) ** 2 - anchor)


def odd_squares(count: int) -> np.ndarray:
    """1, 9, 25, 49, ...: the first ``count`` odd squares (the ruler)."""
    k = np.arange(count, dtype=np.int64)
    return (2 * k + 1) ** 2


def table_with_ruler(index, ruler) -> np.ndarray:
    """buildsystem(index) ^ ruler across the columns: the ruler appears on the diagonal."""
    t = buildsystem(index)
    ruler = np.asarray(ruler, dtype=np.int64)[: t.shape[1]]
    return t ^ ruler[None, :]


def decode_row(row, ruler) -> np.ndarray:
    """The index back from any row of table_with_ruler: strip the ruler, then adjacent XOR."""
    row = np.asarray(row, dtype=np.int64)
    ruler = np.asarray(ruler, dtype=np.int64)[: len(row)]
    a = row ^ ruler
    return a[:-1] ^ a[1:]


def anchor_walk(index, start: int = 1) -> np.ndarray:
    """Column 0 of table_with_ruler with a ruler whose first entry is ``start``: the walk
    start, start ^ index[0], start ^ index[0] ^ index[1], ..."""
    return xor_prefix(index) ^ start


def first_primes(count: int) -> np.ndarray:
    """The first ``count`` primes (small sieve)."""
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    hi = 15 if count < 6 else int(count * (np.log(count) + np.log(np.log(count)))) + 10
    s = np.ones(hi, dtype=bool)
    s[:2] = False
    for q in range(2, int(hi**0.5) + 1):
        if s[q]:
            s[q * q::q] = False
    return np.flatnonzero(s)[:count].astype(np.int64)


def lattice_divisors(anchor: int, primes) -> np.ndarray:
    """The primes in ``primes`` that divide some (1 + 4k)² − anchor: p | anchor, or anchor is a
    nonzero square modulo p (the factor-base rule of the quadratic sieve for n = anchor)."""
    out = []
    for p in map(int, np.asarray(primes)):
        if p == 2:
            out.append(p if (1 - anchor) % 2 == 0 else None)
            continue
        if anchor % p == 0 or pow(anchor % p, (p - 1) // 2, p) == 1:
            out.append(p)
    return np.array([p for p in out if p is not None], dtype=np.int64)
