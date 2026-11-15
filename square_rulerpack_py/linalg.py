"""GF(2) linear algebra for the relation matrix: singleton pruning (as the Rust
``prune_singletons``), dense bit-packed Gaussian elimination in numpy, and block Lanczos
(``lanczos``) above ``LANCZOS_FROM`` rows, dispatched exactly as the Rust engine's ``siqs::solve``.

Each relation is a row; column c is set when that relation has an odd exponent at c. A
dependency is a set of rows whose XOR is zero: multiplying those relations gives a square on
the right-hand side.
"""
from __future__ import annotations

import os
from collections import deque

import numpy as np

from . import lanczos as _lanczos

__all__ = ["prune_singletons", "dependencies", "dense_dependencies", "dense_matrix", "LANCZOS_FROM"]

LANCZOS_FROM = 6000     # rows; the Rust engine's threshold


def prune_singletons(odd: list[list[int]], cols: int) -> list[int]:
    """Indices of the rows that survive removing, repeatedly, every row that is the only one
    with a set bit in some column (such a row can be in no dependency)."""
    count = [0] * cols
    owner = [0] * cols
    for i, r in enumerate(odd):
        for c in r:
            count[c] += 1
            owner[c] ^= i
    alive = [True] * len(odd)
    queue = deque(c for c in range(cols) if count[c] == 1)
    while queue:
        c = queue.popleft()
        if count[c] != 1:
            continue
        i = owner[c]
        if not alive[i]:
            continue
        alive[i] = False
        for d in odd[i]:
            count[d] -= 1
            owner[d] ^= i
            if count[d] == 1:
                queue.append(d)
    return [i for i in range(len(odd)) if alive[i]]


def dense_matrix(rows: list[list[int]], cols: int) -> np.ndarray:
    """The 0/1 matrix (rows × cols) as uint8, for looking at small cases."""
    m = np.zeros((len(rows), cols), dtype=np.uint8)
    for i, r in enumerate(rows):
        m[i, r] = 1
    return m


def dependencies(rows: list[list[int]], cols: int, max_deps: int = 64, lanczos: bool | None = None,
                 use_numba: bool = True, stats: dict | None = None) -> list[list[int]]:
    """Up to ``max_deps`` dependencies of the sparse 0/1 matrix: block Lanczos when there are
    ``LANCZOS_FROM`` rows or more (``lanczos=True/False`` or ``SIQS_LANCZOS=1/0`` force it), seeds
    1..4 until 32 are found, else the dense elimination. ``stats`` receives which solver ran."""
    if lanczos is None:
        env = os.environ.get("SIQS_LANCZOS")
        lanczos = {"1": True, "0": False}.get(env, len(rows) >= LANCZOS_FROM)
    if lanczos and rows:
        out: list[list[int]] = []
        st: dict = {}
        iterations = 0
        for seed in (1, 2, 3, 4):
            out.extend(_lanczos.dependencies(rows, cols, seed, use_numba=use_numba, stats=st))
            iterations += st.get("iterations", 0)
            if len(out) >= 32:
                break
        if out:
            if stats is not None:
                stats.update(solver="block Lanczos", iterations=iterations, seeds=seed, dependencies=len(out))
            return out[:max_deps]
    if stats is not None:
        stats.update(solver="dense elimination")
    return dense_dependencies(rows, cols, max_deps)


def dense_dependencies(rows: list[list[int]], cols: int, max_deps: int = 64) -> list[list[int]]:
    """Up to ``max_deps`` dependencies (lists of row indices) of the sparse 0/1 matrix given
    as per-row column lists. Dense elimination over 64-bit words: the matrix is packed as
    [columns | identity]; eliminating the column part column by column leaves, in every row
    that never became a pivot, the identity part naming the rows that XOR to zero."""
    n = len(rows)
    if n == 0:
        return []
    W = -(-cols // 64)
    Wi = -(-n // 64)
    M = np.zeros((n, W + Wi), dtype=np.uint64)
    one = np.uint64(1)
    for i, r in enumerate(rows):
        for c in r:
            M[i, c >> 6] |= one << np.uint64(c & 63)
        M[i, W + (i >> 6)] |= one << np.uint64(i & 63)
    used = np.zeros(n, dtype=bool)
    # columns that appear at all, in order (columns nobody has need no pass)
    present = np.zeros(cols, dtype=bool)
    for r in rows:
        present[r] = True
    for c in np.flatnonzero(present):
        w, bit = int(c) >> 6, np.uint64(int(c) & 63)
        has = ((M[:, w] >> bit) & one).astype(bool)
        with_bit = np.flatnonzero(has)
        if len(with_bit) == 0:
            continue
        cand = with_bit[~used[with_bit]]
        if len(cand) == 0:
            continue
        piv = cand[0]
        used[piv] = True
        others = with_bit[with_bit != piv]
        if len(others):
            M[others] ^= M[piv]
    deps = []
    for i in np.flatnonzero(~used):
        assert not M[i, :W].any(), "a non-pivot row kept a column bit"
        words = M[i, W:]
        members = []
        for wi in np.flatnonzero(words):
            word = int(words[wi])
            while word:
                low = word & -word
                members.append(int(wi) * 64 + low.bit_length() - 1)
                word ^= low
        deps.append(members)
        if len(deps) >= max_deps:
            break
    return deps
