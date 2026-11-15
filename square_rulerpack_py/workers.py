"""Relation collection across processes. Each worker takes family indices from a shared
counter, sieves the family, and sends the relations back; the parent merges them in family
order, exactly as the Rust engine's threads do, so the relation set depends only on the
family indices, not on how many workers ran."""
from __future__ import annotations

import itertools
import multiprocessing as mp
import os
from typing import Iterator

from .engine import FamilySieve, Settings, choose_a, q2_polynomials, serial_families, HAVE_NUMBA

_CTX: dict = {}


def default_workers() -> int:
    """Half the hardware threads: one worker per physical core on a hyperthreaded CPU."""
    return max(1, (os.cpu_count() or 2) // 2)


def _init(ctx: dict) -> None:
    _CTX.update(ctx)


def _task(idx: int):
    c = _CTX
    fam = choose_a(c["kn"], c["base"], c["params"].half, idx, c["q2"], c["settings"].close_a)
    rels, npolys = FamilySieve(c["kn"], c["base"], fam, c["params"], c["threshold"], c["exact"], c["settings"]).run()
    return idx, rels, npolys


def _warm_kernel(settings: Settings) -> None:
    """Compile the numba kernel in the parent once, so forked workers inherit it."""
    if HAVE_NUMBA and settings.use_numba:
        import numpy as np
        from .engine import _sieve_kernel
        acc = np.zeros(16, dtype=np.int16)
        _sieve_kernel(acc, np.array([1], dtype=np.int64), np.array([2], dtype=np.int64), np.array([3], dtype=np.int64),
                      np.array([2], dtype=np.int16), np.array([True]), 16)


def family_stream(kn, base, params, threshold: float, exact: bool, settings: Settings, workers: int = 1,
                  start: int = 1) -> Iterator[tuple[int, list, int]]:
    """Families in index order from ``workers`` processes (or this process when workers ≤ 1)."""
    if workers <= 1:
        yield from serial_families(kn, base, params, threshold, exact, settings, start)
        return
    _warm_kernel(settings)
    ctx = {"kn": kn, "base": base, "params": params, "threshold": threshold, "exact": exact,
           "settings": settings, "q2": q2_polynomials(kn, settings)}
    pool = mp.get_context("fork").Pool(workers, initializer=_init, initargs=(ctx,))
    try:
        window = 2 * workers
        pending = {}
        ids = itertools.count(start)
        for _ in range(window):
            i = next(ids)
            pending[i] = pool.apply_async(_task, (i,))
        want = start
        while True:
            res = pending.pop(want).get()
            i = next(ids)
            pending[i] = pool.apply_async(_task, (i,))
            want += 1
            yield res
    finally:
        pool.terminate()
        pool.join()
