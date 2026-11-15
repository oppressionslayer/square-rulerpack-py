import numpy as np
import pytest

from square_rulerpack_py import lanczos, linalg


def random_rows(n, cols, weight, seed):
    rng = np.random.default_rng(seed)
    return [sorted(set(rng.integers(0, cols, size=weight).tolist())) for _ in range(n)]


def check(rows, cols, deps):
    assert deps
    for d in deps:
        acc = np.zeros(cols, dtype=np.uint8)
        for i in d:
            acc[rows[i]] ^= 1
        assert not acc.any(), "a returned set is not a dependency"


@pytest.mark.parametrize("n,cols,seed", [(300, 250, 1), (2000, 1800, 2), (5000, 4700, 3)])
def test_lanczos_dependencies_check(n, cols, seed):
    rows = random_rows(n, cols, 12, seed)
    st = {}
    deps = lanczos.dependencies(rows, cols, seed, stats=st)
    check(rows, cols, deps)
    assert st["iterations"] > 0
    # the dense elimination agrees that dependencies exist here
    assert linalg.dense_dependencies(rows, cols)


def test_seeds_differ_and_numpy_fallback_agrees():
    rows = random_rows(400, 350, 10, 7)
    a = lanczos.dependencies(rows, 350, 1)
    b = lanczos.dependencies(rows, 350, 2)
    check(rows, 350, a)
    check(rows, 350, b)
    assert a != b
    c = lanczos.dependencies(rows, 350, 1, use_numba=False)
    check(rows, 350, c)
    assert c == a, "the numpy fallback is the same arithmetic as the numba kernels"


def test_dispatch():
    rows = random_rows(500, 450, 10, 9)
    st = {}
    linalg.dependencies(rows, 450, stats=st)
    assert st["solver"] == "dense elimination"
    st = {}
    deps = linalg.dependencies(rows, 450, lanczos=True, stats=st)
    assert st["solver"] == "block Lanczos" and deps
    check(rows, 450, deps)
    assert linalg.dependencies([], 10) == []
