import numpy as np
from square_rulerpack_py import xor_table as xt


def test_roundtrip_and_diagonal():
    pr = xt.first_primes(26)
    idx = 2 * pr
    t = xt.buildsystem(idx)
    assert (t == t.T).all() and not np.diag(t).any()
    assert (xt.recover_index(t) == idx).all()
    sq = xt.odd_squares(27)
    m = xt.table_with_ruler(idx, sq)
    assert (np.diag(m) == sq).all()
    for row in m:
        assert (xt.decode_row(row, sq) == idx).all()
    assert (m[:, 0] == xt.anchor_walk(idx)).all()


def test_the_walk():
    pr = xt.first_primes(26)
    w = xt.anchor_walk(2 * pr)
    assert list(w[:9]) == [1, 5, 3, 9, 7, 17, 11, 41, 15]
    assert w[7] == w[17] == 41
    assert int(np.bitwise_xor.reduce(pr[7:17])) == 0          # 19 ^ 23 ^ ... ^ 59 == 0
    assert int(w.max()) < 256 and int((2 * pr).sum()) > 2000


def test_lattice_is_the_sieve_polynomial():
    assert (xt.lattice(0, 0, 5) == [1, 25, 81, 169, 289]).all()
    lpp = xt.lattice(7)
    assert set(int(v) % 8 for v in lpp) == {2}
    pr = xt.first_primes(50)
    fb = set(map(int, xt.lattice_divisors(7, pr)))
    seen = set()
    for v in map(int, lpp):
        v = abs(v)
        for p in map(int, pr):
            if v % p == 0:
                seen.add(p)
    assert seen <= fb
    assert 3 in fb and 19 in fb and 5 not in fb and 11 not in fb
