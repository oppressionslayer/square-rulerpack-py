"""pytest: the engine against known factorisations and its own invariants."""
import math
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

import square_rulerpack_py as sq
from square_rulerpack_py import arith, linalg
from square_rulerpack_py.engine import (FamilySieve, Settings, choose_a, exact_for, factor_base, multiplier,
                                     q2_polynomials, threshold_bits)

README_N = 1009732533765211          # 11344301 × 89007911, the README example of the Rust crate


def test_arith():
    assert arith.sqrt_mod(2, 7) ** 2 % 7 == 2
    assert arith.legendre(2, 7) == 1 and arith.legendre(3, 7) == -1
    assert arith.is_probable_prime(11344301) and not arith.is_probable_prime(README_N)
    f = arith.pollard_brent(README_N)
    assert f in (11344301, 89007911)
    assert list(arith.primes_upto(30)) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


def test_params_match_rust_tables():
    r = sq.params.for_bits(sq.YAFU_DIRECT, 332)
    assert (r.base, r.lp_mult, r.blocks, r.tf_bits) == (90432, 350, 5, 102)
    assert sq.Params.for_bits(50, 16).base == 150


def test_readme_example():
    assert sq.factor_completely(README_N) == [11344301, 89007911]


def test_small_inputs_and_prime_powers():
    assert sq.factor_completely(2 ** 10 * 3 ** 3) == [2] * 10 + [3] * 3
    assert sq.factor_completely(1000003 ** 2) == [1000003, 1000003]
    assert sq.factor_completely(999999000001) == [999999000001]      # prime


@pytest.mark.parametrize("p,q", [
    (620962231424641, 683905580565331),                               # 30 digits
    (61744934419289236901, 94866163883639224007),                     # 40 digits
])
def test_semiprimes(p, q):
    assert sq.factor_completely(p * q) == sorted([p, q])


def test_relations_verify_and_gray_steps_match():
    n = 61744934419289236901 * 94866163883639224007
    st = Settings()
    k = multiplier(n)
    kn = n * k
    params = sq.Params.for_n(n)
    base = factor_base(kn, params.base)
    q2 = q2_polynomials(kn, st)
    thr = threshold_bits(kn, base, params, q2)
    fam = choose_a(kn, base, params.half, 1, q2)
    a, b = FamilySieve(kn, base, fam, params, thr, exact_for(kn, st), st), FamilySieve(kn, base, fam, params, thr, exact_for(kn, st), st)
    a.set_poly(0)
    b.set_poly(0)
    for t in range(1, fam.polys):
        a.advance(t)
        b.set_poly(t)
        assert a.b == b.b and np.array_equal(a.starts0, b.starts0) and np.array_equal(a.starts1, b.starts1)
        r1, _ = a.roots()
        ok = fam.ainv != 0
        assert all(a.g(int(x)) % int(p) == 0 for x, p in zip(r1[ok], base.p[ok]))
    rels, polys = FamilySieve(kn, base, fam, params, thr, exact_for(kn, st), st).run()
    assert polys == fam.polys and rels
    assert all(r.check(kn, base) for r in rels)


def test_exact_threshold_relations_verify():
    p, q = 844729315367594426787762868041547, 850126934801873397224646120901021   # 66 digits: exact threshold
    n = p * q
    st = Settings()
    k = multiplier(n)
    kn = n * k
    params = sq.Params.for_n(n)
    base = factor_base(kn, params.base)
    assert exact_for(kn, st)
    fam = choose_a(kn, base, params.half, 1, q2_polynomials(kn, st))
    fs = FamilySieve(kn, base, fam, params, threshold_bits(kn, base, params, fam.q2), True, st)
    rels, _ = fs.run(max_polys=8)
    assert rels and all(r.check(kn, base) for r in rels)


def test_linalg_dependencies_are_dependencies():
    rng = np.random.default_rng(1)
    cols = 40
    rows = [sorted(set(rng.integers(0, cols, size=5).tolist())) for _ in range(60)]
    deps = linalg.dependencies(rows, cols)
    assert deps
    for d in deps:
        acc = np.zeros(cols, dtype=int)
        for i in d:
            acc[rows[i]] ^= 1
        assert not acc.any()


def test_walkthrough_end_to_end():
    w = sq.Walkthrough(README_N, workers=1, verbose=False)
    assert w.multiplier().chosen.any()
    assert len(w.factor_base()) == 150
    w.family()
    s = w.sieve(0)
    td = w.trial_divide(0)
    assert td["y² ≡ product (mod kn)"].dropna().all()
    w.collect()
    w.matrix(dense=False)
    w.dependencies()
    assert sorted(w.factors()) == [11344301, 89007911]


def test_workers_give_the_same_relation_set():
    n = 4837594595591756100052399 * 6679717452512069423002243
    st = Settings()
    k = multiplier(n)
    kn = n * k
    params = sq.Params.for_n(n)
    base = factor_base(kn, params.base)
    q2 = q2_polynomials(kn, st)
    thr = threshold_bits(kn, base, params, q2)
    from square_rulerpack_py.workers import family_stream
    from square_rulerpack_py.engine import relations
    need = len(base) + 1 + params.extra
    one = relations(kn, base, params, need, family_stream(kn, base, params, thr, False, st, 1), family_limit=6)
    two = relations(kn, base, params, need, family_stream(kn, base, params, thr, False, st, 3), family_limit=6)
    assert [r.y for r in one.full] == [r.y for r in two.full]
    assert one.polys == two.polys
