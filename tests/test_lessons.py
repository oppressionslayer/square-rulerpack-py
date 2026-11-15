"""Exact checks of the lesson module, carried over from square_research's own test suite."""
from itertools import product
from math import isqrt, prod

import pytest

import square_rulerpack_py as sq
from square_rulerpack_py import lessons as L


def test_crt_all_root_choices_and_lifts():
    for roots in product(L.sqrt_mod(221, 5), L.sqrt_mod(221, 7)):
        p = sq.construct_polynomial(221, [5, 7], roots=iter(roots))
        assert [p["b"] % q for q in [5, 7]] == list(roots)
        assert sum(t["term"] for t in p["crt_terms"]) % 35 == p["b"]
        shifted = sq.construct_polynomial(221, [5, 7], roots=roots, b=p["b"] - 70)
        for x in [-20, -1, 0, 1, 20]:
            c, b, a = shifted["coefficients"]
            assert (shifted["root_scale"] * x + shifted["b"]) ** 2 - 221 == shifted["divisor"] * (a * x * x + b * x + c)
    for kwargs in [dict(a_primes=[5, 5]), dict(a_primes=[2]), dict(a_primes=[13]), dict(a_primes=[3]),
                   dict(a_primes=[5, 7], roots=[1]), dict(a_primes=[5, 7], roots=[1, 2, 3]),
                   dict(a_primes=[5, 7], roots=[0, 2]), dict(a_primes=[5, 7], b=17), dict(a_primes=[5], form="q2")]:
        with pytest.raises(ValueError):
            sq.construct_polynomial(221, **kwargs)


def test_default_lesson_n_221():
    lesson = sq.factor_walkthrough()
    assert lesson.split == (13, 17)
    assert (lesson.rank, lesson.nullity) == (2, 1)
    proof = lesson.check_dependency([1, 2])
    assert (proof["offset_product"], proof["y"], proof["root_product"]) == (4900, 70, -304)
    assert proof["exponent_totals"] == (0, 2, 2, 2, 0)
    assert lesson.relations[2]["parity"] == (0, 0, 1, 1, 0)      # Q(0) = 1 still carries the primes of a
    with pytest.raises(ValueError):
        lesson.check_dependency([2])
    for ids in [[], [1, 1], [True]]:
        with pytest.raises(ValueError):
            lesson.check_dependency(ids)
    with pytest.raises(IndexError):
        lesson.check_dependency([-1])
    assert list(lesson.matrix().columns) == ["-1", "2", "5", "7", "11"]
    assert len(lesson.dataframe("samples")) == 5


def test_q2_negative_sign_and_restored_four_a():
    p = sq.construct_polynomial(209, [5], roots=[3], form="q2")
    assert (p["b"], p["coefficients"], p["divisor"]) == (3, (-10, 3, 5), 20)
    w = sq.siqs_window(209, p["a"], p["b"], form="q2", start=-2, width=5, seeds=[2, 5])
    lesson = sq.factor_walkthrough(w)
    assert lesson.split == (11, 19)
    assert lesson.check_dependency([1, 3])["offset_product"] == 6400
    assert lesson.relations[1]["exponents"][0] == 1
    assert lesson.relations[0]["exponents"] == (0, 4, 1)
    assert sq.construct_polynomial(209, [5], roots=[2], form="q2")["b"] == 7   # even CRT value lifted by a


def test_basis_matches_exhaustive_square_products():
    for n in range(2, 50):
        w = sq.square_window(n, start_root=-3, width=7, seeds=[2, 3, 5, 7])
        lesson = sq.factor_walkthrough(w)
        oracle = {0}
        for mask in range(1, 1 << len(lesson.relations)):
            value = prod(r["full_offset"] for i, r in enumerate(lesson.relations) if mask >> i & 1)
            if value > 0 and isqrt(value) ** 2 == value:
                oracle.add(mask)
        span = {0}
        for mask in lesson.dependency_masks:
            span |= {x ^ mask for x in tuple(span)}
        assert span == oracle, n
        assert 1 << lesson.nullity == len(oracle), n
        if lesson.split:
            assert prod(lesson.split) == n and all(1 < d < n for d in lesson.split)


def test_multipliers_and_statuses():
    w = sq.square_window(385, start_root=20, width=20, seeds=sq.prime_seeds(31))
    assert sq.factor_walkthrough(w, original_n=77).split == (7, 11)       # gcds against N = 77, sieving 5·N
    for original in [1, 13, True]:
        with pytest.raises(ValueError):
            sq.factor_walkthrough(w, original_n=original)
    with pytest.raises(ValueError):
        sq.factor_walkthrough([w, sq.square_window(221, width=2, seeds=sq.prime_seeds(31))])
    prime = sq.square_window(13, start_root=3, width=8, seeds=sq.prime_seeds(19))
    lesson = sq.factor_walkthrough(prime)
    assert lesson.nullity > 0 and lesson.status == "no_split_found" and lesson.split is None


def test_sieve_walk_and_roots_against_oracles():
    for p in sq.prime_seeds(47):
        for n in range(p):
            assert L.sqrt_mod(n, p) == tuple(x for x in range(p) if x * x % p == n)
        for c, b, a in [(0, 0, 0), (1, 0, 0), (3, 7, 0), (-221, 0, 1), (-10, 3, 5), (5, -4, 7)]:
            assert sq.polynomial_roots((c, b, a), p) == tuple(x for x in range(p) if (a * x * x + b * x + c) % p == 0)
    for step in [-2, 1, 3]:
        w = L.polynomial_window((-10, 3, 5), -17, 64, sq.prime_seeds(29), step=step)
        assert all(r["verified"] for r in sq.sieve_walk(w))
    w = L.polynomial_window((350, 1), 0, 64, sq.prime_seeds(29))
    assert list(L._positions(L.survivor_mask(w))) == [i for i in range(64) if all((350 + i) % p for p in sq.prime_seeds(29))]


def test_relation_study_default():
    study = sq.relation_study()
    assert study.pooled.split == (13, 17) and study.mirror_verified
    assert len(study.polynomials) == 4 and len(study.rows) == 12
    assert "Outcome: split_found (13, 17)" in study.report()


def test_engine_selector_then_lesson_pipeline():
    cap = sq.engine_polynomials(1009732533765211, base=64, half=128, families=8, per_family=4)
    assert (cap.data["k"], cap.data["form"]) == (1, "q1")
    for row in cap.records():
        assert row["identity_valid"] and len(row["a_primes"]) == len(set(row["a_primes"]))
        rebuilt = sq.construct_polynomial(cap.kn, row["a_primes"], roots=(row["b"] % p for p in row["a_primes"]),
                                          b=row["b"], form=cap.data["form"])
        assert rebuilt["coefficients"] == row["coefficients"]
    windows = [cap.window(i, start=-128, width=257) for i in range(len(cap.records()))]
    lesson = sq.factor_walkthrough(windows, original_n=cap.n)
    assert lesson.split == (11344301, 89007911)
    assert all(all(e % 2 == 0 for e in a["exponent_totals"]) for a in lesson.attempts)
