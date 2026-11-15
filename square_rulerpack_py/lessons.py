"""Small, exact lessons: CRT polynomial construction, a few samples, parity, squares, a factor pair.

Carried over from Lars Rocha's ``square_research`` package (MIT) and wired to this engine
instead of the Rust subprocess bridge. Everything here uses plain Python integers over a handful
of samples, so a class can follow each number by hand:

    >>> from square_rulerpack_py import lessons as L
    >>> p = L.construct_polynomial(221, [5, 7])          # every CRT term of b, then a, b, c
    >>> w = L.siqs_window(p['k'], p['a'], p['b'], start=-2, width=5, seeds=[2, 5, 7, 11])
    >>> w.dataframe()                                    # x, root, value, factors, smooth ...
    >>> lesson = L.factor_walkthrough(w)                 # parity matrix -> dependency -> gcd
    >>> lesson.split                                     # (13, 17)

``engine_polynomials`` asks this package's real selector (``choose_a``, the same families as
the Rust engine) for a few polynomials and turns them into such windows, so the lesson can be
run on polynomials the production sieve would actually use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
from math import gcd, isqrt, prod
from operator import index
from pathlib import Path
from statistics import median
import json

from .arith import primes_upto, sqrt_mod as _sqrt_mod, legendre as _legendre, is_probable_prime
from .engine import Settings, choose_a, factor_base, multiplier as _multiplier, q2_polynomials
from .params import Params

__all__ = ["prime_seeds", "legendre", "sqrt_mod", "polynomial_roots", "LessonWindow",
           "polynomial_window", "square_window", "siqs_window", "sieve_walk", "survivor_mask",
           "construct_polynomial", "FactorWalkthrough", "factor_walkthrough",
           "RelationStudy", "relation_study", "Capture", "engine_polynomials", "dependency"]


# ----------------------------------------------------------------------------- small helpers

def _int(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer") from None


def _frame(rows):
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Tables need pandas; use .records() for plain Python data.") from None
    return pd.DataFrame(rows, dtype=object)


def _prime(p):
    p = _int(p, "prime")
    if not 2 <= p <= 100000 or any(p % d == 0 for d in range(2, isqrt(p) + 1)):
        raise ValueError("Use a prime from 2 through 100000 for these small modular experiments")
    return p


def prime_seeds(bound=97):
    """The primes up to ``bound`` (2..100000), the default smoothness base of a lesson."""
    bound = _int(bound, "bound")
    if not 2 <= bound <= 100000:
        raise ValueError("bound must be 2..100000")
    return [int(p) for p in primes_upto(bound)]


def legendre(n, p):
    """(n/p) for an odd prime p: 1, -1 or 0."""
    p = _prime(p)
    if p == 2:
        raise ValueError("The Legendre symbol uses an odd prime")
    return _legendre(_int(n, "n") % p, p)


def sqrt_mod(n, p):
    """All square roots of n modulo the prime p, as a sorted tuple (empty if none)."""
    p = _prime(p)
    n = _int(n, "n") % p
    if p == 2 or n == 0:
        return (n,)
    if pow(n, (p - 1) // 2, p) != 1:
        return ()
    r = _sqrt_mod(n, p)
    return tuple(sorted({r, (-r) % p}))


def polynomial_roots(coefficients, p):
    """Roots modulo a prime of a polynomial of degree at most two, constant first."""
    p = _prime(p)
    coefficients = tuple(_int(c, "coefficient") for c in coefficients)
    if not 1 <= len(coefficients) <= 3:
        raise ValueError("Use one, two or three coefficients, constant first")
    c, b, a = (coefficients + (0, 0))[:3]
    a, b, c = a % p, b % p, c % p
    if p == 2:
        return tuple(x for x in range(2) if (a * x * x + b * x + c) % 2 == 0)
    if a == 0:
        if b:
            return ((-c * pow(b, -1, p)) % p,)
        return tuple(range(p)) if c == 0 else ()
    inverse = pow(2 * a, -1, p)
    return tuple(sorted({(-b + r) * inverse % p for r in sqrt_mod(b * b - 4 * a * c, p)}))


def _divide(value, primes):
    remaining = abs(value)
    exponents = []
    for p in primes:
        e = 0
        while remaining and remaining % p == 0:
            remaining //= p
            e += 1
        exponents.append(e)
    return tuple(exponents), remaining


def _expression(value, primes, exponents):
    parts = ["-1"] if value < 0 else []
    parts.extend(str(p) if e == 1 else f"{p}^{e}" for p, e in zip(primes, exponents) if e)
    return f"{value} = " + (" * ".join(parts) or "1")


def _positions(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


# ----------------------------------------------------------------------------- windows

@dataclass(frozen=True)
class LessonWindow:
    """A few exact samples of Q(x) = c + lin·x + a·x², with the square witness attached.

    ``coefficients`` is constant first. Sample k sits at x = start + step·k. With a witness,
    root = root_scale·x + root_shift and root² − n = divisor · Q(x); ``full_offset`` is that
    root² − n, the number whose factorisation the lesson actually uses (it includes the primes
    of a, or of 4a for Q2). ``seeds`` is the smoothness base; ``smooth`` means Q(x) factors
    completely over it (units count as smooth, zero is undefined).
    """
    coefficients: tuple
    start: int
    width: int
    step: int
    seeds: tuple
    n: int | None = None
    root_scale: int | None = None
    root_shift: int = 0
    divisor: int = 1
    form: str = "general"
    label: str = "polynomial"
    provenance: dict = field(default_factory=dict, compare=False)
    _rows: tuple = field(default=(), repr=False, compare=False)

    def __post_init__(self):
        rows = []
        divisions = 0
        for k in range(self.width):
            x = self.start + self.step * k
            c, b, a = (tuple(self.coefficients) + (0, 0))[:3]
            value = c + b * x + a * x * x
            root = None if self.root_scale is None else self.root_scale * x + self.root_shift
            full = None if root is None else root * root - self.n
            if full is not None and full != self.divisor * value:
                raise ArithmeticError("polynomial / square identity failed")
            exps, cofactor = _divide(value, self.seeds) if value else ((0,) * len(self.seeds), 0)
            divisions += len(self.seeds)
            factors = tuple((p, e) for p, e in zip(self.seeds, exps) if e)
            defined = value != 0
            smooth = defined and cofactor == 1
            rows.append(dict(position=k, x=x, value=value, sign=(value > 0) - (value < 0),
                             factors=factors, cofactor=cofactor if defined else None, smooth=smooth,
                             defined=defined,
                             factorization=_expression(value, self.seeds, exps) if defined else "0",
                             parity=tuple(e % 2 for e in exps), root=root,
                             square=None if root is None else root * root, full_offset=full))
        object.__setattr__(self, "_rows", tuple(rows))
        object.__setattr__(self, "_stats", dict(root_evaluations=self.width, division_checks=divisions))

    # -- access
    @property
    def values(self):
        return [r["value"] for r in self._rows]

    def value(self, position=0):
        return self._rows[position]["value"]

    def root(self, position=0):
        return self._rows[position]["root"]

    def records(self):
        out = []
        for r in self._rows:
            row = dict(r)
            row.update({f"e_{p}": dict(r["factors"]).get(p, 0) for p in self.seeds})
            out.append(row)
        return out

    def dataframe(self):
        return _frame(self.records())

    def divisibility_mask(self, p):
        return sum(1 << r["position"] for r in self._rows if r["value"] and r["value"] % p == 0)

    def smooth_mask(self, *, include_units=True):
        return sum(1 << r["position"] for r in self._rows
                   if r["smooth"] and (include_units or abs(r["value"]) > 1))

    def summary(self):
        a = self._rows
        defined = [r for r in a if r["defined"]]
        return dict(label=self.label, width=self.width, start=self.start, step=self.step,
                    coefficients=tuple(self.coefficients), seeds=self.seeds, target_divisor=self.divisor,
                    polynomial_form=self.form, n_sieved=self.n, defined=len(defined),
                    smooth=sum(r["smooth"] for r in a),
                    nonunit_smooth=sum(r["smooth"] and abs(r["value"]) > 1 for r in a),
                    unique_nonunit_smooth=len({r["value"] for r in a if r["smooth"] and abs(r["value"]) > 1}),
                    median_value_bits=median([abs(r["value"]).bit_length() for r in a]) if a else None,
                    median_cofactor_bits=median([r["cofactor"].bit_length() for r in defined]) if defined else None,
                    **self._stats)

    def dependency(self, mask):
        """Is the product of the chosen full offsets a square? Returns (offset_product, y) or None."""
        if self.root_scale is None:
            raise ValueError("use a square or SIQS window with attached witnesses")
        positions = list(_positions(_int(mask, "mask")))
        if not 1 <= len(positions) <= 256 or positions[-1] >= self.width:
            raise ValueError("select 1..256 positions inside the window")
        offsets = [self._rows[i]["full_offset"] for i in positions]
        if any(o == 0 for o in offsets):
            raise ValueError("zero offsets do not have finite prime factorizations")
        value = prod(offsets)
        if value < 0 or isqrt(value) ** 2 != value:
            return None
        return value, isqrt(value)

    def export(self, path):
        """Write a new JSON file (integers as decimal strings); refuse to overwrite."""
        path = Path(path).expanduser()
        meta = dict(label=self.label, coefficients=[str(c) for c in self.coefficients], start=str(self.start),
                    width=self.width, step=str(self.step), seeds=[str(p) for p in self.seeds],
                    n=None if self.n is None else str(self.n),
                    root_scale=None if self.root_scale is None else str(self.root_scale),
                    root_shift=str(self.root_shift), divisor=str(self.divisor), form=self.form,
                    provenance=self.provenance)
        rows = [{k: (str(v) if isinstance(v, int) and not isinstance(v, bool) else
                     json.dumps(v) if isinstance(v, tuple) else v) for k, v in r.items()}
                for r in self.records()]
        with path.open("x", encoding="utf-8") as f:
            json.dump(dict(schema="square-rulerpack.lesson-window.v1", metadata=meta, rows=rows), f, indent=2)
            f.write("\n")
        return path


def _base(seeds):
    seeds = tuple(dict.fromkeys(_int(p, "seed") for p in (prime_seeds(97) if seeds is None else seeds)))
    if len(seeds) > 128 or any(not 2 <= p <= 100000 for p in seeds):
        raise ValueError("use at most 128 seeds in 2..100000")
    for p in seeds:
        _prime(p)
    return seeds


def polynomial_window(coefficients, start, width=26, seeds=None, *, step=1, label="polynomial", **extra):
    """Sample c + lin·x + a·x² (constant first) at start + step·k for 0 ≤ k < width."""
    coefficients = tuple(_int(c, "coefficient") for c in coefficients)
    if not 1 <= len(coefficients) <= 3:
        raise ValueError("one to three coefficients, constant first")
    width = _int(width, "width")
    if not 0 <= width <= 10000:
        raise ValueError("width must be 0..10000")
    return LessonWindow(coefficients, _int(start, "start"), width, _int(step, "step"), _base(seeds),
                        label=label, **extra)


def square_window(n, start_root=None, width=26, seeds=None, *, step=1):
    """Q(x) = x² − n with the witness x itself; the default start is ⌈√n⌉."""
    n = _int(n, "n")
    if n < 0:
        raise ValueError("n must be nonnegative")
    if start_root is None:
        start_root = isqrt(n)
        start_root += start_root * start_root < n
    return polynomial_window((-n, 0, 1), start_root, width, seeds, step=step, label="square-root window",
                             n=n, root_scale=1, form="square")


def siqs_window(n, a, b, start=0, width=26, seeds=None, *, form="q1", step=1):
    """Q1: (a·x + b)² − n = a·Q(x).  Q2: (2a·x + b)² − n = 4a·Q(x).  ``n`` is the sieved k·N."""
    n, a, b = _int(n, "n"), _int(a, "a"), _int(b, "b")
    if form not in ("q1", "q2"):
        raise ValueError("form must be q1 or q2")
    if a <= 0 or n <= 0:
        raise ValueError("n and a must be positive")
    if form == "q1":
        if (b * b - n) % a:
            raise ValueError("Q1 needs a dividing b² − n")
        coefficients, scale, divisor = ((b * b - n) // a, 2 * b, a), a, a
    else:
        if n % 8 != 1 or b % 2 != 1 or (b * b - n) % (4 * a):
            raise ValueError("Q2 needs n ≡ 1 (mod 8), odd b, and 4a dividing b² − n")
        coefficients, scale, divisor = ((b * b - n) // (4 * a), b, a), 2 * a, 4 * a
    return polynomial_window(coefficients, start, width, seeds, step=step, label=f"SIQS {form.upper()} polynomial",
                             n=n, root_scale=scale, root_shift=b, divisor=divisor, form=form.upper())


def sieve_walk(window):
    """For every seed prime: the roots of Q modulo p, the positions they predict, and a check that
    those positions are exactly where p divides the sampled value. This is the sieve, done by hand."""
    c, b, a = (tuple(window.coefficients) + (0, 0))[:3]
    s, d = window.start, window.step
    shifted = (c + b * s + a * s * s, d * (b + 2 * a * s), a * d * d)        # Q(start + step·k) in k
    rows = []
    for p in window.seeds:
        roots = polynomial_roots(shifted, p)
        predicted = [k for k in range(window.width) if k % p in roots]
        actual = [k for k, v in enumerate(window.values) if v % p == 0]
        if predicted != actual:
            raise ArithmeticError("Modular roots disagree with direct divisibility")
        rows.append(dict(prime=p, coordinate_roots=polynomial_roots(window.coefficients, p),
                         position_roots=roots, hits=len(predicted), positions=tuple(predicted),
                         prime_power_hits=sum(v != 0 and v % (p * p) == 0 for v in window.values),
                         mask=sum(1 << k for k in predicted), verified=True))
    return rows


def survivor_mask(window):
    """Positions no seed prime divides (the complement of all the hit masks)."""
    hit = 0
    for p in window.seeds:
        hit |= window.divisibility_mask(p)
    return ((1 << window.width) - 1) ^ hit


def dependency(window, mask, *, original_n=None):
    """Check that the chosen full offsets multiply to a square and take the two gcds with N."""
    result = window.dependency(mask)
    if result is None:
        return None
    n = window.n if original_n is None else _int(original_n, "original_n")
    if n < 2 or window.n % n:
        raise ValueError("original_n must be at least 2 and divide the sieved K")
    offset_product, y = result
    x = prod(window.root(i) for i in _positions(mask)) % n
    return dict(x=x, y=y, gcd_minus=gcd(x - y, n), gcd_plus=gcd(x + y, n), modulus=n, sieved_k=window.n,
                offset_product=offset_product)


# ----------------------------------------------------------------------------- CRT polynomial

def construct_polynomial(k, a_primes, *, roots=None, form="q1", b=None):
    """Build a Q1/Q2 polynomial from distinct odd primes and show every CRT term of b.

    ``k`` is the sieved integer (with any multiplier). For each prime choose a square root of k;
    the default is the smaller one. An explicit ``b`` keeps the exact signed b of a captured
    engine polynomial after checking its congruences. Coefficients come back constant first.
    """
    k = _int(k, "k")
    if not 2 <= k or k.bit_length() > 1024:
        raise ValueError("Use K >= 2 with at most 1024 bits for this lesson")
    primes = tuple(_prime(p) for p in a_primes)
    if not 1 <= len(primes) <= 20 or len(set(primes)) != len(primes):
        raise ValueError("Use 1..20 distinct A primes")
    if any(p == 2 or gcd(k, p) != 1 for p in primes):
        raise ValueError("A primes must be odd and coprime to K")
    if form not in ("q1", "q2") or (form == "q2" and k % 8 != 1):
        raise ValueError("Use q1, or q2 with K congruent to 1 modulo 8")
    available = [sqrt_mod(k, p) for p in primes]
    if any(not choices for choices in available):
        raise ValueError("K must have a square root modulo each A prime")
    if roots is None:
        chosen = tuple(options[0] for options in available)
    else:
        roots = tuple(roots)
        if len(roots) != len(primes):
            raise ValueError("Supply exactly one root for each A prime")
        chosen = tuple(_int(r, "root") % p for r, p in zip(roots, primes))
    if any(r not in options for r, options in zip(chosen, available)):
        raise ValueError("A chosen root does not square to K modulo its prime")
    a = prod(primes)
    terms = []
    for p, r, options in zip(primes, chosen, available):
        other = a // p
        inverse = pow(other, -1, p)
        coefficient = r * inverse % p
        terms.append(dict(prime=p, roots=options, chosen_root=r, a_over_p=other, inverse=inverse,
                          coefficient=coefficient, term=other * coefficient))
    crt_b = sum(row["term"] for row in terms) % a
    if b is None:
        b = crt_b
        if form == "q2" and b % 2 == 0:
            b += a
    else:
        b = _int(b, "b")
    divisor = a * (4 if form == "q2" else 1)
    if b % a != crt_b or (b * b - k) % divisor:
        raise ValueError("B does not satisfy the selected roots and polynomial identity")
    c = (b * b - k) // divisor
    return dict(k=k, a=a, b=b, c=c, form=form, a_primes=primes, root_scale=a * (2 if form == "q2" else 1),
                divisor=divisor, coefficients=(c, b if form == "q2" else 2 * b, a), crt_terms=terms,
                crt_b=crt_b, b_lift=(b - crt_b) // a, identity_valid=True)


# ----------------------------------------------------------------------------- the lesson

@dataclass(repr=False)
class FactorWalkthrough:
    """Samples → smooth relations → parity matrix → dependencies → gcd attempts, all kept.

    Matrix rows are relations; columns are (−1, *seeds). Dependency masks select relation rows.
    ``split`` is an exact divisor pair, not a primality certificate.
    """
    n: int
    k: int
    seeds: tuple
    samples: list
    relations: list
    elimination: list
    dependency_masks: list
    attempts: list
    rank: int

    def __repr__(self):
        return (f"FactorWalkthrough(n={self.n}, samples={len(self.samples)}, relations={len(self.relations)}, "
                f"rank={self.rank}, nullity={self.nullity}, split={self.split})")

    @property
    def columns(self):
        return (-1, *self.seeds)

    @property
    def nullity(self):
        return len(self.relations) - self.rank

    @property
    def split(self):
        return next((a["factors"] for a in self.attempts if a["factors"]), None)

    @property
    def status(self):
        if self.split:
            return "split_found"
        return "no_split_found" if self.dependency_masks else "no_dependency"

    def records(self, table="relations"):
        if table not in ("samples", "relations", "elimination", "attempts"):
            raise ValueError("Choose samples, relations, elimination or attempts")
        return [dict(row) for row in getattr(self, table)]

    def dataframe(self, table="relations"):
        """The chosen table as a DataFrame (object dtype keeps the big integers exact)."""
        return _frame(self.records(table))

    def matrix(self):
        """The parity matrix: one row per relation, columns (−1, *seeds), entries 0/1."""
        try:
            import pandas as pd
        except ImportError:
            return [list(r["parity"]) for r in self.relations]
        return pd.DataFrame([list(r["parity"]) for r in self.relations],
                            columns=[str(c) for c in self.columns],
                            index=[r["relation_id"] for r in self.relations])

    def matrix_summary(self):
        distinct = len({row["full_offset"] for row in self.relations})
        return dict(smooth_rows=len(self.relations), distinct_offsets=distinct,
                    duplicate_rows=len(self.relations) - distinct, rank=self.rank, nullity=self.nullity,
                    unique_nullity=distinct - self.rank)

    def check_dependency(self, relation_ids):
        """Reconstruct a chosen set of relations and show its square, x, y and the two gcds."""
        ids = tuple(_int(i, "relation index") for i in relation_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Select a nonempty set of distinct relation indices")
        if any(i < 0 or i >= len(self.relations) for i in ids):
            raise IndexError("relation index out of range")
        rows = [self.relations[i] for i in ids]
        totals = tuple(sum(row["exponents"][j] for row in rows) for j in range(len(self.columns)))
        if any(e % 2 for e in totals):
            raise ValueError("The selected full offsets do not have all-even exponents")
        product = prod(row["full_offset"] for row in rows)
        y = prod(p ** (e // 2) for p, e in zip(self.seeds, totals[1:]))
        root_product = prod(row["root"] for row in rows)
        x = root_product % self.n
        if y * y != product or (x * x - y * y) % self.n:
            raise ArithmeticError("Square reconstruction or congruence failed")
        minus, plus = gcd(x - y, self.n), gcd(x + y, self.n)
        factor = next((d for d in (minus, plus) if 1 < d < self.n), None)
        factors = tuple(sorted((factor, self.n // factor))) if factor else None
        return dict(relation_ids=ids, exponent_totals=totals, offset_product=product, root_product=root_product,
                    x=x, y=y, y_mod_n=y % self.n, gcd_minus=minus, gcd_plus=plus, square_verified=True,
                    congruence_verified=True, factors=factors)


def factor_walkthrough(windows=None, *, original_n=None, max_dependencies=64):
    """Sieve samples → smooth relations → parity → squares → an exact factor pair.

    With no windows, run the tiny N = 221 lesson (Q1 from A primes 5 and 7). Otherwise pass one
    square/SIQS window or a list sharing K and ordered seeds. Every nullspace basis vector is
    found; up to ``max_dependencies`` of them (then pairwise XORs) are tried for a split.
    Bounds: 64 windows, 16384 samples, 512 smooth relations.
    """
    max_dependencies = _int(max_dependencies, "max_dependencies")
    if not 1 <= max_dependencies <= 1024:
        raise ValueError("Use 1..1024 dependency attempts")
    if windows is None:
        p = construct_polynomial(221, [5, 7])
        windows = siqs_window(p["k"], p["a"], p["b"], start=-2, width=5, seeds=[2, 5, 7, 11])
    windows = [windows] if isinstance(windows, LessonWindow) else list(windows)
    if not 1 <= len(windows) <= 64 or sum(w.width for w in windows) > 16384:
        raise ValueError("Use 1..64 windows and at most 16384 samples")
    first = windows[0]
    k, seeds = first.n, tuple(first.seeds)
    if k is None or k < 2:
        raise ValueError("Windows need square witnesses with K >= 2")
    n = k if original_n is None else _int(original_n, "original_n")
    if n < 2 or k % n:
        raise ValueError("original_n must be at least 2 and divide the sieved K")
    samples, relations = [], []
    for wi, w in enumerate(windows):
        if w.n != k or tuple(w.seeds) != seeds or w.root_scale is None:
            raise ValueError("Windows must share K and ordered seeds, with square witnesses")
        removed, remainder = _divide(w.divisor, seeds)
        if remainder != 1:
            raise ValueError("Include every prime factor of the removed divisor in seeds")
        for row in w.records():
            if row["root"] ** 2 - k != w.divisor * row["value"]:
                raise ArithmeticError("A sampled polynomial identity failed")
            sample = dict(row, window=wi, divisor=w.divisor, relation_id=None)
            if row["defined"] and row["smooth"]:
                factors = dict(row["factors"])
                powers = tuple(factors.get(p, 0) + e for p, e in zip(seeds, removed))
                exponents = (int(row["value"] < 0), *powers)
                parity = tuple(e % 2 for e in exponents)
                full = row["full_offset"]
                if (-1 if exponents[0] else 1) * prod(p ** e for p, e in zip(seeds, powers)) != full:
                    raise ArithmeticError("Full-offset factorization failed")
                ri = len(relations)
                if ri >= 512:
                    raise ValueError("At most 512 smooth relations per lesson; use fewer samples")
                sample["relation_id"] = ri
                relations.append(dict(relation_id=ri, window=wi, position=row["position"], x=row["x"],
                                      root=row["root"], value=row["value"], divisor=w.divisor, full_offset=full,
                                      full_factorization=_expression(full, seeds, powers), exponents=exponents,
                                      parity=parity, parity_mask=sum(bit << j for j, bit in enumerate(parity))))
            samples.append(sample)
    pivots, basis, elimination = {}, [], []
    for ri, row in enumerate(relations):
        reduced, mask, used = row["parity_mask"], 1 << ri, []
        while reduced:
            pivot = reduced.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = (reduced, mask)
                break
            value, source = pivots[pivot]
            reduced ^= value
            mask ^= source
            used.append(pivot)
        if reduced == 0:
            basis.append(mask)
        elimination.append(dict(relation_id=ri, original=row["parity_mask"], pivot_columns=tuple(used),
                                reduced=reduced, combination=tuple(_positions(mask)), dependency=(reduced == 0)))
    lesson = FactorWalkthrough(n, k, seeds, samples, relations, elimination, basis, [], len(pivots))

    def choices():
        yield from basis
        for left, right in combinations(basis, 2):
            yield left ^ right

    for attempt, mask in enumerate(choices()):
        if attempt == max_dependencies:
            break
        result = lesson.check_dependency(_positions(mask))
        lesson.attempts.append(result)
        if result["factors"]:
            break
    return lesson


# ----------------------------------------------------------------------------- relation study

@dataclass(repr=False)
class RelationStudy:
    """Every CRT root choice of one a, sampled in left/centre/right windows, with the lessons."""
    n: int
    k: int
    seeds: tuple
    polynomials: tuple
    windows: dict
    lessons: dict
    rows: list
    pooled: FactorWalkthrough
    mirror_verified: bool

    def __repr__(self):
        stats = self.pooled.matrix_summary()
        return (f"RelationStudy(n={self.n}, polynomials={len(self.polynomials)}, windows={len(self.windows)}, "
                f"center_relations={stats['smooth_rows']}, distinct_offsets={stats['distinct_offsets']}, "
                f"split={self.pooled.split})")

    @property
    def proof(self):
        return next((p for p in self.pooled.attempts if p["factors"]), None)

    def window(self, b=None, region="center"):
        b = self.polynomials[0]["b"] if b is None else _int(b, "b")
        return self.windows[(b, region)]

    def lesson(self, b=None, region="center"):
        b = self.polynomials[0]["b"] if b is None else _int(b, "b")
        return self.lessons[(b, region)]

    def records(self):
        return [dict(row) for row in self.rows]

    def dataframe(self):
        return _frame(self.records())

    def report(self):
        first = self.polynomials[0]
        lines = [f"N={self.n}; K={self.k}; A={first['a']}; {len(self.seeds)} seed primes.",
                 f"Each window has {self.window().width} samples. Units count as smooth; zero is excluded.",
                 f"{'B':>8} {'region':>7} {'Q bits*':>8} {'smooth':>7} {'distinct':>9} {'rank':>5} {'nullity':>8} {'split':>24}"]
        for r in self.rows:
            lines.append(f"{r['b']:8} {r['region']:>7} {str(r['median_q_bits']):>8} {r['smooth_rows']:7} "
                         f"{r['distinct_offsets']:9} {r['rank']:5} {r['nullity']:8} {str(r['split']):>24}")
        lines += ["* Median bit length of |Q|; not a timing measurement.", "",
                  "Combined centers: " + str(self.pooled.matrix_summary()),
                  f"Outcome: {self.pooled.status} {self.pooled.split}",
                  "Mirror identity verified for every CRT choice."]
        return "\n".join(lines)


def relation_study(n=221, *, a_primes=(5, 7), seeds=(2, 5, 7, 11), start=-2, width=5, multiplier=1, form="q1"):
    """Compare every CRT root choice of a in left, centre and right windows; pool the centres.

    The default reproduces the N = 221 experiment. K = multiplier·N. Bounds: four A primes,
    16384 samples in all, 512 pooled smooth relations.
    """
    n, multiplier = _int(n, "n"), _int(multiplier, "multiplier")
    start, width = _int(start, "start"), _int(width, "width")
    if n < 2 or multiplier < 1 or width < 1:
        raise ValueError("Use N >= 2, a positive multiplier and positive width")
    a_primes, seeds = tuple(a_primes), tuple(seeds)
    if not 1 <= len(a_primes) <= 4:
        raise ValueError("Use 1..4 A primes for this comparison")
    k = n * multiplier
    initial = construct_polynomial(k, a_primes, form=form)
    choices = [sqrt_mod(k, p) for p in initial["a_primes"]]
    if (1 << len(choices)) * width * 3 > 16384:
        raise ValueError("Use at most 16384 samples across all CRT choices and regions")
    probe = siqs_window(k, initial["a"], initial["b"], form=form, start=start, width=1, seeds=seeds)
    seeds = tuple(probe.seeds)
    removed, remainder = _divide(probe.divisor, seeds)
    if remainder != 1:
        raise ValueError("Include every prime factor of the removed divisor in seeds")
    starts = dict(left=start - width, center=start, right=start + width)
    polynomials, windows, lessons, rows = [], {}, {}, []
    for roots in product(*choices):
        p = construct_polynomial(k, a_primes, roots=roots, form=form)
        polynomials.append(p)
        for region, first_x in starts.items():
            w = siqs_window(k, p["a"], p["b"], form=form, start=first_x, width=width, seeds=seeds)
            lesson = factor_walkthrough(w, original_n=n)
            windows[(p["b"], region)] = w
            lessons[(p["b"], region)] = lesson
            stats = w.summary()
            rows.append(dict(b=p["b"], region=region, start=first_x, samples=width,
                             median_q_bits=stats["median_value_bits"], **lesson.matrix_summary(), split=lesson.split))
    by_b = {p["b"]: p for p in polynomials}
    for p in polynomials:                         # b' = root_scale − b is the mirror choice of roots
        mirror = by_b[p["root_scale"] - p["b"]]
        c, b, a = p["coefficients"]
        mc, mb, ma = mirror["coefficients"]
        if (ma, 2 * ma - mb, ma - mb + mc) != (a, b, c):
            raise ArithmeticError("CRT mirror polynomial identity failed")
    centers = [windows[(p["b"], "center")] for p in polynomials]
    pooled = factor_walkthrough(centers, original_n=n)
    return RelationStudy(n, k, seeds, tuple(polynomials), windows, lessons, rows, pooled, True)


# ----------------------------------------------------------------------------- engine bridge

@dataclass
class Capture:
    """Polynomials from this package's own selector (the same families as the Rust engine)."""
    data: dict

    @property
    def n(self):
        return int(self.data["n"])

    @property
    def kn(self):
        return int(self.data["kn"])

    @property
    def seeds(self):
        """One inspection base for every polynomial: the A primes plus the smallest base primes."""
        required = {p for row in self.data["polynomials"] for p in row["a_primes"]}
        if len(required) > 128:
            raise ValueError("Use fewer families or supply an explicit inspection seed set")
        chosen = set(required)
        for p in self.data["base"]:
            if len(chosen) == 128:
                break
            chosen.add(p)
        return tuple(sorted(chosen))

    def records(self):
        return [dict(row, form=self.data["form"]) for row in self.data["polynomials"]]

    def dataframe(self):
        return _frame(self.records())

    def window(self, index=0, *, start=-32, width=64, seeds=None, step=1):
        index = _int(index, "index")
        if not 0 <= index < len(self.data["polynomials"]):
            raise IndexError("polynomial index out of range")
        row = self.data["polynomials"][index]
        w = siqs_window(self.kn, row["a"], row["b"], form=self.data["form"], start=start, width=width,
                        seeds=self.seeds if seeds is None else seeds, step=step)
        w.provenance.update(source="square_rulerpack_py choose_a", family=row["family"],
                            polynomial_index=row["index"], original_n=self.n, multiplier=self.data["k"])
        return w


def engine_polynomials(n, *, families=4, per_family=4, first_family=0, base=None, half=None,
                       multiplier=None, form="auto", close_a=True):
    """Run the real selector for a few families and keep the first ``per_family`` polynomials of each.

    ``base``/``half`` default to the parameter table for n; ``form`` is 'auto' (Q2 when k·n ≡ 1 mod 8),
    'q1' or 'q2'. Bounds: 16 families, 16 polynomials each, base at most 512 primes.
    """
    n = _int(n, "n")
    families, per_family, first_family = (_int(v, k) for v, k in
                                          ((families, "families"), (per_family, "per_family"), (first_family, "first_family")))
    if n < 2 or not 1 <= families <= 16 or not 1 <= per_family <= 16 or first_family < 0:
        raise ValueError("Use N >= 2, 1..16 families, 1..16 polynomials per family")
    params = Params.for_n(n)
    k = _multiplier(n) if multiplier is None else _int(multiplier, "multiplier")
    kn = k * n
    count = params.base if base is None else _int(base, "base")
    if not 2 <= count <= 512:
        raise ValueError("Use a base of 2..512 primes for this lesson (pass base=...)")
    half = params.half if half is None else _int(half, "half")
    if form == "auto":
        q2 = q2_polynomials(kn, Settings())
    elif form in ("q1", "q2"):
        q2 = form == "q2"
        if q2 and kn % 8 != 1:
            raise ValueError("Q2 needs k·n ≡ 1 (mod 8)")
    else:
        raise ValueError("form must be auto, q1 or q2")
    fb = factor_base(kn, count)
    rows = []
    for fam_index in range(first_family, first_family + families):
        fam = choose_a(kn, fb, half, fam_index, q2, close_a)
        a_primes = tuple(int(fb.p[j]) for j in fam.qs)
        for t in range(min(per_family, fam.polys)):
            b, lin, c = fam.coefficients(kn, t)
            divisor = 4 * fam.a if q2 else fam.a
            scale = 2 * fam.a if q2 else fam.a
            rows.append(dict(family=fam_index, index=t, a=fam.a, b=b, c=c, a_primes=a_primes,
                             coefficients=(c, lin, fam.a), root_scale=scale, divisor=divisor,
                             identity_valid=(b * b - kn) == divisor * c))
    return Capture(dict(n=n, k=k, kn=kn, form="q2" if q2 else "q1", base=[int(p) for p in fb.p],
                        polynomials=rows))
