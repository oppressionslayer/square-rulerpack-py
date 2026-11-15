"""The self-initialising quadratic sieve with double large primes, ported stage by stage from
square-rulerpack 0.2.0 (``siqs.rs``). Each stage is one function or method so that
``Walkthrough`` can run them one at a time and hand the intermediate data back as tables.

The stages, in the order the factorisation runs them:

1. ``multiplier``      Knuth–Schroeppel: pick k so that k·n has many small primes in its base.
2. ``factor_base``     the first ``base`` primes p with (k·n / p) = 1, with sqrt(k·n) mod p.
3. ``choose_a``        one polynomial *family*: a = q₁⋯q_s from the base, the B_l, a⁻¹ mod p.
4. ``FamilySieve``     for each of the 2^(s−1) b's of the family: roots, the log sieve over
                       x ∈ [−M, M], the threshold scan, exact division of the survivors,
                       one or two large primes, and the relation (y, exponents, large primes).
5. ``Cycles``          partial relations are edges between large primes; a cycle in that graph
                       multiplies to a full relation.
6. ``linalg``          GF(2) dependencies among the full relations' odd exponents.
7. ``square_root``     X = ∏ y, Y = ∏ p^(e/2); gcd(X − Y, n) splits n.
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

import numpy as np

from .arith import (XorShift64, is_probable_prime, is_square, legendre, modinv, pollard_brent,
                    primes_upto, sqrt_mod)
from .params import Params
from .fastarith import is_prime_cofactor, split_cofactor

try:  # the sieve kernel is 10-50× faster compiled; numpy slices do the same work without it
    import numba as _numba
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _numba = None
    HAVE_NUMBA = False

__all__ = [
    "Settings", "FactorBase", "Family", "Relation", "Candidate", "FamilySieve", "Cycles",
    "multiplier", "multiplier_scores", "factor_base", "q2_polynomials", "choose_a",
    "threshold_bits", "exact_for", "relations", "factor_with", "factor", "factor_completely",
    "square_root", "ProgressHook", "HAVE_NUMBA",
]

EXACT_MIN_BITS = 215          # below this (about 65 digits) the plain threshold is used
SMALL_EXACT = 512             # primes below this are not sieved under the exact threshold ...
SMALL_PLAIN = 30              # ... or this under the plain one; their logs are accounted otherwise
ECAP = 32.0                   # bits the exact template is lowered by so small-prime-rich values are seen
MARGIN = 1.0                  # bits of remainder allowed past the large-prime ceiling (byte-log rounding)

ProgressHook = Callable[[int, int, int, int, int], None]   # (stage, full, need, partials, polys)


@dataclass
class Settings:
    """Switches that the Rust engine reads from the environment (SIQS_EXACT, SIQS_SMALL,
    SIQS_Q2, SIQS_DLP, SIQS_CLOSE_A, SIQS_ECAP, SIQS_MARGIN); here they are a dataclass."""
    exact: bool | None = None       # None: exact threshold from EXACT_MIN_BITS, like the Rust default
    small: int | None = None        # unsieved bound; None: 512 exact / 30 plain
    q2: bool = True                 # Q2 polynomials when k·n ≡ 1 (mod 8)
    dlp: bool = True                # two large primes
    close_a: bool = True            # repair a toward the target √(2n)/M
    ecap: float = ECAP
    margin: float = MARGIN
    use_numba: bool = True
    lanczos: bool | None = None     # None: block Lanczos from linalg.LANCZOS_FROM rows, like the Rust engine
    multiplier: int | None = None   # force k

    def small_bound(self, exact: bool) -> int:
        return self.small if self.small is not None else (SMALL_EXACT if exact else SMALL_PLAIN)


# ----------------------------------------------------------------------------- factor base

@dataclass
class FactorBase:
    """The primes the sieve uses. ``root[j]`` is a square root of k·n modulo ``p[j]`` (0 when
    p divides k·n; k·n mod 2 for p = 2) and ``log[j]`` the byte log2 p the sieve adds."""
    p: np.ndarray
    root: np.ndarray
    log: np.ndarray

    def __len__(self) -> int:
        return len(self.p)

    @property
    def pmax(self) -> int:
        return int(self.p[-1])

    def frame(self):
        import pandas as pd
        return pd.DataFrame({"p": self.p, "root": self.root, "log2p": np.log2(self.p), "byte_log": self.log})


def multiplier_scores(n: int) -> list[tuple[int, float]]:
    """Knuth–Schroeppel scores for the candidate multipliers: Σ g(p)·ln p over primes to 1000
    minus ½ ln k, where a prime where k·n is a square counts 2/(p−1), a prime dividing k counts
    1/(p−1), and 2 counts by k·n mod 8. Higher is better."""
    candidates = [1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35,
                  37, 38, 39, 41, 42, 43, 46, 47, 51, 53, 55, 57, 58, 59, 61, 62, 65, 66, 67, 69, 70, 71, 73]
    primes = [int(p) for p in primes_upto(1000)]
    out = []
    ln2 = math.log(2)
    for k in candidates:
        kn = n * k
        score = -0.5 * math.log(k)
        m8 = kn % 8
        score += {1: 2 * ln2, 5: ln2, 3: 0.5 * ln2, 7: 0.5 * ln2}.get(m8, 0.0)
        for p in primes[1:]:
            lp = math.log(p)
            if k % p == 0:
                score += lp / (p - 1)
            elif legendre(kn % p, p) == 1:
                score += 2 * lp / (p - 1)
        out.append((k, score))
    return out


def multiplier(n: int) -> int:
    return max(multiplier_scores(n), key=lambda t: t[1])[0]


def factor_base(kn: int, count: int) -> FactorBase:
    """The first ``count`` primes modulo which k·n is a square (2 always first)."""
    ps = [2]
    roots = [kn % 2]
    bound = 1000
    checked = 2
    while len(ps) < count:
        for p in primes_upto(bound):
            p = int(p)
            if p <= checked:
                continue
            if len(ps) >= count:
                break
            r = kn % p
            if r == 0:
                ps.append(p)
                roots.append(0)
            elif legendre(r, p) == 1:
                ps.append(p)
                roots.append(sqrt_mod(r, p))
        checked = bound
        bound *= 2
    p = np.array(ps, dtype=np.int64)
    return FactorBase(p=p, root=np.array(roots, dtype=np.int64), log=np.floor(np.log2(p) + 0.5).astype(np.int16))


# ----------------------------------------------------------------------------- families

def q2_polynomials(kn: int, settings: Settings | None = None) -> bool:
    """YAFU's Q2 form for k·n ≡ 1 (mod 8): g(x) = a·x² + b·x + (b² − k·n)/(4a) with odd b and
    witness 2a·x + b, so the values are half the size of Q1's."""
    allow = settings.q2 if settings else True
    return allow and kn % 8 == 1


@dataclass
class Family:
    """One a = ∏ q_i and everything that is constant across its 2^(s−1) polynomials."""
    index: int
    a: int
    qs: list[int]                  # indices into the factor base of a's primes
    bs: list[int]                  # B_l: B_l ≡ ±√(k·n) (mod q_l), ≡ 0 (mod the other q's)
    ainv: np.ndarray               # (step·a)⁻¹ mod p per base prime, 0 where p divides a
    q2: bool
    b_adjust: int = 0              # Q2: a is added to b when Σ B_l is even, to make b odd

    @property
    def s(self) -> int:
        return len(self.qs)

    @property
    def step(self) -> int:
        return 2 if self.q2 else 1

    @property
    def polys(self) -> int:
        return 1 << (self.s - 1)

    def gray(self, t: int) -> int:
        return t ^ (t >> 1)

    def b_of(self, t: int) -> int:
        """b of polynomial t: Σ ±B_l with the signs of the Gray code of t."""
        g = self.gray(t)
        return sum(-B if (g >> l) & 1 else B for l, B in enumerate(self.bs)) + self.b_adjust

    def coefficients(self, kn: int, t: int) -> tuple[int, int, int]:
        """(b, lin, c) of polynomial t: g(x) = (a·x + lin)·x + c with lin = 2b, c = (b² − k·n)/a
        for Q1 and lin = b, c = (b² − k·n)/(4a) for Q2."""
        b = self.b_of(t)
        num = b * b - kn
        if self.q2:
            assert num % (4 * self.a) == 0
            return b, b, num // (4 * self.a)
        assert num % self.a == 0
        return b, 2 * b, num // self.a

    def gray_step(self, t: int) -> tuple[int, bool]:
        """From polynomial t−1 to t: which B_v flips, and whether the roots move up (+delta)."""
        v = (t & -t).bit_length() - 1
        return v, ((t >> (v + 1)) & 1) == 0


def choose_a(kn: int, base: FactorBase, half: int, seed: int, q2: bool, close: bool = True) -> Family:
    """a's primes: s of them from the primes just above 1000, by a seeded walk (family ``seed``
    picks the same primes as the Rust engine's family ``seed``), so that a ≈ √(2·k·n)/(M·step);
    then optionally one prime swapped to land closer to that target (``repair_a``)."""
    step = 2 if q2 else 1
    target_log = 0.5 * math.log(2 * kn) - math.log(half * step)
    n = len(base)
    p = base.p
    above = np.flatnonzero(p > 1000)
    lo = int(above[0]) if len(above) else n // 2
    lo = min(lo, max(n - 10, 0))
    hi = min(max(min(n, lo + 300), lo + 10), n)
    pool = list(range(lo, hi))
    rng = XorShift64(seed)
    qs: list[int] = []
    log_a = 0.0
    mid_log = math.log(int(p[pool[len(pool) // 2]]))
    while log_a < target_log - mid_log / 2 and len(qs) < 20:
        i = pool[rng.next() % len(pool)]
        if i in qs:
            continue
        qs.append(i)
        log_a += math.log(int(p[i]))
    for extra in pool:                 # a tiny base can leave one prime: make it at least two
        if len(qs) >= 2:
            break
        if extra not in qs:
            qs.append(extra)
    if close:
        _repair_a(qs, base, target_log)
    qs.sort()
    a = 1
    for i in qs:
        a *= int(p[i])
    bs = []
    for i in qs:
        q = int(p[i])
        aq = a // q
        gamma = int(base.root[i]) * modinv(aq % q, q) % q
        if gamma > q // 2:
            gamma = q - gamma
        bs.append(aq * gamma)
    ainv = np.zeros(n, dtype=np.int64)
    for i in range(n):
        pi = int(p[i])
        am = (a % pi) * step % pi
        ainv[i] = 0 if am == 0 else modinv(am, pi)
    fam = Family(index=seed, a=a, qs=qs, bs=bs, ainv=ainv, q2=q2)
    if q2 and sum(bs) % 2 == 0:
        fam.b_adjust = a
    return fam


def _repair_a(qs: list[int], base: FactorBase, target_log: float) -> None:
    p = base.p
    log_a = sum(math.log(int(p[i])) for i in qs)
    best = (abs(log_a - target_log), None)
    for slot, old in enumerate(qs):
        rest = log_a - math.log(int(p[old]))
        wanted = math.exp(target_log - rest)
        at = int(np.searchsorted(p, wanted, side="left"))
        radius = len(qs) + 2
        for j in range(max(at - radius, 0), min(at + radius, len(p))):
            if p[j] == 2 or base.root[j] == 0 or (j != old and j in qs):
                continue
            err = abs(rest + math.log(int(p[j])) - target_log)
            if err < best[0]:
                best = (err, (slot, j))
    if best[1] is not None:
        slot, j = best[1]
        qs[slot] = j


# ----------------------------------------------------------------------------- relations

@dataclass
class Relation:
    """y² ≡ (−1)^e₀ · ∏ p_j^e_j · ∏ large^e  (mod k·n).

    ``exps`` maps a column to its exponent: column 0 is the sign, column j+1 the j-th base
    prime. ``large`` holds the primes outside the base with their exponents (empty = full)."""
    y: int
    exps: dict
    large: tuple = ()

    def times(self, other: "Relation", kn: int) -> "Relation":
        exps = dict(self.exps)
        for c, e in other.exps.items():
            exps[c] = exps.get(c, 0) + e
        large = dict(self.large)
        for q, e in other.large:
            large[q] = large.get(q, 0) + e
        return Relation(self.y * other.y % kn, exps, tuple(sorted(large.items())))

    def is_square(self) -> bool:
        return all(e % 2 == 0 for e in self.exps.values()) and all(e % 2 == 0 for _, e in self.large)

    def odd_columns(self) -> list[int]:
        return sorted(c for c, e in self.exps.items() if e % 2 == 1)

    def check(self, kn: int, base: FactorBase) -> bool:
        """Verify the congruence y² ≡ ± ∏ p^e · ∏ large^e (mod k·n) by recomputing it."""
        rhs = 1
        for c, e in self.exps.items():
            rhs = rhs * (pow(int(base.p[c - 1]), e, kn) if c else pow(kn - 1, e, kn)) % kn
        for q, e in self.large:
            rhs = rhs * pow(q, e, kn) % kn
        return self.y * self.y % kn == rhs


@dataclass
class Candidate:
    """What happened to one sieve survivor: the record ``Walkthrough.trial_divide`` shows."""
    pos: int
    x: int
    g: int
    sieve_sum: int
    threshold: float
    factors: dict          # base index → exponent of g(x)'s factor-base part (sign not included)
    cofactor: int
    kind: str              # full | one-large | two-large | drop: ...
    relation: Relation | None


def threshold_bits(kn: int, base: FactorBase, params: Params, q2: bool) -> float:
    """The plain report threshold in bits: log2 of the typical |g(x)| ≈ M·√(k·n/2) (half that
    for Q2), less the slack and the room for one large prime."""
    size = math.log2(params.half) + 0.5 * (math.log2(kn) - 1.0) - (1.0 if q2 else 0.0)
    pmax = base.pmax
    return size - params.slack * math.log2(pmax) - math.log2(pmax * params.large)


def exact_for(kn: int, settings: Settings) -> bool:
    if settings.exact is not None:
        return settings.exact
    return kn.bit_length() >= EXACT_MIN_BITS


if HAVE_NUMBA:
    @_numba.njit(cache=True, nogil=True)
    def _sieve_kernel(acc, s0, s1, p, lg, two, width):   # pragma: no cover (compiled)
        for i in range(p.shape[0]):
            pi = p[i]
            l = lg[i]
            j = s0[i]
            while j < width:
                acc[j] += l
                j += pi
            if two[i]:
                j = s1[i]
                while j < width:
                    acc[j] += l
                    j += pi

    @_numba.njit(cache=True, nogil=True)
    def _root_hits(mark, s0, s1, p, width, out_row, out_prime):   # pragma: no cover (compiled)
        """Resieve: walk every prime's two root progressions and record (survivor row, prime)
        wherever the walk lands on a marked survivor. Returns the number of pairs (may exceed
        the capacity of the output arrays, in which case the caller enlarges them and reruns)."""
        n = 0
        cap = out_row.shape[0]
        for i in range(p.shape[0]):
            pi = p[i]
            j = s0[i]
            while j < width:
                r = mark[j]
                if r >= 0:
                    if n < cap:
                        out_row[n] = r
                        out_prime[n] = i
                    n += 1
                j += pi
            if s1[i] != s0[i]:
                j = s1[i]
                while j < width:
                    r = mark[j]
                    if r >= 0:
                        if n < cap:
                            out_row[n] = r
                            out_prime[n] = i
                        n += 1
                    j += pi
        return n


def _sieve_numpy(acc, s0, s1, p, lg, two, width):
    wide = p >= width
    m = wide & (s0 < width)
    np.add.at(acc, s0[m], lg[m])
    m = wide & two & (s1 < width)
    np.add.at(acc, s1[m], lg[m])
    for i in np.flatnonzero(~wide):
        acc[s0[i]::p[i]] += lg[i]
        if two[i]:
            acc[s1[i]::p[i]] += lg[i]


class FamilySieve:
    """The per-polynomial work of one family. ``run`` is the fast path (what the workers call);
    ``set_poly`` / ``sieve`` / ``scan`` / ``candidates`` are the same steps one at a time."""

    def __init__(self, kn: int, base: FactorBase, fam: Family, params: Params, threshold: float,
                 exact: bool, settings: Settings | None = None):
        st = settings or Settings()
        self.kn, self.base, self.fam, self.params, self.settings = kn, base, fam, params, st
        self.exact = exact
        self.half = params.half
        self.width = 2 * params.half + 1
        self.pmax = base.pmax
        self.large_bound = self.pmax * params.large
        self.step = fam.step
        self.y_a = self.step * fam.a
        p = base.p
        self.small = st.small_bound(exact)
        self.sieved = (p >= self.small) & (fam.ainv != 0)
        self.small_primes = [int(q) for q in p[p < self.small]]
        self.small_idx = np.flatnonzero(p < self.small)
        self.sp_p = p[self.small_idx]
        self.sp_pi = [int(q) for q in self.sp_p]
        self.sp_log = np.log2(self.sp_p.astype(float))
        self.sp_direct = (fam.ainv[self.small_idx] == 0) | (self.sp_p < 5)
        self.small_log = float(sum(2 * math.log2(q) / q for q in self.small_primes))
        self.tested = fam.ainv != 0                       # primes with sieve roots (root test)
        self.direct = [j for j in range(len(p)) if fam.ainv[j] == 0 or p[j] < 5]   # tested by division
        # Gray-code deltas: 2·B_v·(step·a)⁻¹ mod p, one row per B_v
        self.deltas = np.zeros((fam.s, len(p)), dtype=np.int64)
        for v, B in enumerate(fam.bs):
            bm = np.fromiter((B % int(q) for q in p), dtype=np.int64, count=len(p))
            self.deltas[v] = (2 * bm % p) * fam.ainv % p
        self.deltas[:, ~self.tested] = 0
        # threshold: plain (one byte everywhere) or exact (a template by 64-position chunks)
        self.threshold = threshold
        self.t = -1
        self.b = self.lin = self.c = None
        self.starts0 = self.starts1 = None
        if exact:
            self._exact_setup()
        else:
            thr = int(min(max(threshold - self.small_log, 1.0), 120.0))
            self.thr = float(thr)
            self.init = 128 - thr                     # a hit is a byte that reached 128
        # the sieved subset as contiguous arrays for the kernel
        self.sv_idx = np.flatnonzero(self.sieved)
        self.ts_idx = np.flatnonzero(self.tested)                # root-tested primes, for the resieve
        self.ts_p = p[self.ts_idx]
        self._mark = np.full(self.width, -1, dtype=np.int32)
        self.sv_p = p[self.sv_idx]
        self.sv_log = base.log[self.sv_idx].astype(np.int16)
        self.polys_done = 0

    # -- the exact threshold: per-position size of g, template, report test -------------------
    def _exact_setup(self):
        fam, kn, st = self.fam, self.kn, self.settings
        b0, lin0, c0 = fam.coefficients(kn, 0)
        disc = lin0 * lin0 - 4 * fam.a * c0        # 4·k·n for Q1, k·n for Q2: the family's constant
        self.af = float(fam.a)
        self.knoa = disc / (4 * fam.a)             # g(x) = a·(x + lin/2a)² − knoa
        lp_bits = math.log2(self.large_bound)
        self.allow = (lp_bits * self.params.dlp_exp if (st.dlp and self.params.dlp) else lp_bits) + st.margin
        lg_max = math.log2(self.knoa)
        floor = min(max(lg_max - self.allow - st.ecap - 12.0, 1.0), 120.0)
        width, half = self.width, self.half
        los = np.arange(0, width, 64, dtype=np.int64)
        his = np.minimum(los + 64, width)
        root = math.sqrt(self.knoa / self.af)
        x0 = los.astype(float) - half - 8.0
        x1 = his.astype(float) - half + 8.0
        crosses = ((x0 - root) * (x1 - root) <= 0) | ((x0 + root) * (x1 + root) <= 0)
        g0 = np.abs(self.af * x0 * x0 - self.knoa)
        g1 = np.abs(self.af * x1 * x1 - self.knoa)
        lg_min = np.where(crosses, 0.0, np.log2(np.maximum(np.minimum(g0, g1), 1.0)))
        thr = np.clip(np.floor(lg_min - self.allow - st.ecap), floor, 120.0).astype(np.int16)
        self.tmpl_thr = thr                          # scan threshold per 64-chunk (the exact test follows)
        self.tmpl_init = (128 - thr).astype(np.int16)
        self.init_array = np.repeat(self.tmpl_init, 64)[:width].astype(np.int16)

    def small_bits(self, g: int) -> float:
        """Exact log2 of the unsieved small-prime part of g(x), with multiplicity."""
        v = abs(g)
        e = 0.0
        for q in self.small_primes:
            if v % q == 0:
                k = 0
                while v % q == 0:
                    v //= q
                    k += 1
                e += k * math.log2(q)
        return e

    # -- polynomial selection ------------------------------------------------------------------
    def set_poly(self, t: int) -> None:
        """Polynomial t from scratch: b, the coefficients and both sieve starts per prime."""
        fam, p = self.fam, self.base.p
        self.t = t
        self.b, self.lin, self.c = fam.coefficients(self.kn, t)
        bm = np.fromiter((self.b % int(q) for q in p), dtype=np.int64, count=len(p))
        r1 = (self.base.root + p - bm) % p * fam.ainv % p
        r2 = (2 * p - self.base.root - bm) % p * fam.ainv % p
        self.starts0 = (r1 + self.half) % p
        self.starts1 = (r2 + self.half) % p
        self._after_poly_change()

    def advance(self, t: int) -> None:
        """Polynomial t from t−1 by the Gray-code step: one add or subtract per prime."""
        fam, p = self.fam, self.base.p
        v, up = fam.gray_step(t)
        d = self.deltas[v]
        if up:
            self.b -= 2 * fam.bs[v]
            self.starts0 = (self.starts0 + d) % p
            self.starts1 = (self.starts1 + d) % p
        else:
            self.b += 2 * fam.bs[v]
            self.starts0 = (self.starts0 - d + p) % p
            self.starts1 = (self.starts1 - d + p) % p
        self.t = t
        num = self.b * self.b - self.kn
        if fam.q2:
            self.lin, self.c = self.b, num // (4 * fam.a)
        else:
            self.lin, self.c = 2 * self.b, num // fam.a
        self._after_poly_change()

    def _after_poly_change(self):
        if self.exact:
            self.boa = self.lin / (2 * self.fam.a)
        self.sv_s0 = self.starts0[self.sv_idx]
        self.sv_s1 = self.starts1[self.sv_idx]
        self.sv_two = self.sv_s0 != self.sv_s1

    def roots(self) -> tuple[np.ndarray, np.ndarray]:
        """The two roots of g modulo each base prime for the current polynomial."""
        p = self.base.p
        return (self.starts0 - self.half) % p, (self.starts1 - self.half) % p

    def g(self, x: int) -> int:
        return (self.fam.a * x + self.lin) * x + self.c

    # -- sieve, scan, candidates ---------------------------------------------------------------
    def sieve(self) -> np.ndarray:
        """The log sieve: every sieved prime adds its byte log at both roots along the interval.
        Returns the int16 array; position i is x = i − half."""
        if self.exact:
            acc = self.init_array.copy()
        else:
            acc = np.full(self.width, self.init, dtype=np.int16)
        if HAVE_NUMBA and self.settings.use_numba:
            _sieve_kernel(acc, self.sv_s0, self.sv_s1, self.sv_p, self.sv_log, self.sv_two, self.width)
        else:
            _sieve_numpy(acc, self.sv_s0, self.sv_s1, self.sv_p, self.sv_log, self.sv_two, self.width)
        return acc

    def sieve_sum(self, acc: np.ndarray, pos) -> np.ndarray:
        """What the sieve added at ``pos``: the byte minus its starting value."""
        pos = np.asarray(pos)
        if self.exact:
            return acc[pos] - self.tmpl_init[pos >> 6]
        return acc[pos] - self.init

    def threshold_at(self, pos) -> np.ndarray:
        """The bits a position must reach: the plain threshold, or the exact one
        log2|g(x)| − allow (what the exact report test compares against)."""
        pos = np.asarray(pos, dtype=np.int64)
        if not self.exact:
            return np.full(pos.shape, self.thr)
        t = (pos - self.half).astype(float) + self.boa
        g = np.abs(self.af * t * t - self.knoa)
        return np.where(g <= 1.0, 0.0, np.log2(np.maximum(g, 1.0)) - self.allow)

    def small_bits_many(self, positions: np.ndarray) -> np.ndarray:
        """Exact log2 of the unsieved small-prime part of g(x) at each position: which small
        primes divide g(x) comes from the root test (p | g(x) ⇔ x ≡ root mod p), the
        multiplicity from dividing the few that do."""
        out = np.zeros(len(positions), dtype=float)
        if len(positions) == 0 or len(self.small_idx) == 0:
            return out
        pos = positions[:, None]
        p = self.sp_p[None, :]
        hit = ((pos - self.starts0[self.small_idx][None, :]) % p == 0) | ((pos - self.starts1[self.small_idx][None, :]) % p == 0)
        hit |= self.sp_direct[None, :]                       # primes without roots: divide and see
        rows, cols = np.nonzero(hit)
        last = -1
        for r, c in zip(rows.tolist(), cols.tolist()):
            if r != last:
                v = abs(self.g(int(positions[r]) - self.half))
                last = r
            q = self.sp_pi[c]
            k = 0
            while v % q == 0:
                v //= q
                k += 1
            out[r] += k * self.sp_log[c]
        return out

    def scan(self, acc: np.ndarray) -> np.ndarray:
        """Positions whose sieve byte reached 128 (plain: sum ≥ threshold). Under the exact
        threshold each is then re-tested against its own log2|g(x)|, with the exact small-prime
        part counted only when the sieve sum alone falls short."""
        reports = np.flatnonzero(acc >= 128)
        if not self.exact or len(reports) == 0:
            return reports
        need = self.threshold_at(reports)
        summ = self.sieve_sum(acc, reports).astype(float)
        ok = (summ >= need) | (need <= 0.0)
        short = reports[~ok]
        if len(short):
            ok[~ok] = summ[~ok] + self.small_bits_many(short) >= need[~ok]
        return reports[ok]

    def classify_cofactor(self, v: int) -> tuple[str, tuple]:
        """The Rust ``split_cofactor`` rule: 1 → full; below the large-prime bound → one large
        prime; between pmax² and bound^dlp_exp, below 2^63, composite and not a square → split
        with Pollard–Brent into two large primes, each below the bound; anything else dropped."""
        lb = self.large_bound
        if v == 1:
            return "full", ()
        if v < lb:
            return "one-large", ((v, 1),)
        st = self.settings
        if not (st.dlp and self.params.dlp):
            return "drop: cofactor above the large-prime bound", ()
        if v < self.pmax * self.pmax:
            return "drop: cofactor below pmax² but above the bound (it would be a prime)", ()
        if v >= 1 << 63 or v >= lb ** self.params.dlp_exp:
            return "drop: cofactor above the double-large-prime ceiling", ()
        r = math.isqrt(v)
        if r * r == v:
            return ("two-large", ((r, 2),)) if r < lb else ("drop: square of a prime past the bound", ())
        if is_prime_cofactor(v):
            return "drop: probable prime, too big for one large prime", ()
        f = split_cofactor(v, v)
        if f is None:
            return "drop: cofactor not split", ()
        a, b = min(f, v // f), max(f, v // f)
        if b >= lb:
            return "drop: a large prime past the bound", ()
        return "two-large", (((a, 2),) if a == b else ((a, 1), (b, 1)))

    def _root_test(self, hits: np.ndarray) -> list[list[int]]:
        """For each survivor, the indices of the root-tested base primes dividing g(x), by
        resieving: every prime walks its two root progressions over a mark array of the
        survivors, O(width/p) per prime like the sieve itself, instead of hits × primes modulos."""
        mark = self._mark
        mark[hits] = np.arange(len(hits), dtype=np.int32)
        s0 = self.starts0[self.ts_idx]
        s1 = self.starts1[self.ts_idx]
        cap = 64 * len(hits) + 1024
        while True:
            out_row = np.empty(cap, dtype=np.int32)
            out_prime = np.empty(cap, dtype=np.int32)
            n = _root_hits(mark, s0, s1, self.ts_p, self.width, out_row, out_prime)
            if n <= cap:
                break
            cap = n + 64
        mark[hits] = -1
        per_row: list[list[int]] = [[] for _ in range(len(hits))]
        ts = self.ts_idx
        for r, i in zip(out_row[:n].tolist(), out_prime[:n].tolist()):
            per_row[r].append(int(ts[i]))
        return per_row

    def candidates(self, acc: np.ndarray, hits: np.ndarray) -> list[Candidate]:
        """Exact division of every survivor: which base primes divide g(x) (the root test,
        vectorised over the base), their multiplicities, the cofactor, and the relation."""
        fam, base, kn = self.fam, self.base, self.kn
        p = base.p
        tested = self.tested
        out = []
        if len(hits) == 0:
            return out
        sums = self.sieve_sum(acc, hits)
        thrs = self.threshold_at(hits)
        # which base primes divide g(x) at each survivor: resieve the root progressions over the
        # marked survivors (numba), or the 2-D root test hits × primes in numpy
        per_row = self._root_test(hits) if HAVE_NUMBA and self.settings.use_numba else None
        if per_row is None:
            hitm = (((hits[:, None] - self.starts0[None, :]) % p == 0) | ((hits[:, None] - self.starts1[None, :]) % p == 0)) & tested[None, :]
        for row, (pos, ssum, thr) in enumerate(zip(hits.tolist(), sums.tolist(), thrs.tolist())):
            x = pos - self.half
            g = self.g(x)
            if g == 0:
                continue
            exps = {qi + 1: 1 for qi in fam.qs}
            if fam.q2:
                exps[1] = exps.get(1, 0) + 2
            if g < 0:
                exps[0] = 1
            v = -g if g < 0 else g
            idx = per_row[row] if per_row is not None else np.flatnonzero(hitm[row]).tolist()
            factors = {}
            for j in idx + self.direct:
                q = int(p[j])
                while v % q == 0:
                    v //= q
                    factors[j] = factors.get(j, 0) + 1
            for j, e in factors.items():
                exps[j + 1] = exps.get(j + 1, 0) + e
            kind, large = self.classify_cofactor(v)
            rel = None
            if not kind.startswith("drop"):
                rel = Relation((self.y_a * x + self.b) % kn, exps, large)
            out.append(Candidate(pos, x, g, int(ssum), float(thr), factors, v, kind, rel))
        return out

    def run(self, max_polys: int | None = None) -> tuple[list[Relation], int]:
        """Every polynomial of the family; the relations found and the polynomial count."""
        out = []
        polys = self.fam.polys if max_polys is None else min(max_polys, self.fam.polys)
        for t in range(polys):
            if t == 0:
                self.set_poly(0)
            else:
                self.advance(t)
            acc = self.sieve()
            hits = self.scan(acc)
            if len(hits):
                out.extend(c.relation for c in self.candidates(acc, hits) if c.relation is not None)
        self.polys_done = polys
        return out, polys


# ----------------------------------------------------------------------------- cycles

class Cycles:
    """Partials as edges between their odd-exponent large primes (1 stands for "no second
    prime"), kept as a forest with parent pointers. A partial that joins two vertices already
    connected closes a cycle; the relations along that cycle multiply to a full relation,
    every large prime appearing an even number of times."""

    def __init__(self):
        self.edges: list[Relation] = []
        self.up: dict[int, tuple[int, int]] = {}
        self.size: dict[int, int] = {}
        self.closed = 0

    def root(self, v: int) -> int:
        while v in self.up:
            v = self.up[v][0]
        return v

    def _reroot(self, x: int) -> None:
        path = []
        v = x
        while v in self.up:
            parent, e = self.up[v]
            path.append((v, parent, e))
            v = parent
        for child, parent, e in path:
            del self.up[child]
            self.up[parent] = (child, e)
        self.up.pop(x, None)

    def add(self, r: Relation) -> list[Relation] | None:
        odd = [q for q, e in r.large if e % 2 == 1]
        if not odd:
            return [r]
        if len(odd) == 1:
            u, v = 1, odd[0]
        elif len(odd) == 2:
            u, v = odd
        else:
            return None
        idx = len(self.edges)
        self.edges.append(r)
        ru, rv = self.root(u), self.root(v)
        if ru != rv:
            su, sv = self.size.get(ru, 1), self.size.get(rv, 1)
            small, big, sroot, broot = (u, v, ru, rv) if su <= sv else (v, u, rv, ru)
            self._reroot(small)
            self.up[small] = (big, idx)
            self.size.pop(sroot, None)
            self.size[broot] = su + sv
            return None
        seen = {}
        x, i = u, 0
        while True:
            seen[x] = i
            i += 1
            if x in self.up:
                x = self.up[x][0]
            else:
                break
        path = [self.edges[idx]]
        y = v
        up_from_v = []
        while y not in seen:
            parent, e = self.up[y]
            up_from_v.append(e)
            y = parent
        meet = y
        x = u
        while x != meet:
            parent, e = self.up[x]
            path.append(self.edges[e])
            x = parent
        for e in up_from_v:
            path.append(self.edges[e])
        self.closed += 1
        return path

    @property
    def vertices(self) -> int:
        return len(set(self.up) | {p for p, _ in self.up.values()})

    def frame(self):
        import pandas as pd
        rows = [{"edge": i, "large_primes": " ".join(f"{q}^{e}" if e > 1 else str(q) for q, e in r.large),
                 "y": r.y} for i, r in enumerate(self.edges)]
        return pd.DataFrame(rows)


def combine(path: list[Relation], kn: int) -> Relation:
    acc = path[0]
    for r in path[1:]:
        acc = acc.times(r, kn)
    return acc


# ----------------------------------------------------------------------------- collection

@dataclass
class Collection:
    """The outcome of relation collection."""
    full: list[Relation]
    polys: int
    partials: int
    families: int
    seconds: float
    cycles: Cycles


def relations(kn: int, base: FactorBase, params: Params, need: int, families: Iterable[tuple[int, list, int]],
              progress: ProgressHook | None = None, verbose: bool = False, budget: float | None = None,
              family_limit: int | None = None, cycles: Cycles | None = None,
              full: list[Relation] | None = None) -> Collection:
    """Merge families (in index order, from the serial generator or the worker pool) until
    ``need`` full relations are held: fulls go straight in, partials into the cycle forest, and
    each closed cycle is multiplied into one more full relation."""
    full = full if full is not None else []
    graph = cycles if cycles is not None else Cycles()
    started = time.time()
    last_print = started
    mark = (started, len(full))
    polys = 0
    rounds = 0
    if progress:
        progress(1, len(full), need, len(graph.edges), 0)
    for idx, rels, npolys in families:
        polys += npolys
        rounds += 1
        for r in rels:
            if not r.large:
                full.append(r)
                continue
            cyc = graph.add(r)
            if cyc:
                full.append(combine(cyc, kn))
        if progress:
            progress(1, len(full), need, len(graph.edges), polys)
        now = time.time()
        if verbose and now - last_print >= 5.0:
            t0, c0 = mark
            recent = (len(full) - c0) / max(now - t0, 1e-9)
            overall = len(full) / max(now - started, 1e-9)
            rate = recent if len(full) > c0 + 20 else overall
            left = (need - len(full)) / rate if rate > 0 else float("inf")
            print(f"  {len(full)}/{need} relations, {len(graph.edges)} partial edges, {polys} polynomials, "
                  f"{now - started:.0f} s elapsed, about {left:.0f} s to go", file=sys.stderr, flush=True)
            if now - t0 >= 20.0:
                mark = (now, len(full))
            last_print = now
        if len(full) >= need or (family_limit is not None and rounds >= family_limit) \
                or (budget is not None and now - started >= budget):
            break
    return Collection(full, polys, len(graph.edges), rounds, time.time() - started, graph)


def serial_families(kn: int, base: FactorBase, params: Params, threshold: float, exact: bool,
                    settings: Settings, start: int = 1) -> Iterator[tuple[int, list, int]]:
    q2 = q2_polynomials(kn, settings)
    idx = start
    while True:
        fam = choose_a(kn, base, params.half, idx, q2, settings.close_a)
        rels, npolys = FamilySieve(kn, base, fam, params, threshold, exact, settings).run()
        yield idx, rels, npolys
        idx += 1


# ----------------------------------------------------------------------------- square root

def square_root(dep: list[int], rels: list[Relation], base: FactorBase, kn: int, n: int) -> dict:
    """One dependency: X = ∏ y mod k·n, Y = ∏ p^(Σe/2) · ∏ large^(Σe/2) mod k·n; then
    gcd(X − Y, n) and gcd(X + Y, n). Returns every intermediate for inspection."""
    X = 1
    cols = len(base) + 1
    total = {}
    larges = {}
    for k in dep:
        r = rels[k]
        X = X * r.y % kn
        for c, e in r.exps.items():
            total[c] = total.get(c, 0) + e
        for q, e in r.large:
            larges[q] = larges.get(q, 0) + e
    even = all(e % 2 == 0 for e in total.values()) and all(e % 2 == 0 for e in larges.values())
    Y = 1
    if even:
        for c, e in total.items():
            if c and e:
                Y = Y * pow(int(base.p[c - 1]), e // 2, kn) % kn
        for q, e in larges.items():
            Y = Y * pow(q, e // 2, kn) % kn
    g1 = math.gcd(abs(X - Y), n)
    g2 = math.gcd(X + Y, n)
    found = next((g for g in (g1, g2) if 1 < g < n), None)
    return {"relations": len(dep), "all_even": even, "X": X, "Y": Y, "X2_eq_Y2_mod_kn": even and X * X % kn == Y * Y % kn,
            "gcd(X-Y, n)": g1, "gcd(X+Y, n)": g2, "factor": found}


# ----------------------------------------------------------------------------- top level

def factor_with(n: int, params: Params, workers: int = 1, settings: Settings | None = None,
                progress: ProgressHook | None = None, verbose: bool = False) -> int | None:
    """A nontrivial factor of the odd composite n, or None (prime, too small, or unlucky)."""
    from . import linalg
    from .workers import family_stream
    st = settings or Settings()
    if progress:
        progress(0, 0, 0, 0, 0)
    if n < 4:
        return None
    for p in primes_upto(10_000):
        p = int(p)
        if n % p == 0:
            return None if n == p else p
    r = math.isqrt(n)
    if r * r == n:
        return r
    if is_probable_prime(n):
        return None
    k = st.multiplier or multiplier(n)
    kn = n * k
    q2 = q2_polynomials(kn, st)
    exact = exact_for(kn, st)
    if verbose:
        print(f"  siqs: input {len(str(n))} digits / {n.bit_length()} bits; kN {len(str(kn))} digits / {kn.bit_length()} bits; multiplier {k}", file=sys.stderr)
        print(f"  base {params.base} half {params.half} ({params.half // 32768} blocks/side) large {params.large} slack {params.slack} extra {params.extra}; "
              f"{workers} workers; {'numba' if HAVE_NUMBA and st.use_numba else 'numpy'}; {'Q2' if q2 else 'Q1'}; {'exact' if exact else 'plain'} threshold", file=sys.stderr)
    base = factor_base(kn, params.base)
    need = len(base) + 1 + params.extra
    threshold = threshold_bits(kn, base, params, q2)
    coll = relations(kn, base, params, need, family_stream(kn, base, params, threshold, exact, st, workers),
                     progress=progress, verbose=verbose)
    rels = coll.full
    if progress:
        progress(2, len(rels), need, coll.partials, coll.polys)
    if verbose:
        print(f"  {len(rels)} relations ({coll.partials} partial edges) from {coll.polys} polynomials in {coll.seconds:.1f} s", file=sys.stderr)
    cols = len(base) + 1
    rels = rels[:need + 64]                      # more rows than that only cost elimination time
    odd = [r.odd_columns() for r in rels]
    keep = linalg.prune_singletons(odd, cols)
    lstats: dict = {}
    t0 = time.perf_counter()
    deps = linalg.dependencies([odd[i] for i in keep], cols, lanczos=st.lanczos, use_numba=st.use_numba, stats=lstats)
    if verbose:
        extra = f", {lstats['iterations']} iterations" if "iterations" in lstats else ""
        print(f"  {len(keep)} rows × {cols} columns: {len(deps)} dependencies by {lstats.get('solver')}{extra} in {time.perf_counter() - t0:.2f} s", file=sys.stderr)
    if progress:
        progress(3, len(rels), need, coll.partials, coll.polys)
    for dep in deps:
        res = square_root([keep[i] for i in dep], rels, base, kn, n)
        if res["factor"]:
            return res["factor"]
    return None


def _use_native(engine: str) -> bool:
    """engine: "auto" (the Rust extension when installed, else Python), "native" or "python"."""
    from . import native
    if engine == "python":
        return False
    if engine == "native":
        native.require()
        return True
    if engine != "auto":
        raise ValueError('engine must be "auto", "native" or "python"')
    return native.HAVE_NATIVE


def factor(n: int, workers: int = 1, settings: Settings | None = None, verbose: bool = False,
           engine: str = "auto") -> int | None:
    """A nontrivial factor of n from the sieve, or None. ``engine="auto"`` uses the Rust extension
    (square-rulerpack-native) when it is installed; ``"python"`` forces this port."""
    if _use_native(engine):
        from . import native
        return native.factor(n, threads=workers, settings=settings, verbose=verbose)
    return factor_with(n, Params.for_n(n), workers=workers, settings=settings, verbose=verbose)


def factor_completely(n: int, workers: int = 1, settings: Settings | None = None, verbose: bool = False,
                      engine: str = "auto") -> list[int]:
    """Trial division to 100,000, squares, probable-prime tests, and the sieve on every
    composite piece until all are prime. Sorted, with multiplicity. ``engine="auto"`` uses the
    Rust extension when it is installed (``pip install square-rulerpack[native]``), ``"python"``
    forces this port, ``"native"`` requires the extension."""
    if _use_native(engine):
        from . import native
        return native.factor_completely(n, threads=workers, settings=settings, verbose=verbose)
    out = []
    if n == 0:
        return out
    for p in primes_upto(100_000):
        p = int(p)
        while n % p == 0:
            n //= p
            out.append(p)

    def split(m: int):
        if m == 1:
            return
        if is_probable_prime(m):
            out.append(m)
            return
        r = math.isqrt(m)
        if r * r == m:
            split(r)
            split(r)
            return
        f = factor(m, workers=workers, settings=settings, verbose=verbose, engine="python")
        if f is None:
            out.append(m)
        else:
            split(f)
            split(m // f)

    if n > 1:
        split(n)
    return sorted(out)
