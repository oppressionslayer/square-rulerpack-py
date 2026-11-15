"""Step through a factorisation by hand, one stage at a time, with every intermediate result
as a pandas DataFrame or numpy array. Made for IPython / Jupyter:

    >>> import square_rulerpack_py as sq
    >>> w = sq.Walkthrough(1009732533765211)
    >>> w.summary()                 # size, trial division, primality
    >>> w.multiplier()              # Knuth–Schroeppel scores, the chosen k
    >>> w.factor_base()             # p, sqrt(k·n) mod p, byte logs
    >>> w.family()                  # a = q1·q2·…, the B_l, how many polynomials
    >>> w.polynomials()             # every b of the family, Gray-code order
    >>> w.roots(0)                  # the two roots of g modulo each prime
    >>> s = w.sieve(0); s.table()   # the log sieve: positions, sums, threshold, survivors
    >>> w.trial_divide(0)           # exact division of the survivors, cofactor, relation kind
    >>> w.collect()                 # the rest, with workers, until enough relations
    >>> w.relations(); w.partials() # what was collected
    >>> w.matrix(); w.dependencies()
    >>> w.square_root(0)            # X, Y, gcd(X − Y, n): the factor
    >>> w.factors()

Each method stores its result on the object too (``w.k``, ``w.base``, ``w.fam``, ``w.full``,
``w.deps`` …), so anything can be picked apart further in the shell.
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass

import numpy as np

from . import linalg
from .arith import is_probable_prime, is_square, legendre, primes_upto
from .engine import (Candidate, Cycles, FactorBase, Family, FamilySieve, Relation, Settings, choose_a,
                     exact_for, factor_base, multiplier_scores, q2_polynomials, relations, square_root,
                     threshold_bits, HAVE_NUMBA)
from .params import Params
from .workers import default_workers, family_stream


def _pd():
    import pandas as pd
    return pd


def _fmt_factors(exps: dict, base: FactorBase, large=()) -> str:
    parts = []
    for c in sorted(exps):
        e = exps[c]
        if e == 0:
            continue
        name = "-1" if c == 0 else str(int(base.p[c - 1]))
        parts.append(name if e == 1 else f"{name}^{e}")
    for q, e in large:
        parts.append(f"[{q}]" if e == 1 else f"[{q}]^{e}")
    return " · ".join(parts) if parts else "1"


@dataclass
class SieveResult:
    """One polynomial's sieve: the array (position i is x = i − half), the survivors, and the
    objects to read them with."""
    t: int
    acc: np.ndarray
    hits: np.ndarray
    fs: FamilySieve

    @property
    def half(self) -> int:
        return self.fs.half

    def table(self, top: int = 25, positions=None):
        """The best ``top`` positions by sieve sum (or the given positions): x, g(x), its size
        in bits, what the sieve summed there, the threshold it had to reach, and whether it
        survived (under the exact threshold the survivor test is per position)."""
        pd = _pd()
        fs = self.fs
        if positions is None:
            summ = fs.sieve_sum(self.acc, np.arange(fs.width))
            positions = np.argsort(-summ, kind="stable")[:top]
        pos = np.asarray(positions, dtype=np.int64)
        xs = pos - fs.half
        gs = [fs.g(int(x)) for x in xs]
        hitset = set(self.hits.tolist())
        return pd.DataFrame({
            "pos": pos, "x": xs, "g(x)": gs,
            "log2|g|": [math.log2(abs(g)) if g else 0.0 for g in gs],
            "sieve_sum": fs.sieve_sum(self.acc, pos),
            "threshold": fs.threshold_at(pos),
            "byte": self.acc[pos],
            "survives": [int(p) in hitset for p in pos],
        })

    def __repr__(self):
        return f"SieveResult(t={self.t}, width={len(self.acc)}, survivors={len(self.hits)}, max_sum={int(self.fs.sieve_sum(self.acc, int(np.argmax(self.acc))))})"


class Walkthrough:
    """The factorisation of n as a sequence of steps. See the module docstring for the order."""

    def __init__(self, n: int, workers: int | None = None, params: Params | None = None,
                 settings: Settings | None = None, verbose: bool = True):
        self.n = int(n)
        self.workers = workers if workers is not None else default_workers()
        self.params = params or Params.for_n(self.n)
        self.settings = settings or Settings()
        self.verbose = verbose
        self.k = self.kn = None
        self.base: FactorBase | None = None
        self.q2 = self.exact = None
        self.threshold = None
        self.fam: Family | None = None
        self.fs: FamilySieve | None = None
        self.families: list[Family] = []
        self.next_family = 1
        self.full: list[Relation] = []
        self.cycles = Cycles()
        self.polys = 0
        self.done_polys: set[tuple[int, int]] = set()
        self.keep = self.odd = self.deps = None
        self.collection = None
        self._last_sieve: SieveResult | None = None

    def _say(self, *a):
        if self.verbose:
            print(*a, file=sys.stderr)

    # ------------------------------------------------------------------ 0. the input
    def summary(self):
        """Size of n, the cheap checks the sieve is preceded by, and the parameters chosen."""
        pd = _pd()
        n = self.n
        small = next((int(p) for p in primes_upto(10_000) if n % int(p) == 0), None)
        d = {
            "n": n, "digits": len(str(n)), "bits": n.bit_length(),
            "small factor (< 10,000)": small, "perfect square": is_square(n), "probable prime": is_probable_prime(n),
            "base (primes)": self.params.base, "half interval M": self.params.half, "large-prime multiple": self.params.large,
            "threshold slack": self.params.slack, "extra relations": self.params.extra, "double-large-prime exponent": self.params.dlp_exp,
            "workers": self.workers, "sieve kernel": "numba" if HAVE_NUMBA and self.settings.use_numba else "numpy",
        }
        return pd.Series(d, name="input")

    # ------------------------------------------------------------------ 1. multiplier
    def multiplier(self):
        """Knuth–Schroeppel: score every candidate k by how many small primes k·n is a square
        modulo (weighted by log p / (p − 1)), minus ½ ln k for the growth. The best k wins; the
        sieve factors k·n and the gcd at the end is taken with n."""
        pd = _pd()
        scores = multiplier_scores(self.n)
        df = pd.DataFrame(scores, columns=["k", "score"])
        df["kn mod 8"] = [(self.n * k) % 8 for k in df.k]
        df["kn bits"] = [(self.n * k).bit_length() for k in df.k]
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        self.k = self.settings.multiplier or int(df.k[0])
        self.kn = self.n * self.k
        self.q2 = q2_polynomials(self.kn, self.settings)
        self.exact = exact_for(self.kn, self.settings)
        df["chosen"] = df.k == self.k
        self._say(f"k = {self.k}: k·n has {self.kn.bit_length()} bits, k·n mod 8 = {self.kn % 8} → "
                  f"{'Q2' if self.q2 else 'Q1'} polynomials, {'exact' if self.exact else 'plain'} threshold")
        return df

    # ------------------------------------------------------------------ 2. factor base
    def factor_base(self):
        """The first ``base`` primes with (k·n / p) = 1 (plus 2, and any p dividing k·n), each
        with a square root of k·n mod p and the rounded log2 the sieve adds."""
        pd = _pd()
        if self.kn is None:
            self.multiplier()
        self.base = factor_base(self.kn, self.params.base)
        self.threshold = threshold_bits(self.kn, self.base, self.params, self.q2)
        b = self.base
        small = self.settings.small_bound(self.exact)
        df = pd.DataFrame({"p": b.p, "root": b.root, "p - root": (b.p - b.root) % b.p,
                           "legendre(kn, p)": [legendre(self.kn, int(p)) if p != 2 else 1 for p in b.p],
                           "log2 p": np.log2(b.p), "byte log": b.log, "sieved": b.p >= small})
        df.index.name = "j"
        self._say(f"{len(b)} primes, largest {b.pmax:,}; large-prime bound {b.pmax * self.params.large:,}; "
                  f"primes below {small} are not sieved; plain threshold {self.threshold:.1f} bits; "
                  f"need {len(b) + 1 + self.params.extra} relations")
        return df

    # ------------------------------------------------------------------ 3. a family
    def family(self, index: int | None = None):
        """One a = q₁⋯q_s: its primes (picked from those just above 1000 so that a is near the
        target √(2kn)/M), and the B_l from which every b of the family is Σ ±B_l. The family has
        2^(s−1) polynomials; the same ``index`` gives the same a as the Rust engine."""
        pd = _pd()
        if self.base is None:
            self.factor_base()
        idx = index if index is not None else self.next_family
        self.next_family = idx + 1
        self.fam = choose_a(self.kn, self.base, self.params.half, idx, self.q2, self.settings.close_a)
        self.families.append(self.fam)
        self.fs = FamilySieve(self.kn, self.base, self.fam, self.params, self.threshold, self.exact, self.settings)
        self.fs.set_poly(0)
        fam, b = self.fam, self.base
        target = math.sqrt(2 * self.kn) / (self.params.half * fam.step) if self.kn.bit_length() < 1000 else float("nan")
        df = pd.DataFrame({"j": fam.qs, "q": [int(b.p[j]) for j in fam.qs], "root of kn mod q": [int(b.root[j]) for j in fam.qs],
                           "B_l": fam.bs, "B_l mod q": [B % int(b.p[j]) for B, j in zip(fam.bs, fam.qs)]})
        self._say(f"family {idx}: a = {fam.a} ({fam.a.bit_length()} bits; target {target:.4g}), s = {fam.s} primes, "
                  f"{fam.polys} polynomials; b₀ = {fam.b_of(0)}" + (f" (a added to make b odd)" if fam.b_adjust else ""))
        return df

    def polynomials(self):
        """Every polynomial of the family: the Gray code picks which B_l flips sign between
        consecutive b's, so g(x) = (a·x + lin)·x + c changes by one root shift per prime."""
        pd = _pd()
        if self.fam is None:
            self.family()
        fam = self.fam
        rows = []
        for t in range(fam.polys):
            b, lin, c = fam.coefficients(self.kn, t)
            g = fam.gray(t)
            signs = "".join("-" if (g >> l) & 1 else "+" for l in range(fam.s))
            v, up = fam.gray_step(t) if t else (None, None)
            rows.append({"t": t, "gray": format(g, f"0{fam.s}b"), "signs of B_l": signs, "flips B": "" if t == 0 else v, "roots move": "" if t == 0 else ("+delta" if up else "-delta"),
                         "b": b, "lin": lin, "c": c, "b² ≡ kn (mod a)": (b * b - self.kn) % fam.a == 0})
        return pd.DataFrame(rows)

    def _poly(self, t: int):
        if self.fs is None:
            self.family()
        if self.fs.t != t:
            self.fs.set_poly(t)

    def roots(self, t: int = 0):
        """The two roots of g modulo each base prime for polynomial t: x ≡ (±√kn − b)·(step·a)⁻¹
        (mod p). The sieve starts at position root + M and strides by p. Primes dividing a have
        no roots (they divide every value exactly once and are credited directly)."""
        pd = _pd()
        self._poly(t)
        fs, b = self.fs, self.base
        r1, r2 = fs.roots()
        has = self.fam.ainv != 0
        df = pd.DataFrame({"p": b.p, "root1": np.where(has, r1, -1), "root2": np.where(has, r2, -1),
                           "start1": np.where(has, fs.starts0, -1), "start2": np.where(has, fs.starts1, -1),
                           "g(root1) mod p": [fs.g(int(x)) % int(p) if h else None for x, p, h in zip(r1, b.p, has)],
                           "sieved": fs.sieved})
        df.index.name = "j"
        return df

    # ------------------------------------------------------------------ 4. sieve and divide
    def sieve(self, t: int = 0) -> SieveResult:
        """Polynomial t's log sieve: an int16 array over x ∈ [−M, M], each sieved prime adding
        its byte log at both roots along its stride. Positions whose byte reaches 128 are
        survivors (under the exact threshold they are then re-tested against their own log2|g|)."""
        self._poly(t)
        acc = self.fs.sieve()
        hits = self.fs.scan(acc)
        res = SieveResult(t, acc, hits, self.fs)
        self._last_sieve = res
        self._say(f"polynomial {t}: {len(acc):,} positions, {int((acc >= 128).sum())} reports, {len(hits)} survivors")
        return res

    def trial_divide(self, t: int | None = None, sieve: SieveResult | None = None, add: bool = True):
        """Exact division of each survivor of polynomial t: which base primes divide g(x) (found
        by the root test, p | g(x) ⇔ x ≡ root mod p), with multiplicity; the cofactor left; and
        what that makes the relation: full, one large prime, two large primes (the cofactor
        split by Pollard–Brent), or dropped. Relations found are kept for ``collect``."""
        pd = _pd()
        if sieve is None:
            t = 0 if t is None else t
            last = self._last_sieve
            sieve = last if (last is not None and last.t == t and last.fs is self.fs) else self.sieve(t)
        else:
            self._poly(sieve.t)
        fs = self.fs
        cands = fs.candidates(sieve.acc, sieve.hits)
        key = (self.fam.index, sieve.t)
        if add and key not in self.done_polys:
            self.done_polys.add(key)
            self.polys += 1
            for c in cands:
                if c.relation is not None:
                    self._absorb(c.relation)
        rows = []
        for c in cands:
            rows.append({"x": c.x, "g(x)": c.g, "log2|g|": math.log2(abs(c.g)), "sieve_sum": c.sieve_sum, "threshold": round(c.threshold, 1),
                         "base part": _fmt_factors({j + 1: e for j, e in c.factors.items()} | ({0: 1} if c.g < 0 else {}), self.base),
                         "cofactor": c.cofactor, "cofactor bits": c.cofactor.bit_length(), "kind": c.kind,
                         "large primes": " ".join(f"{q}^{e}" if e > 1 else str(q) for q, e in (c.relation.large if c.relation else ())),
                         "y = step·a·x + b mod kn": c.relation.y if c.relation else None,
                         "y² ≡ product (mod kn)": c.relation.check(self.kn, self.base) if c.relation else None})
        kinds = pd.Series([c.kind.split(":")[0] for c in cands]).value_counts().to_dict() if cands else {}
        self._say(f"polynomial {sieve.t}: {len(cands)} survivors → {kinds}; held: {len(self.full)} full, {len(self.cycles.edges)} partial edges")
        df = pd.DataFrame(rows)
        if len(df):
            df["y = step·a·x + b mod kn"] = pd.Series([r["y = step·a·x + b mod kn"] for r in rows], dtype=object)
        return df

    def _absorb(self, r: Relation):
        if not r.large:
            self.full.append(r)
            return
        cyc = self.cycles.add(r)
        if cyc:
            from .engine import combine
            self.full.append(combine(cyc, self.kn))

    def relation(self, i: int):
        """Full relation i spelled out: y² ≡ (−1)^e · ∏ p^e · ∏ [large]^e (mod k·n), verified."""
        pd = _pd()
        if not self.full:
            raise IndexError("no full relation held yet: sieve more polynomials or call collect()")
        r = self.full[i]
        return pd.Series({"y": r.y, "right-hand side": _fmt_factors(r.exps, self.base, r.large),
                          "odd columns": r.odd_columns(), "large primes (even powers)": r.large,
                          "y² ≡ rhs (mod kn)": r.check(self.kn, self.base)}, name=f"relation {i}")

    # ------------------------------------------------------------------ 5. collect the rest
    @property
    def need(self) -> int:
        return len(self.base) + 1 + self.params.extra

    def collect(self, families: int | None = None, workers: int | None = None, budget: float | None = None):
        """Run the remaining families (from ``workers`` processes) until ``need`` full relations
        are held, or ``families`` more have been sieved, or ``budget`` seconds have passed. Partials
        go into the cycle forest; each closed cycle becomes one more full relation."""
        pd = _pd()
        if self.base is None:
            self.factor_base()
        w = self.workers if workers is None else workers
        stream = family_stream(self.kn, self.base, self.params, self.threshold, self.exact, self.settings, w, start=self.next_family)
        coll = relations(self.kn, self.base, self.params, self.need, stream, verbose=self.verbose, budget=budget,
                         family_limit=families, cycles=self.cycles, full=self.full)
        self.next_family += coll.families
        self.polys += coll.polys
        self.collection = coll
        self.keep = self.odd = self.deps = None
        return pd.Series({"families sieved": coll.families, "polynomials": coll.polys, "seconds": round(coll.seconds, 2),
                          "polynomials / s": round(coll.polys / max(coll.seconds, 1e-9)), "full relations": len(self.full),
                          "needed": self.need, "partial edges": len(self.cycles.edges), "cycles closed": self.cycles.closed,
                          "workers": w}, name="collection")

    def relations(self):
        """The full relations held: direct ones and those combined from cycles of partials."""
        pd = _pd()
        rows = [{"i": i, "y": r.y, "odd columns": len(r.odd_columns()), "nonzero columns": sum(1 for e in r.exps.values() if e),
                 "large primes": len(r.large), "from partials": bool(r.large)} for i, r in enumerate(self.full)]
        return pd.DataFrame(rows)

    def partials(self):
        """The partial relations as edges of the large-prime graph (1 = no second prime)."""
        pd = _pd()
        rows = []
        for i, r in enumerate(self.cycles.edges):
            odd = [q for q, e in r.large if e % 2 == 1]
            u, v = (1, odd[0]) if len(odd) == 1 else tuple(odd)
            rows.append({"edge": i, "prime u": u, "prime v": v, "y": r.y})
        return pd.DataFrame(rows)

    def cycle_summary(self):
        pd = _pd()
        g = self.cycles
        return pd.Series({"partial edges": len(g.edges), "vertices (large primes, incl. 1)": g.vertices,
                          "cycles closed → full relations": g.closed}, name="large-prime graph")

    # ------------------------------------------------------------------ 6. the matrix
    def matrix(self, dense: bool | None = None, max_dense: int = 400):
        """Rows = full relations, columns = sign and base primes, entry 1 where the exponent is
        odd. Rows alone in some column are pruned first (they can be in no dependency). Dense
        0/1 DataFrame for small cases, otherwise one row per relation listing its odd columns."""
        pd = _pd()
        cols = len(self.base) + 1
        if len(self.full) > self.need + 64:
            self._say(f"using the first {self.need + 64} of {len(self.full)} relations (more only cost elimination time)")
        self.odd = [r.odd_columns() for r in self.full[:self.need + 64]]
        self.keep = linalg.prune_singletons(self.odd, cols)
        kept = [self.odd[i] for i in self.keep]
        self._say(f"{len(self.full)} relations × {cols} columns; {len(self.keep)} rows after singleton pruning")
        if dense is None:
            dense = len(kept) <= max_dense and cols <= max_dense
        if dense:
            names = ["-1"] + [str(int(p)) for p in self.base.p]
            return pd.DataFrame(linalg.dense_matrix(kept, cols), index=self.keep, columns=names)
        return pd.DataFrame({"relation": self.keep, "odd columns": kept, "weight": [len(r) for r in kept]})

    def dependencies(self, max_deps: int = 64):
        """Row sets whose odd-exponent vectors XOR to zero (GF(2) Gaussian elimination): each is
        a product of relations with every exponent even, hence a congruence of squares."""
        pd = _pd()
        if self.keep is None:
            self.matrix(dense=False)
        cols = len(self.base) + 1
        st: dict = {}
        deps = linalg.dependencies([self.odd[i] for i in self.keep], cols, max_deps=max_deps,
                                   lanczos=self.settings.lanczos, use_numba=self.settings.use_numba, stats=st)
        self.deps = [[self.keep[i] for i in d] for d in deps]
        how = st.get("solver", "")
        if "iterations" in st:
            how += f", {st['iterations']} iterations"
        self._say(f"{len(self.deps)} dependencies by {how}")
        return pd.DataFrame({"dependency": range(len(self.deps)), "relations": [len(d) for d in self.deps], "rows": self.deps})

    # ------------------------------------------------------------------ 7. the square root
    def square_root(self, which: int = 0):
        """Dependency ``which``: X = ∏ y, Y = √(∏ right-hand sides) = ∏ p^(Σe/2), both mod k·n;
        X² ≡ Y² (mod n), so gcd(X − Y, n) is a factor unless X ≡ ±Y."""
        pd = _pd()
        if self.deps is None:
            self.dependencies()
        res = square_root(self.deps[which], self.full, self.base, self.kn, self.n)
        res["X mod n == ±Y mod n (trivial)"] = (res["X"] - res["Y"]) % self.n == 0 or (res["X"] + res["Y"]) % self.n == 0
        return pd.Series(res, name=f"dependency {which}")

    def factors(self):
        """Try the dependencies in order until one splits n; the two factors."""
        if self.deps is None:
            self.dependencies()
        for i in range(len(self.deps)):
            res = square_root(self.deps[i], self.full, self.base, self.kn, self.n)
            if res["factor"]:
                f = res["factor"]
                self._say(f"dependency {i} splits n: {f} × {self.n // f}")
                return [f, self.n // f]
        self._say("no dependency split n; collect more relations and try again")
        return []

    # ------------------------------------------------------------------ all at once
    def run(self):
        """Every stage in order, nothing kept back: the factors of n."""
        t = time.time()
        self.factor_base()
        self.collect()
        self.matrix(dense=False)
        self.dependencies()
        out = self.factors()
        self._say(f"{time.time() - t:.2f} s")
        return out
