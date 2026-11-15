# square-rulerpack-py

The self-initialising quadratic sieve with double large primes of
[square-rulerpack](https://github.com/oppressionslayer/square-rulerpack-rs) (Rust, by Lars Rocha),
ported stage by stage to pure Python so that every step can be run by hand in IPython and looked
at as a DataFrame or a numpy array. Same parameter tables, same Knuth–Schroeppel multiplier,
same family selection (family *i* picks the same primes for *a* as the Rust engine's family *i*),
same Gray-code polynomial switching, same Q1/Q2 forms, same exact and plain thresholds, same
large-prime cycle finder, same block Lanczos, same square root.

This is the Python-only edition: no Rust, no compiled extension, nothing to build. The hot loops
(the sieve, the root test of the survivors, the 64-bit cofactor arithmetic, the Lanczos kernels)
are numba when it is installed and numpy otherwise, and relation collection runs across
processes. It is fast for Python: the 60-digit number in `TIMINGS.md` takes 12.8 s on one worker,
where sympy's own `qs` had not finished after 15 minutes on the same machine. An 80-digit
semiprime takes about three minutes on 8 workers.

If you want the compiled engine as well (the same algorithm, 100 digits in about nine minutes on
8 threads), that is the sister package [square-rulerpack](https://pypi.org/project/square-rulerpack/)
with its `[native]` extra. That package installs the module `square_rulerpack`; this one installs
`square_rulerpack_py`, so the two can live side by side in one environment.

**Give me three months with a problem your quant team keeps coming back to.**

I’m Lars Rocha. My work spans integer factorization, Sudoku constraint solving, and structured random forests. I turn mathematical observations into working software and experiments others can inspect.

I’m interested in a paid research collaboration exploring better representations, useful patterns, and computational shortcuts. We choose a focused problem, establish a baseline, and evaluate the results on held-out data.

Your team gets working code, reproducible experiments, and documented findings it can build on.

**You’ve seen what I build independently. Let’s explore what I could build alongside your team.**


## Install

```sh
pip install square-rulerpack-py           # pure Python
pip install 'square-rulerpack-py[fast]'   # plus numba: the compiled kernels, where the speed is
pip install -e .                          # or from this checkout
python -m square_rulerpack_py 1009732533765211 --workers 8
```

```
1009732533765211 = 11344301 × 89007911
2 factors, product checks: true, all probable prime: true, 0.34 s
```

## Factor a number

```python
import square_rulerpack_py as sq
sq.factor_completely(1009732533765211)                 # [11344301, 89007911]
sq.factor_completely(n, workers=8, verbose=True)       # progress on stderr
sq.factor(n, workers=4)                                # one nontrivial factor or None
```

`Settings` carries the knobs the Rust engine reads from the environment, as a dataclass:

| Settings field | Rust variable | default |
|---|---|---|
| `exact`   | `SIQS_EXACT` | exact threshold from 215 bits (about 65 digits), plain below |
| `small`   | `SIQS_SMALL` | 512 unsieved primes under the exact threshold, 30 under the plain |
| `q2`      | `SIQS_Q2`    | Q2 polynomials when k·n ≡ 1 (mod 8) |
| `dlp`     | `SIQS_DLP`   | two large primes |
| `close_a` | `SIQS_CLOSE_A` | repair *a* toward √(2kn)/M |
| `lanczos` | `SIQS_LANCZOS` | block Lanczos from 6,000 matrix rows, dense elimination below |
| `ecap`, `margin` | `SIQS_ECAP`, `SIQS_MARGIN` | 32, 1 |
| `multiplier` | `SIQS_MULT` | choose by score |

```python
sq.factor_completely(n, settings=sq.Settings(exact=False, q2=False))
```

### With sympy

`sympy.factorint` has Pollard rho, p−1 and ECM but no quadratic sieve, so its time grows with the
size of the *smaller factor*; the sieve's grows with the size of *n* and does not care how the
factors are split. sympy asks `factor_cache.get_external` for every cofactor it cannot place:

```python
import square_rulerpack_py as sq
from sympy import factor_cache, factorint

def siqs(n):
    """Return the prime factors of n, or None to let sympy handle it."""
    if n < 10**40:
        return None
    return sq.factor_completely(n, workers=8)

factor_cache.get_external = siqs
factorint(50726044447758565750355444206008211963939957754977146054012847169421070122136537)
# {6245169037808927489391096084448856781527: 1, 8122445387892243915736861382463382759631: 1}
```

## Walk through a factorisation by hand

`Walkthrough` runs the stages one at a time. Each method prints a one-line summary on stderr
and returns the stage's data; the object keeps everything (`w.k`, `w.kn`, `w.base`, `w.fam`,
`w.fs`, `w.full`, `w.cycles`, `w.deps` …) for further digging. `WALKTHROUGH.md` is a full
session on a 40-digit number; `examples_walkthrough.py` reproduces it.

```python
import square_rulerpack_py as sq
w = sq.Walkthrough(5857505067604849028619727330686829482307, workers=4)
w.summary()          # digits, bits, trial division, square and primality checks, parameters
w.multiplier()       # Knuth–Schroeppel scores for every candidate k; the chosen one
fb = w.factor_base() # DataFrame: p, sqrt(kn) mod p, both roots, log2 p, byte log, sieved?
w.family()           # a = q1 q2 … q_s, the B_l, how many polynomials (2^(s-1))
w.polynomials()      # every b of the family in Gray-code order, lin and c of g(x)
w.roots(0)           # the two roots of g mod each prime, their sieve starts, g(root) mod p == 0
s = w.sieve(0)       # s.acc: the int16 sieve array over x in [-M, M]; s.hits: the survivors
s.table(20)          # positions by sieve sum: x, g(x), log2|g|, sum, threshold, survives
w.trial_divide(0)    # each survivor: factor-base part, cofactor, kind (full / one-large /
                     #   two-large / drop: why), y, and the check y² ≡ product (mod kn)
w.collect()          # the rest with worker processes until enough relations
w.relations(); w.partials(); w.cycle_summary(); w.relation(0)
w.matrix()           # 0/1 DataFrame (odd exponents) for small cases, sparse rows otherwise
w.dependencies()     # row sets that XOR to zero; says whether Lanczos or elimination found them
w.square_root(0)     # X, Y, X² ≡ Y² (mod kn), gcd(X − Y, n), gcd(X + Y, n), the factor
w.factors()          # try the dependencies in order
```

Or `w.run()` for everything at once.

What the stages are, in the engine's own words (module docstrings say more):

1. **multiplier** — pick k so that k·n is a square modulo many small primes.
2. **factor base** — the first `base` primes with (k·n / p) = 1, each with √(k·n) mod p.
3. **family** — a = q₁⋯q_s from the primes just above 1000 with a ≈ √(2kn)/M; B_l with
   B_l ≡ √kn (mod q_l) and ≡ 0 modulo the other q's; every b = Σ ±B_l gives b² ≡ kn (mod a).
4. **polynomial** — g(x) = a·x² + 2b·x + (b² − kn)/a (Q1) or a·x² + b·x + (b² − kn)/(4a)
   (Q2, for kn ≡ 1 mod 8); (a·x + b)² ≡ a·g(x) (mod kn), so a smooth g(x) is a relation.
   Consecutive b's differ by one B_l (Gray code), so every root moves by a precomputed delta.
5. **sieve** — each prime adds its byte log2 p at x ≡ root (mod p) along the interval; a
   position whose sum reaches the threshold is a survivor. The exact threshold compares each
   survivor with its own log2|g(x)| and counts the unsieved small primes exactly.
6. **trial division** — p | g(x) ⇔ x ≡ root (mod p), so the dividing primes are read off the
   roots by resieving them over the survivors; the cofactor is 1 (full), one prime below the
   bound, or two (split by Pollard–Brent on 64-bit integers).
7. **cycles** — partials are edges between large primes; a cycle multiplies to a full relation.
8. **matrix** — odd exponents over GF(2); singleton pruning; dense elimination for dependencies
   below 6,000 rows, block Lanczos (Montgomery 1995, the engine's own `rulerpack::lanczos` ported)
   above it. Every dependency is checked exactly before it is used.
9. **square root** — X = ∏ y, Y = ∏ p^(Σe/2); gcd(X − Y, n).

## Small exact lessons (`square_rulerpack_py.lessons`)

For a class that wants to see the whole idea on numbers that fit on a blackboard, before the
real sieve. Everything is plain Python integers over a handful of samples; no threshold, no
large primes, no engine. Carried over from Lars Rocha's `square_research` package.

```python
import square_rulerpack_py as sq

p = sq.construct_polynomial(221, [5, 7])      # a = 35; every CRT term of b; (c, 2b, a) = (1, 32, 35)
p["crt_terms"]                                # per prime: roots of 221 mod p, chosen root, a/p, inverse, term

w = sq.siqs_window(p["k"], p["a"], p["b"], start=-2, width=5, seeds=[2, 5, 7, 11])
w.dataframe()                                 # x, root = a·x + b, Q(x), root² − 221 = a·Q(x), factors, smooth
sq.sieve_walk(w)                              # the sieve by hand: roots of Q mod p and the positions they predict

lesson = sq.factor_walkthrough(w)             # FactorWalkthrough(n=221, relations=3, rank=2, nullity=1, split=(13, 17))
lesson.matrix()                               # parity matrix: rows = relations, columns = (−1, 2, 5, 7, 11)
lesson.dataframe("elimination")               # the Gaussian elimination, row by row
lesson.check_dependency([1, 2])               # 4900 = 70², x = 138, gcd(138 − 70, 221) = 17, gcd(138 + 70, 221) = 13

sq.relation_study()                           # all four CRT root choices of a = 35, left/centre/right windows
sq.factor_walkthrough()                       # the same N = 221 lesson with no arguments
```

The bridge to the real engine: `engine_polynomials` asks this package's own selector (the same
families as the Rust engine) for a few polynomials and turns them into such windows.

```python
cap = sq.engine_polynomials(1009732533765211, base=64, half=128, families=8, per_family=4)
cap.dataframe()                               # family, index, a, b, c, a_primes, coefficients, identity_valid
windows = [cap.window(i, start=-128, width=257) for i in range(16)]
sq.factor_walkthrough(windows, original_n=cap.n).split      # (11344301, 89007911)
```

Also there: `square_window(n)` for x² − n, `polynomial_roots`, `sqrt_mod` (all roots), `prime_seeds`,
`survivor_mask`, `dependency`, and `LessonWindow.export` for JSON.

## Speed

Measured on an i9-11900H (8 cores), numba kernels, verified factorisations; the full story with
profiles is in `TIMINGS.md`.

| digits | workers | time | note |
|---|---|---|---|
| 40 | 1 | 0.2 s | |
| 50 | 1 | 1.4 s | |
| 60 (199 bits) | 1 | 12.8 s | YAFU table row; 144.6 s before the 2026-11-14 work |
| 70 | 8 | 33–43 s | three runs, the spread is thermal |
| 80 | 8 | 190 s | matrix 4.1 s by Lanczos, 53 s by dense elimination |

Where the time went before and after: the sieve itself was never the cost in Python (about 1 %).
The survivors were: each one root-tested against the whole factor base and its cofactor
classified with big-integer arithmetic. Resieving the survivors, 64-bit Miller–Rabin and
Pollard–Brent in numba, a smaller slack, and block Lanczos for the matrix are the four changes,
all derived from the published algorithms (Rabin 1980, Brent 1980, Montgomery 1995) and the
engine's own code; nothing is ported from any other siever.

Above 80 digits this port works but has not been timed here; 90 digits is the open item for a
bigger machine. For 100 digits use the Rust engine.

## The XOR build system (`square_rulerpack_py.xor_table`)

Lars Rocha's prefix-XOR Cayley table and the odd-square rulers it is read against, as a module
with tests (`tests/test_xor_table.py`). `buildsystem(2*primes)` is the table `a[i] ^ a[j]` of the
anchor row `a` (exclusive prefix XOR of the doubled primes); `table_with_ruler(2*primes,
odd_squares(n))` puts the odd squares on the diagonal and makes every row a running XOR cipher
whose key is the squares and whose message is the primes; `decode_row` reads the primes back
from any row; `anchor_walk` is column 0, the walk 1, 5, 3, 9, 7, 17, 11, 41, … where consecutive
entries XOR to 2p and which stays within 8 bits for the first 26 primes. `lattice(anchor)` is
`(1+4k)² − anchor`, the value set of the sieve polynomial x² − n at n = anchor over the odd x,
and `lattice_divisors` is the factor-base rule that says which primes can divide it.

The module docstring records what was measured on 2026-11-14: the ten primes 19 through 59 XOR
to zero (the walk returns to 41), the walk's extra revisits come from the primes' slow monotone
growth rather than from primality, and the raw adjacent XORs of row 0 hit twice-a-prime at the
random rate. The Gray-code polynomial walk of the SIQS engine is the same object: a reversible
path where every step is one small flip and the state never grows.

## Files

- `square_rulerpack_py/arith.py` — primes, Legendre, Tonelli–Shanks, Miller–Rabin, Pollard–Brent, xorshift64.
- `square_rulerpack_py/fastarith.py` — Miller–Rabin and Pollard–Brent on 64-bit machine integers (numba)
  for the double-large-prime cofactors.
- `square_rulerpack_py/params.py` — the YAFU direct-sieve table and the hand rows, `Params.for_n`.
- `square_rulerpack_py/engine.py` — multiplier, factor base, families, `FamilySieve` (sieve, scan,
  resieve, candidates), `Cycles`, `relations`, `square_root`, `factor_completely`.
- `square_rulerpack_py/linalg.py` — singleton pruning, bit-packed Gaussian elimination, and the dispatch to
  block Lanczos from `LANCZOS_FROM` (6,000) rows.
- `square_rulerpack_py/lanczos.py` — block Lanczos over GF(2) in numpy `uint64` blocks with numba kernels.
- `square_rulerpack_py/workers.py` — the process pool; families merged in index order, so the
  relation set is the same at any worker count.
- `square_rulerpack_py/walkthrough.py` — `Walkthrough` and `SieveResult`.
- `square_rulerpack_py/lessons.py` — `construct_polynomial`, `LessonWindow`, `factor_walkthrough`,
  `relation_study`, `engine_polynomials`.
- `square_rulerpack_py/xor_table.py` — the XOR build system and the odd-square rulers.
- `square_rulerpack_py/native.py` — the optional bridge to the compiled engine of the sister package;
  inert here unless `square-rulerpack-native` happens to be installed.
- `tests/` — `pytest tests` (33 tests, under a minute; the native tests skip without the extension).

## Credits

Lars Rocha: the engine, the XOR build system, the lessons, the direction of every measurement.
GPT and Claude: the ports, kernels and profiles recorded in `TIMINGS.md`. YAFU (Ben Buhrow) and
msieve: the parameter table and the techniques the Rust engine names in its own README.

## License

MIT, as the Rust crate.
