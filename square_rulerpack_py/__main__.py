"""python -m square_rulerpack_py N [--workers W] [--engine auto|native|python] [--plain] [--no-q2] [--no-dlp] [--quiet]"""
from __future__ import annotations

import argparse
import sys
import time

from .arith import is_probable_prime
from .engine import Settings, factor_completely
from .workers import default_workers


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="square_rulerpack_py", description="Factor N with the self-initialising quadratic sieve.")
    ap.add_argument("n", help="a positive decimal integer")
    ap.add_argument("--workers", "-w", type=int, default=default_workers(), help="sieve processes (default: half the hardware threads)")
    ap.add_argument("--plain", action="store_true", help="plain threshold instead of the exact one")
    ap.add_argument("--no-q2", action="store_true", help="Q1 polynomials even when k·n ≡ 1 (mod 8)")
    ap.add_argument("--no-dlp", action="store_true", help="single large prime only")
    ap.add_argument("--no-numba", action="store_true", help="numpy sieve kernel even if numba is installed")
    ap.add_argument("--engine", choices=["auto", "native", "python"], default="auto",
                    help="auto: the Rust extension when installed, else this Python port")
    ap.add_argument("--quiet", "-q", action="store_true", help="no progress on stderr")
    a = ap.parse_args(argv)
    if not a.n.isdigit() or int(a.n) == 0:
        print("The input must be a positive decimal integer.", file=sys.stderr)
        return 2
    n = int(a.n)
    st = Settings(exact=False if a.plain else None, q2=not a.no_q2, dlp=not a.no_dlp, use_numba=not a.no_numba)
    t = time.time()
    if a.engine != "native" and not a.quiet:
        from .native import HAVE_NATIVE
        if a.engine == "python" or not HAVE_NATIVE:
            print("  engine: python" + ("" if HAVE_NATIVE else " (pip install square-rulerpack[native] for the Rust engine)"), file=sys.stderr)
    fs = factor_completely(n, workers=a.workers, settings=st, verbose=not a.quiet, engine=a.engine)
    print(f"{n} = {' × '.join(map(str, fs)) if fs else 1}")
    prod = 1
    for f in fs:
        prod *= f
    ok = prod == n
    prime = all(is_probable_prime(f) for f in fs)
    print(f"{len(fs)} factors, product checks: {str(ok).lower()}, all probable prime: {str(prime).lower()}, {time.time() - t:.2f} s")
    return 0 if ok and prime else 1


if __name__ == "__main__":
    sys.exit(main())
