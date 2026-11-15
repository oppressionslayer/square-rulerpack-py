"""The Rust engine as an extension module (``pip install square-rulerpack[native]``).

``square_rulerpack_native`` is Lars Rocha's square-rulerpack 0.2.0 crate compiled with PyO3. It
reads the same switches the Rust CLI reads from the environment, so a ``Settings`` is bridged by
setting ``SIQS_*`` variables for the duration of the call.
"""
from __future__ import annotations

import contextlib
import os
import sys
from typing import Callable

try:
    import square_rulerpack_native as _native
except ImportError:          # pragma: no cover - depends on the environment
    _native = None

HAVE_NATIVE = _native is not None
INSTALL_HINT = "the Rust engine is not installed: pip install square-rulerpack[native]"

ProgressHook = Callable[[int, int, int, int, int], None]


def load():
    """The extension module, or None when it is not installed."""
    return _native


def require():
    if _native is None:
        raise ImportError(INSTALL_HINT)
    return _native


def version() -> str | None:
    return None if _native is None else _native.version()


@contextlib.contextmanager
def settings_env(settings=None, verbose: bool = False):
    """Export a ``Settings`` (and ``verbose``) as the engine's ``SIQS_*`` variables, then restore."""
    wanted: dict[str, str | None] = {}
    if settings is not None:
        if settings.exact is not None:
            wanted["SIQS_EXACT"] = "1" if settings.exact else "0"
        if settings.small is not None:
            wanted["SIQS_SMALL"] = str(int(settings.small))
        wanted["SIQS_Q2"] = "1" if settings.q2 else "0"
        wanted["SIQS_DLP"] = "1" if settings.dlp else "0"
        wanted["SIQS_CLOSE_A"] = "1" if settings.close_a else "0"
        if settings.lanczos is not None:
            wanted["SIQS_LANCZOS"] = "1" if settings.lanczos else "0"
        wanted["SIQS_ECAP"] = repr(float(settings.ecap))
        wanted["SIQS_MARGIN"] = repr(float(settings.margin))
        if settings.multiplier:
            wanted["SIQS_MULT"] = str(int(settings.multiplier))
    wanted["SIQS_PROGRESS"] = "1" if verbose else None
    wanted["SIQS_QUIET"] = None if verbose else "1"
    saved = {k: os.environ.get(k) for k in wanted}
    try:
        for k, v in wanted.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def factor_completely(n: int, threads: int | None = None, settings=None, verbose: bool = False,
                      progress: ProgressHook | None = None) -> list[int]:
    """Sorted prime factors with multiplicity, from the Rust engine on ``threads`` rayon threads."""
    m = require()
    n = int(n)
    if n < 1:
        raise ValueError("n must be a positive integer")
    if verbose:
        print(f"  engine: rust (square-rulerpack-native {m.version()}, engine {m.engine_version()}, "
              f"{'AVX2' if m.have_avx2() else 'portable'}), {threads or 'default'} threads", file=sys.stderr)
    with settings_env(settings, verbose):
        return [int(f) for f in m.factor_completely(n, threads=threads, progress=progress)]


def factor(n: int, threads: int | None = None, settings=None, verbose: bool = False,
           progress: ProgressHook | None = None) -> int | None:
    """One nontrivial factor from the sieve, or None."""
    m = require()
    n = int(n)
    if n < 1:
        raise ValueError("n must be a positive integer")
    with settings_env(settings, verbose):
        f = m.factor(n, threads=threads, progress=progress)
    return None if f is None else int(f)


def is_prime(n: int) -> bool:
    return bool(require().is_prime(int(n)))
