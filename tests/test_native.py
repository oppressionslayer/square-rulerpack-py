"""The Rust extension (square-rulerpack-native) against the Python engine. Skipped when not installed."""
import pytest

import square_rulerpack_py as sq
from square_rulerpack_py import native

pytestmark = pytest.mark.skipif(not native.HAVE_NATIVE, reason="square-rulerpack-native is not installed")

P30 = 618970019642690137449562111          # 2^89 - 1
Q30 = 162259276829213363391578010288127    # 2^107 - 1
N40 = 1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139
SEMIPRIMES = {
    30: 100000000000000000039 * 1000000000000000000117,                      # 21 + 22 digits
    40: 1000000000000000000000000000000000000057 * 1000000007,
    50: 4837594595591756100052399 * 6679717452512069423002243,
}


def test_small_inputs_match_python():
    for n in [1, 2, 3, 4, 12, 221, 1009, 65536, 3 ** 20, 1009732533765211, 2 ** 67 - 1]:
        assert native.factor_completely(n) == sq.factor_completely(n, engine="python"), n


def test_semiprimes_match_python_and_threads_agree():
    for digits, n in SEMIPRIMES.items():
        got1 = native.factor_completely(n, threads=1)
        got4 = native.factor_completely(n, threads=4)
        assert got1 == got4 == sq.factor_completely(n, engine="python", workers=2), digits
        prod = 1
        for f in got1:
            prod *= f
        assert prod == n and all(native.is_prime(f) for f in got1)


def test_is_prime():
    assert native.is_prime(2 ** 61 - 1) and native.is_prime(P30) and native.is_prime(Q30)
    assert not native.is_prime(2 ** 61 + 1) and not native.is_prime(P30 * Q30) and not native.is_prime(1)


def test_engine_dispatch_and_settings_env():
    n = SEMIPRIMES[40]
    assert sq.factor_completely(n, engine="native", workers=2) == sq.factor_completely(n, engine="python")
    assert sq.factor_completely(n, engine="auto", workers=2) == sq.factor_completely(n, engine="python")
    with pytest.raises(ValueError):
        sq.factor_completely(n, engine="rust")
    import os
    st = sq.Settings(q2=False, small=256)
    assert "SIQS_Q2" not in os.environ
    with native.settings_env(st, verbose=False):
        assert os.environ["SIQS_Q2"] == "0" and os.environ["SIQS_SMALL"] == "256" and os.environ["SIQS_QUIET"] == "1"
    assert "SIQS_Q2" not in os.environ and "SIQS_SMALL" not in os.environ
    assert native.factor_completely(n, settings=st, threads=2) == sq.factor_completely(n, engine="python")


def test_progress_callback_sees_stages():
    seen = []
    native.factor_completely(SEMIPRIMES[50], threads=2, progress=lambda *a: seen.append(a))
    stages = {s[0] for s in seen}
    assert 1 in stages and 2 in stages and 3 in stages, stages
    assert all(len(s) == 5 for s in seen)


def test_sieve_factor_only():
    f = native.factor(SEMIPRIMES[30], threads=2)
    assert f in (100000000000000000039, 1000000000000000000117)
    assert native.factor(2 ** 61 - 1) is None
    with pytest.raises(ValueError):
        native.factor_completely(0)
