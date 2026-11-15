"""Sieve parameters by input size, ported from ``params.rs`` and ``Params::for_bits`` in
``siqs.rs`` of square-rulerpack 0.2.0.

Two tables: YAFU 3.x's direct-sieve rows (used from 198 bits ≈ 60 digits, interpolated the
way YAFU's ``get_params`` does), and the hand rows below that from the rulerpack-rs sweeps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

__all__ = ["Row", "YAFU_DIRECT", "for_bits", "Params", "digits_of_bits"]


@dataclass(frozen=True)
class Row:
    bits: int        # input size in bits
    base: int        # factor-base primes
    lp_mult: int     # large-prime bound as a multiple of the largest base prime
    blocks: int      # 32 KB sieve blocks per side (half interval = blocks · 32768)
    tf_bits: int     # trial-division cutoff in bits (0 = formula)
    dlp_exp: float   # cofactor bound = (lp_mult · pmax)^dlp_exp (0 = single large prime)


# YAFU `siqs_table_direct` (plain and AVX2 builds), 60-110 digits.
YAFU_DIRECT = [
    Row(198, 5472, 180, 1, 66, 0.0),
    Row(215, 10096, 225, 1, 73, 1.62),
    Row(232, 8208, 225, 1, 68, 1.62),
    Row(248, 8416, 300, 1, 72, 1.72),
    Row(265, 11856, 300, 1, 77, 1.76),
    Row(281, 16128, 300, 2, 85, 1.74),
    Row(298, 35232, 350, 2, 89, 1.74),
    Row(310, 56071, 350, 3, 94, 1.74),
    Row(320, 71745, 350, 3, 97, 1.78),
    Row(330, 87500, 350, 4, 102, 1.8),
    Row(340, 102120, 350, 6, 104, 1.8),
    Row(350, 119102, 350, 8, 106, 1.8),
    Row(360, 132742, 350, 8, 111, 1.8),
    Row(370, 239500, 150, 16, 0, 1.9),
]


def _round16(b: int) -> int:
    return -(-b // 16) * 16


def for_bits(table: list[Row], bits: int) -> Row:
    """The row for ``bits``, interpolating between bracketing rows as YAFU does: base linear
    in bits, lp_mult and blocks the rounded average, tf_bits and dlp_exp linear when both
    rows have them. Below the first row the first row; above the last, the last row's slope."""
    first, last = table[0], table[-1]
    if bits <= first.bits:
        return Row(bits, first.base, first.lp_mult, first.blocks, first.tf_bits, first.dlp_exp)
    if bits >= last.bits:
        prev = table[-2]
        slope = (last.base - prev.base) / (last.bits - prev.bits)
        base = _round16(round(last.base + slope * (bits - last.bits)))
        return Row(bits, base, last.lp_mult, last.blocks, last.tf_bits, last.dlp_exp)
    for r in table:
        if r.bits == bits:
            return r
    i = next(i for i, r in enumerate(table) if r.bits > bits)
    lo, hi = table[i - 1], table[i]
    t = (bits - lo.bits) / (hi.bits - lo.bits)
    lerp = lambda a, b: a + (b - a) * t
    return Row(
        bits,
        _round16(round(lerp(lo.base, hi.base))),
        round((lo.lp_mult + hi.lp_mult) / 2),
        round((lo.blocks + hi.blocks) / 2),
        round(lerp(lo.tf_bits, hi.tf_bits)) if lo.tf_bits > 0 and hi.tf_bits > 0 else 0,
        lerp(lo.dlp_exp, hi.dlp_exp) if lo.dlp_exp > 0 and hi.dlp_exp > 0 else 0.0,
    )


def digits_of_bits(bits: int) -> int:
    return math.ceil(bits * math.log10(2))


@dataclass
class Params:
    """The knobs of one factorisation.

    base      factor-base size (number of primes)
    half      half the sieve interval: x runs over [−half, half]
    large     single-large-prime bound as a multiple of the largest base prime
    slack     threshold slack in units of log2(pmax) (1.3 on the YAFU rows, 0.7 on the hand rows)
    extra     relations to collect past the factor-base size
    dlp_exp   double-large-prime exponent: cofactors up to (large·pmax)^dlp_exp are split
    dlp       whether two large primes are allowed at all
    """
    base: int
    half: int
    large: int
    slack: float
    extra: int
    dlp_exp: float = 1.8
    dlp: bool = True

    @staticmethod
    def for_n(n: int) -> "Params":
        return Params.for_bits(n.bit_length(), len(str(n)))

    @staticmethod
    def for_digits(d: int) -> "Params":
        return Params.for_bits(round(d / math.log10(2)), d)

    @staticmethod
    def for_bits(bits: int, d: int) -> "Params":
        if bits >= YAFU_DIRECT[0].bits:
            r = for_bits(YAFU_DIRECT, bits)
            # slack 0.9, not YAFU's 1.3: the Python candidate stage is the long pole, so fewer
            # survivors win (60 digits, one worker, 2026-11-14: 1.3 → 19.5 s, 1.1 → 16.9, 0.9 → 14.6, 0.7 → 16.3)
            return Params(base=r.base, half=r.blocks * 32768, large=r.lp_mult, slack=0.9, extra=64,
                          dlp_exp=r.dlp_exp if r.dlp_exp > 0 else 1.8)
        # the hand rows below 60 digits (rulerpack-rs sweeps), copied from Params::for_bits
        rows = [
            (30, 150, 1 << 13, 30, 24), (40, 400, 1 << 14, 40, 32), (50, 1100, 1 << 15, 50, 48),
            (56, 1800, 1 << 15, 60, 64), (62, 3000, 1 << 16, 60, 64), (67, 4500, 1 << 16, 80, 80),
            (72, 15000, 1 << 16, 100, 96), (77, 30000, 1 << 17, 120, 120), (84, 50000, 1 << 17, 100, 150),
            (92, 80000, 1 << 17, 200, 200), (97, 120000, 1 << 17, 250, 250),
        ]
        for top, base, half, large, extra in rows:
            if d <= top:
                return Params(base=base, half=half, large=large, slack=0.7, extra=extra)
        return Params(base=200000, half=1 << 17, large=300, slack=0.7, extra=300)

    def as_dict(self) -> dict:
        return asdict(self)
