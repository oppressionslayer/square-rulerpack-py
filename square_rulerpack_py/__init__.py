"""square_rulerpack_py: the self-initialising quadratic sieve of square-rulerpack 0.2.0 in Python.

Two ways in:

    >>> import square_rulerpack_py as sq
    >>> sq.factor_completely(1009732533765211)
    [11344301, 89007911]

    >>> w = sq.Walkthrough(1009732533765211)      # step by step, every stage as a DataFrame
    >>> w.multiplier(); w.factor_base(); w.family(); w.sieve(0); w.trial_divide(0); ...

    >>> sq.factor_completely(n, workers=8)             # the Rust engine when square-rulerpack[native] is installed

    >>> sq.factor_walkthrough()                        # the tiny exact N = 221 lesson, by hand
    FactorWalkthrough(n=221, samples=5, relations=3, rank=2, nullity=1, split=(13, 17))
"""
from .params import Params, Row, YAFU_DIRECT
from .engine import (Settings, FactorBase, Family, Relation, Candidate, FamilySieve, Cycles,
                     multiplier, multiplier_scores, factor_base, choose_a, threshold_bits,
                     relations, factor, factor_with, factor_completely, square_root, HAVE_NUMBA)
from .workers import default_workers
from . import native
from .native import HAVE_NATIVE
from .walkthrough import Walkthrough
from . import lessons
from .lessons import (prime_seeds, polynomial_roots, LessonWindow, square_window, siqs_window, sieve_walk,
                      construct_polynomial, FactorWalkthrough, factor_walkthrough, RelationStudy, relation_study,
                      Capture, engine_polynomials)

__version__ = "0.3.0"
__all__ = ["Params", "Row", "YAFU_DIRECT", "Settings", "FactorBase", "Family", "Relation", "Candidate",
           "FamilySieve", "Cycles", "multiplier", "multiplier_scores", "factor_base", "choose_a",
           "threshold_bits", "relations", "factor", "factor_with", "factor_completely", "square_root",
           "default_workers", "Walkthrough", "HAVE_NUMBA", "HAVE_NATIVE", "native", "__version__", "lessons", "prime_seeds",
           "polynomial_roots", "LessonWindow", "square_window", "siqs_window", "sieve_walk", "construct_polynomial",
           "FactorWalkthrough", "factor_walkthrough", "RelationStudy", "relation_study", "Capture", "engine_polynomials"]
