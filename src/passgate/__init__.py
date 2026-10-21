"""passgate — statistical pass^N reliability gate for evals."""

from .gate import Round, GateResult, evaluate_rounds, compare
from .stats import wilson_lower, wilson_interval, two_proportion_z, fisher_exact_2x2

__version__ = "0.1.0"
__all__ = [
    "Round",
    "GateResult",
    "evaluate_rounds",
    "compare",
    "wilson_lower",
    "wilson_interval",
    "two_proportion_z",
    "fisher_exact_2x2",
    "__version__",
]
