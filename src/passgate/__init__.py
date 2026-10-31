"""passgate - statistical pass^N reliability gate for evals."""

from .gate import GateResult, Round, compare, evaluate_rounds
from .stats import fisher_exact_2x2, two_proportion_z, wilson_interval, wilson_lower

__version__ = "0.1.1"
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
