"""passgate - statistical pass^N reliability gate for evals."""

from .gate import (
    Comparison,
    GateResult,
    PairedComparison,
    Round,
    compare,
    compare_paired,
    evaluate_rounds,
)
from .stats import (
    clopper_pearson_interval,
    fisher_exact_2x2,
    lower_bound,
    mcnemar_exact,
    min_passes,
    trials_needed,
    two_proportion_z,
    wilson_interval,
    wilson_lower,
)

__version__ = "0.2.0"
__all__ = [
    "Round",
    "GateResult",
    "Comparison",
    "PairedComparison",
    "evaluate_rounds",
    "compare",
    "compare_paired",
    "trials_needed",
    "min_passes",
    "lower_bound",
    "wilson_lower",
    "wilson_interval",
    "clopper_pearson_interval",
    "two_proportion_z",
    "fisher_exact_2x2",
    "mcnemar_exact",
    "__version__",
]
