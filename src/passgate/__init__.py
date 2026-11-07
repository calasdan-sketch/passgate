"""passgate - statistical pass^N reliability gate for evals."""

from .cluster import Item, ItemsResult, evaluate_items, pass_at_k, pass_hat_k
from .gate import (
    Comparison,
    GateResult,
    PairedComparison,
    Round,
    compare,
    compare_paired,
    evaluate_rounds,
)
from .multi import Case, CasesResult, compare_cases
from .noninf import NonInferiority, non_inferiority, non_inferiority_paired
from .power import oc_curve, pass_probability, plan_power
from .sequential import SequentialResult, anytime_lower, run_sequential
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

__version__ = "0.3.0"
__all__ = [
    "Item",
    "ItemsResult",
    "evaluate_items",
    "pass_at_k",
    "pass_hat_k",
    "Case",
    "CasesResult",
    "compare_cases",
    "NonInferiority",
    "non_inferiority",
    "non_inferiority_paired",
    "oc_curve",
    "pass_probability",
    "plan_power",
    "SequentialResult",
    "anytime_lower",
    "run_sequential",
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
