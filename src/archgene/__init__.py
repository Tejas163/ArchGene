"""ArchGene - Self-healing multi-agent cognitive architecture evaluation system."""

__version__ = "0.3.0"

from .gene_schema import Gene, ActivationType, AttentionType, PoolingType, ArchitectureFamily, QuantizationType
from .verifier import Verifier, VerificationResult
from .evaluation import Evaluator, EvaluationScore
from .cost_estimator import CostEstimator, CostEstimates
from .model_zoo import ModelZoo

__all__ = [
    "__version__",
    "Gene",
    "ActivationType",
    "AttentionType",
    "PoolingType",
    "ArchitectureFamily",
    "QuantizationType",
    "Verifier",
    "VerificationResult",
    "Evaluator",
    "EvaluationScore",
    "CostEstimator",
    "CostEstimates",
    "ModelZoo",
]
