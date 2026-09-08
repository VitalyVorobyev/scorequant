"""Information-preserving hard quantization for statistical inference."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

from ._errors import ContractError, RefusalError, ScoreQuantError
from .api import fit_quantizer, optimize_partition
from .artifact import Quantizer
from .certify import CertificationConfig, certify_partition
from .components import (
    LinearComponents,
    scores_from_components,
)
from .config import (
    DExchangeConfig,
    ExecutionConfig,
    KMeansConfig,
    MahalanobisLloydConfig,
    ScalarDPConfig,
    SoftVoronoiConfig,
)
from .criteria import DOptimality, NormalizedTrace, ProfiledDOptimality
from .information import (
    binned_fisher_information,
    efficient_score_bound,
    efficient_scores,
    fisher_information,
    fractional_fisher_information,
    information_report,
    profiled_information_report,
    retention_uncertainty,
)
from .partition import exchange_stability_report
from .providers import (
    CentralLogRatioScore,
    DensityRatioScore,
    LinearComponentScore,
    ScoreFunction,
    ScoreProvider,
)
from .ratios import (
    IntensityParameterization,
    MixtureParameterization,
    mixture_scores_from_ratios,
    ratio_closure_report,
    ratios_from_posteriors,
)
from .reports import RatioClosureReport, RetentionUncertainty
from .result import (
    EfficientScoreBound,
    GeometryReport,
    InformationReport,
    OptimizationTrace,
    PartitionCertificate,
    PartitionResult,
    ProfiledGeometryReport,
    ProfiledInformationReport,
    QuantizerResult,
    StabilityReport,
)
from .sources import (
    GaussLegendreConfig,
    IntegrationSource,
    ObservationSample,
    RatioProvenance,
    ScoreProvenance,
    ScoreSample,
    ScoreSchema,
)
from .transforms import FisherTransform
from .visualization import plot_information, plot_optimization, plot_partition, plot_summary

#: Installed distribution version. Resolved from package metadata so
#: ``pyproject.toml`` stays the single source of truth; a source tree that was
#: never installed reports ``"0.0.0.dev0"`` rather than failing to import.
try:
    __version__ = _distribution_version("scorequant")
except PackageNotFoundError:  # pragma: no cover - only hit in an uninstalled tree
    __version__ = "0.0.0.dev0"

__all__ = [
    "CentralLogRatioScore",
    "CertificationConfig",
    "ContractError",
    "DExchangeConfig",
    "ExecutionConfig",
    "DOptimality",
    "DensityRatioScore",
    "EfficientScoreBound",
    "FisherTransform",
    "GaussLegendreConfig",
    "GeometryReport",
    "InformationReport",
    "IntegrationSource",
    "IntensityParameterization",
    "KMeansConfig",
    "LinearComponentScore",
    "LinearComponents",
    "MahalanobisLloydConfig",
    "MixtureParameterization",
    "NormalizedTrace",
    "ObservationSample",
    "OptimizationTrace",
    "PartitionCertificate",
    "PartitionResult",
    "ProfiledDOptimality",
    "ProfiledGeometryReport",
    "ProfiledInformationReport",
    "Quantizer",
    "QuantizerResult",
    "RatioClosureReport",
    "RatioProvenance",
    "RefusalError",
    "RetentionUncertainty",
    "ScalarDPConfig",
    "ScoreFunction",
    "ScoreProvenance",
    "ScoreProvider",
    "ScoreQuantError",
    "ScoreSample",
    "ScoreSchema",
    "SoftVoronoiConfig",
    "StabilityReport",
    "binned_fisher_information",
    "certify_partition",
    "efficient_score_bound",
    "efficient_scores",
    "exchange_stability_report",
    "fisher_information",
    "fit_quantizer",
    "fractional_fisher_information",
    "information_report",
    "mixture_scores_from_ratios",
    "optimize_partition",
    "plot_information",
    "plot_optimization",
    "plot_partition",
    "plot_summary",
    "profiled_information_report",
    "ratio_closure_report",
    "ratios_from_posteriors",
    "retention_uncertainty",
    "scores_from_components",
]
