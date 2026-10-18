"""Information-preserving hard quantization for statistical inference.

The top level holds what a first fit needs. Public submodules hold the rest:
:mod:`scorequant.ratios` (density-ratio providers), :mod:`scorequant.information`
(the information algebra), :mod:`scorequant.certify` (certificates),
:mod:`scorequant.results` (result and report types) and
:mod:`scorequant.plotting` (optional figures, outside the stable API).
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

from . import certify as certify
from . import information as information
from . import ratios as ratios
from . import results as results
from ._errors import ContractError, RefusalError, ScoreQuantError
from .api import fit_quantizer, optimize_partition
from .artifact import Quantizer
from .components import LinearComponentScore
from .config import (
    DExchangeConfig,
    ExecutionConfig,
    KMeansConfig,
    MahalanobisLloydConfig,
    ScalarDPConfig,
    SoftVoronoiConfig,
)
from .criteria import DOptimality, NormalizedTrace, ProfiledDOptimality
from .information import efficient_score_bound, information_report
from .providers import ScoreFunction, ScoreProvider
from .sources import (
    GaussLegendreConfig,
    IntegrationSource,
    ObservationSample,
    ScoreProvenance,
    ScoreSample,
    ScoreSchema,
)
from .uncertainty import retention_uncertainty

#: Installed distribution version. Resolved from package metadata so
#: ``pyproject.toml`` stays the single source of truth; a source tree that was
#: never installed reports ``"0.0.0.dev0"`` rather than failing to import.
try:
    __version__ = _distribution_version("scorequant")
except PackageNotFoundError:  # pragma: no cover - only hit in an uninstalled tree
    __version__ = "0.0.0.dev0"

__all__ = [
    "ContractError",
    "DExchangeConfig",
    "DOptimality",
    "ExecutionConfig",
    "GaussLegendreConfig",
    "IntegrationSource",
    "KMeansConfig",
    "LinearComponentScore",
    "MahalanobisLloydConfig",
    "NormalizedTrace",
    "ObservationSample",
    "ProfiledDOptimality",
    "Quantizer",
    "RefusalError",
    "ScalarDPConfig",
    "ScoreFunction",
    "ScoreProvenance",
    "ScoreProvider",
    "ScoreQuantError",
    "ScoreSample",
    "ScoreSchema",
    "SoftVoronoiConfig",
    "efficient_score_bound",
    "fit_quantizer",
    "information_report",
    "optimize_partition",
    "retention_uncertainty",
]
