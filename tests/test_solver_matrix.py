"""Guard the generated criterion/solver compatibility matrix against drift.

``website/scripts/generate_data.py`` derives the criterion/solver table from
``scorequant.api._SOLVER_TABLE`` and writes it into a marked region of ``docs/guide.md``,
``docs/api.md`` and ``README.md``. These tests check those copies against the registry, and
check that the library accepts exactly the declared pairs.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

from scorequant.api import _SOLVER_TABLE

ROOT = Path(__file__).resolve().parents[1]
GENERATE_DATA_MODULE = ROOT / "website" / "scripts" / "generate_data.py"

# The Markdown files whose ``<!-- generated: solver-matrix -->`` region must equal
# the table rendered from the current registry. README and the API guide carry the extra
# "Contract" column.
_CONSUMERS = (
    (ROOT / "docs" / "guide.md", False),
    (ROOT / "README.md", True),
    (ROOT / "docs" / "api.md", True),
)


def _load_generate_data() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_scorequant_portal_generate_data", GENERATE_DATA_MODULE
    )
    assert spec is not None and spec.loader is not None, GENERATE_DATA_MODULE
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generate_data() -> ModuleType:
    assert GENERATE_DATA_MODULE.is_file(), GENERATE_DATA_MODULE
    return _load_generate_data()


def _extract_generated_region(text: str, start: str, end: str) -> str:
    match = re.search(re.escape(start) + r"(.*?)" + re.escape(end), text, flags=re.DOTALL)
    assert match is not None, "solver-matrix generated region not found"
    return match.group(1).strip("\n")


def test_solver_matrix_matches_registry(generate_data: ModuleType) -> None:
    """``_solver_matrix()`` reflects ``_SOLVER_TABLE`` exactly, row for row, in order."""
    rows = generate_data._solver_matrix()
    assert [row["configuration"] for row in rows] == [
        config_type.__name__ for config_type in _SOLVER_TABLE
    ]
    for row, (config_type, spec) in zip(rows, _SOLVER_TABLE.items(), strict=True):
        assert row["configuration"] == config_type.__name__
        assert row["partitionCriteria"] == [
            criterion.__name__ for criterion in spec.partition_criteria
        ]
        assert row["quantizerCriteria"] == [
            criterion.__name__ for criterion in spec.quantizer_criteria
        ]


@pytest.mark.parametrize(("path", "with_contract"), _CONSUMERS, ids=lambda value: str(value))
def test_generated_region_matches_rendered_matrix(
    generate_data: ModuleType, path: Path, with_contract: bool
) -> None:
    """Each committed generated region equals the table rendered from the current registry."""
    rows = generate_data._solver_matrix()
    expected = generate_data._render_solver_matrix(rows, with_contract=with_contract)
    actual = _extract_generated_region(
        path.read_text(encoding="utf-8"),
        generate_data._SOLVER_MATRIX_START,
        generate_data._SOLVER_MATRIX_END,
    )
    assert actual == expected


def test_every_task_accepts_exactly_the_declared_pairs() -> None:
    """All thirty (task, config, criterion) combinations: ten run, twenty refuse up front."""
    import numpy as np

    import scorequant as sq

    rng = np.random.default_rng(140)
    mixing = np.array([[1.0, 0.5, -0.2], [0.0, 1.1, 0.3], [0.0, 0.0, 0.9]])
    table = rng.normal(size=(160, 3)) @ mixing
    scalar = rng.normal(size=(160, 1))
    configs = {
        "DExchangeConfig": sq.DExchangeConfig(seed=0, initializer_restarts=2),
        "MahalanobisLloydConfig": sq.MahalanobisLloydConfig(seed=0, initializer_restarts=2),
        "KMeansConfig": sq.KMeansConfig(seed=0, solver_restarts=2),
        "SoftVoronoiConfig": sq.SoftVoronoiConfig(
            seed=0, initializer_restarts=2, max_steps=20, record_every=20
        ),
        "ScalarDPConfig": sq.ScalarDPConfig(),
    }
    criteria = {
        "DOptimality": sq.DOptimality(),
        "ProfiledDOptimality": sq.ProfiledDOptimality((0,)),
        "NormalizedTrace": sq.NormalizedTrace(),
    }
    supported = {
        ("optimize_partition", "DExchangeConfig"): {"DOptimality", "ProfiledDOptimality"},
        ("optimize_partition", "MahalanobisLloydConfig"): {"DOptimality", "ProfiledDOptimality"},
        ("fit_quantizer", "DExchangeConfig"): {"DOptimality"},
        ("fit_quantizer", "MahalanobisLloydConfig"): {"DOptimality"},
        ("fit_quantizer", "KMeansConfig"): {"NormalizedTrace"},
        ("fit_quantizer", "SoftVoronoiConfig"): {"DOptimality", "ProfiledDOptimality"},
        ("fit_quantizer", "ScalarDPConfig"): {"DOptimality"},
    }
    accepted = refused = 0
    for task in ("optimize_partition", "fit_quantizer"):
        for config_name, config in configs.items():
            for criterion_name, criterion in criteria.items():
                data = scalar if config_name == "ScalarDPConfig" else table
                expected = criterion_name in supported.get((task, config_name), set())
                try:
                    if task == "optimize_partition":
                        sq.optimize_partition(data, n_bins=4, criterion=criterion, config=config)
                    else:
                        sq.fit_quantizer(
                            sq.ScoreSample(data), n_bins=4, criterion=criterion, config=config
                        )
                except (TypeError, ValueError) as error:
                    assert not expected, f"{task}/{config_name}/{criterion_name}: {error}"
                    refused += 1
                else:
                    assert expected, f"{task}/{config_name}/{criterion_name} should be refused"
                    accepted += 1
    assert (accepted, refused) == (10, 20)
