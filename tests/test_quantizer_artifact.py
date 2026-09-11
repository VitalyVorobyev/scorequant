"""The deployable rule and its versioned, non-pickle storage format.

The point of separating :class:`~scorequant.Quantizer` from the fitted result is
that a rule can leave the process that produced it. These tests pin the two
claims that makes: a saved rule reproduces its labels exactly, and it can be
loaded and applied where JAX is not importable at all.
"""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

import scorequant as sq
from scorequant.artifact import FORMAT_VERSION

SEED = 19


def _fit(dims: int = 3, bins: int = 5) -> sq.QuantizerResult:
    scores = np.asarray(np.random.default_rng(SEED).normal(size=(600, dims)))
    schema = sq.ScoreSchema(tuple(f"theta_{index}" for index in range(dims)))
    return sq.fit_quantizer(
        sq.ScoreSample(
            scores,
            schema=schema,
            provenance=sq.ScoreProvenance(kind="exact", reference_point=(0.0,) * dims),
        ),
        n_bins=bins,
        config=sq.DExchangeConfig(seed=SEED),
    )


def test_the_fit_exposes_the_rule_it_produced() -> None:
    result = _fit()
    assert isinstance(result.quantizer, sq.Quantizer)
    assert result.quantizer.n_bins == result.n_bins == 5
    assert result.quantizer.rank == result.rank
    assert result.centers is result.quantizer.centers


def test_the_result_and_the_rule_predict_identically() -> None:
    """`predict_scores` stays the one prediction verb; the rule owns it."""
    result = _fit()
    scores = np.asarray(np.random.default_rng(SEED + 1).normal(size=(300, 3)))
    assert np.array_equal(
        np.asarray(result.predict_scores(scores)),
        np.asarray(result.quantizer.predict_scores(scores)),
    )


def test_a_saved_rule_reproduces_its_labels_exactly(tmp_path: Path) -> None:
    result = _fit()
    scores = np.asarray(np.random.default_rng(SEED + 2).normal(size=(400, 3)))
    expected = np.asarray(result.predict_scores(scores))

    path = result.quantizer.save(tmp_path / "rule")
    restored = sq.Quantizer.load(path)

    assert np.array_equal(np.asarray(restored.predict_scores(scores)), expected)
    assert restored.schema is not None
    assert restored.schema.parameters == ("theta_0", "theta_1", "theta_2")
    assert restored.provenance.kind == "exact"
    assert restored.provenance.reference_point == (0.0, 0.0, 0.0)
    assert restored.criterion.name == "d_optimality"
    assert restored.information_kind == "exact_fisher"


def test_the_artifact_contains_no_pickles(tmp_path: Path) -> None:
    """A reader must never execute code from the file it is loading."""
    path = _fit().quantizer.save(tmp_path / "rule.sqz")
    with zipfile.ZipFile(path) as archive:
        members = sorted(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
        payloads = [archive.read(name) for name in members if name.endswith(".npy")]

    assert "manifest.json" in members
    assert all(name == "manifest.json" or name.endswith(".npy") for name in members)
    assert manifest["format_version"] == FORMAT_VERSION
    # NumPy writes a pickled object array with this dtype descriptor; a rule
    # made only of numbers never carries one.
    assert not any(b"'O'" in payload for payload in payloads)


def test_save_appends_the_suffix_only_when_the_path_has_none(tmp_path: Path) -> None:
    quantizer = _fit().quantizer
    assert quantizer.save(tmp_path / "rule").name == "rule.sqz"
    assert quantizer.save(tmp_path / "rule.bin").name == "rule.bin"


def test_an_unknown_format_version_is_refused_by_name(tmp_path: Path) -> None:
    path = _fit().quantizer.save(tmp_path / "rule.sqz")
    future = tmp_path / "future.sqz"
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(future, "w") as target:
        for name in source.namelist():
            payload = source.read(name)
            if name == "manifest.json":
                manifest = json.loads(payload)
                manifest["format_version"] = FORMAT_VERSION + 1
                payload = json.dumps(manifest).encode()
            target.writestr(name, payload)

    with pytest.raises(ValueError, match=f"format_version {FORMAT_VERSION + 1}"):
        sq.Quantizer.load(future)


def test_a_file_that_is_not_an_artifact_is_refused(tmp_path: Path) -> None:
    stranger = tmp_path / "stranger.sqz"
    with zipfile.ZipFile(stranger, "w") as archive:
        archive.writestr("hello.txt", "not a quantizer")
    with pytest.raises(ValueError, match="not a ScoreQuant quantizer artifact"):
        sq.Quantizer.load(stranger)


def test_a_rule_fitted_under_jax_predicts_where_jax_is_absent(tmp_path: Path) -> None:
    """The reason the artifact exists: fit offline, deploy without the fitting stack."""
    result = _fit()
    scores = np.asarray(np.random.default_rng(SEED + 3).normal(size=(250, 3)))
    expected = np.asarray(result.predict_scores(scores))

    rule_path = result.quantizer.save(tmp_path / "rule.sqz")
    scores_path = tmp_path / "scores.npy"
    np.save(scores_path, scores)

    program = f"""
import sys

class _NoJax:
    def find_module(self, name, path=None):
        if name == "jax" or name.startswith("jax."):
            raise ImportError(name)
        return None

sys.meta_path.insert(0, _NoJax())

import numpy as np
from scorequant import ExecutionConfig
from scorequant.artifact import Quantizer

rule = Quantizer.load({str(rule_path)!r})
labels = rule.predict_scores(
    np.load({str(scores_path)!r}), execution=ExecutionConfig(backend="numpy")
)
assert "jax" not in sys.modules, "the deployment path must not import JAX"
np.save({str(tmp_path / "labels.npy")!r}, np.asarray(labels))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert np.array_equal(np.load(tmp_path / "labels.npy"), expected)


def test_a_compiled_partition_is_a_rule_not_a_second_fit() -> None:
    scores = np.asarray(np.random.default_rng(SEED).normal(size=(600, 3)))
    partition = sq.optimize_partition(
        sq.ScoreSample(scores, schema=sq.ScoreSchema(("a", "b", "c"))),
        n_bins=5,
        config=sq.DExchangeConfig(seed=SEED),
    )
    compiled = partition.compile_quantizer()

    assert isinstance(compiled, sq.Quantizer)
    assert not hasattr(compiled, "train_report")
    assert compiled.schema is not None
    assert compiled.schema.parameters == ("a", "b", "c")
    assert np.array_equal(np.asarray(compiled.predict_scores(scores)), np.asarray(partition.labels))


def test_a_profiled_rule_records_the_parameters_it_was_built_for(tmp_path: Path) -> None:
    scores = np.asarray(np.random.default_rng(SEED).normal(size=(500, 4)))
    schema = sq.ScoreSchema(("T", "B", "mono", "HSPC"))
    result = sq.fit_quantizer(
        sq.ScoreSample(scores, schema=schema),
        n_bins=5,
        criterion=sq.ProfiledDOptimality(interest=("HSPC",)),
        config=sq.SoftVoronoiConfig(seed=SEED, max_steps=40, record_every=20),
    )
    restored = sq.Quantizer.load(result.quantizer.save(tmp_path / "profiled.sqz"))
    assert isinstance(restored.criterion, sq.ProfiledDOptimality)
    assert restored.criterion.interest == (3,)
    assert result.train_profiled_report is not None
    assert result.train_profiled_report.interest_names == ("HSPC",)
    assert result.train_profiled_report.nuisance_names == ("T", "B", "mono")
    assert "interest: HSPC" in result.train_profiled_report.describe()


def test_historical_v020_artifact_loads_without_jax() -> None:
    """A committed old-writer artifact guards compatibility independently of save()."""
    fixture = Path(__file__).parent / "fixtures" / "quantizer_v020"
    program = f"""
import sys, json, hashlib
from pathlib import Path
class NoJax:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "jax" or fullname.startswith("jax."):
            raise ImportError(fullname)
sys.meta_path.insert(0, NoJax())
import numpy as np
import scorequant as sq
fixture = Path({str(fixture)!r})
expected = json.loads((fixture / "expected.json").read_text())
artifact = fixture / "rule.sqz"
assert hashlib.sha256(artifact.read_bytes()).hexdigest() == expected["artifact_sha256"]
rule = sq.Quantizer.load(artifact)
actual = rule.predict_scores(
    expected["prediction_scores"], execution=sq.ExecutionConfig(backend="numpy")
)
assert np.array_equal(actual, expected["expected_labels"])
assert rule.schema.parameters == ("location",)
assert "jax" not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
