from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import jax.numpy as jnp

import scorequant
from tests._fit import fit_test_quantizer


def test_all_visualizations_construct_figures() -> None:
    scores = jnp.linspace(-2, 2, 80)[:, None]
    result = fit_test_quantizer(
        scores,
        n_bins=4,
        config=scorequant.KMeansConfig(solver_restarts=2),
        diagnostics="full",
    )
    figures = [
        scorequant.plot_optimization(result.trace),
        scorequant.plot_partition(result, scores),
        scorequant.plot_information(result.train_report),
        scorequant.plot_summary(result, scores),
        result.plot_summary(scores),
    ]
    assert all(figure.axes for figure in figures)
    assert figures[2].axes[0].images[0].get_clim() == (-1.0, 1.0)
    assert figures[3].axes[1].images[0].get_clim() == (-1.0, 1.0)
    for figure in figures:
        figure.clf()


def test_retained_information_plot_preserves_negative_matrix_entries() -> None:
    scores = jnp.asarray([[-1.0, 1.0], [0.0, 0.2], [1.0, -1.0], [2.0, 0.5]])
    labels = jnp.asarray([0, 0, 1, 1])
    report = scorequant.information_report(scores, labels, n_bins=2)
    assert np.asarray(report.retained_matrix).min() < 0
    figure = scorequant.plot_information(report)
    image = figure.axes[0].images[0]
    assert image.get_cmap().name == "coolwarm"
    assert image.get_clim() == (-1.0, 1.0)
    figure.clf()


def test_partition_rejects_implicit_high_dimensional_projection() -> None:
    rng = np.random.default_rng(15)
    scores = jnp.asarray(rng.normal(size=(120, 3)))
    result = fit_test_quantizer(scores, n_bins=5)
    with pytest.raises(ValueError, match="projection-free"):
        scorequant.plot_partition(result, scores)


def test_high_dimensional_summary_uses_information_spectrum() -> None:
    rng = np.random.default_rng(16)
    scores = jnp.asarray(rng.normal(size=(120, 3)))
    result = fit_test_quantizer(scores, n_bins=5, diagnostics="full")
    figure = scorequant.plot_summary(result, scores)
    first_axis = figure.axes[0]
    assert first_axis.get_title() == "Retained-information spectrum"
    assert not first_axis.collections
    assert len(first_axis.patches) == 3
    figure.clf()


def test_high_dimensional_center_motion_uses_all_coordinates() -> None:
    rng = np.random.default_rng(17)
    scores = jnp.asarray(rng.normal(size=(120, 3)))
    result = fit_test_quantizer(
        scores,
        n_bins=5,
        config=scorequant.SoftVoronoiConfig(max_iter=8, record_every=1),
        diagnostics="full",
    )
    figure = scorequant.plot_optimization(result.trace)
    motion_axis = figure.axes[3]
    assert motion_axis.get_title() == "Center displacement across all dimensions"
    expected = np.linalg.norm(np.diff(np.asarray(result.trace.centers), axis=0), axis=2)
    np.testing.assert_allclose(motion_axis.lines[0].get_ydata(), np.median(expected, axis=1))
    np.testing.assert_allclose(motion_axis.lines[1].get_ydata(), np.max(expected, axis=1))
    figure.clf()
