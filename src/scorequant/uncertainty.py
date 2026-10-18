"""Sampling uncertainty of a frozen rule's held-out geometric-mean retention."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from ._binstats import scatter_bin_statistics
from ._errors import ContractError
from ._execution import canonicalize_public, execution_scope
from ._execution import xp as jnp
from ._typing import ArrayLike
from ._validation import _ValidatedSample, validate_assignments, validate_sample
from .config import ExecutionConfig, _validate_finite, validate_rank_rtol
from .reports import RetentionUncertainty, RetentionUncertaintyStatus
from .transforms import _default_rank_rtol, fisher_transform, is_numerically_singular


@dataclass(frozen=True, slots=True)
class _RetentionInfluence:
    """Outcome of the guarded plug-in: a status, the estimate, and the influence values."""

    status: RetentionUncertaintyStatus
    estimate: float | None
    influence: jnp.ndarray | None


def _retention_influence(
    sample: _ValidatedSample,
    labels: jnp.ndarray,
    n_bins: int,
    rank_rtol: float | None,
) -> _RetentionInfluence:
    r"""Evaluate the O7 plug-in and its influence values behind the rank guards.

    With \(W\) the whitening matrix of the full moment \(\hat V\)
    (\(W^\top\hat V W=I\), so \(\hat V^{-1}=WW^\top\)) and
    \(R=W^\top\hat I_Z W\), the influence of
    ``RETENTION-PLUGIN-CLT-FROZEN-VECTOR`` becomes
    \(\psi_i=(\hat\eta_D/d)[2t_i^\top R^{-1}u_{z_i}-u_{z_i}^\top R^{-1}u_{z_i}-\|t_i\|^2]\)
    in the whitened rows \(t_i=W^\top s_i\) and cell means \(u_b=W^\top c_b\),
    because \(\hat I_Z^{-1}=WR^{-1}W^\top\). One symmetric solve per cell
    replaces both inverses, and every rank decision is taken on \(R\), whose
    spectrum is invariant under a linear reparameterization of the scores.

    \(R\) is formed from the cell means of the whitened rows, never as
    \(W^\top\hat I_Z W\) from a retained matrix assembled in the supplied
    coordinates: on an ill-conditioned but regular full moment the latter
    amplifies the rounding of \(\hat I_Z\) by the condition number of
    \(\hat V\) and can invent a spurious smallest eigenvalue on an exactly
    rank-deficient between-cell matrix, which the rank guard then misses.
    """
    scores = sample.effective_scores
    n_rows, dimension = scores.shape
    # A rule with at most d cells retains no D-information at the reference
    # law: FI-RANK-CEILING bounds the population rank of I_Z by K - 1, so the
    # target is zero there and the sample plug-in is the upward-biased
    # endpoint estimator whatever the empirical rank happens to be. This is
    # decided on the declared cell count, empty cells included, before any
    # moment is formed.
    if n_bins <= dimension:
        return _RetentionInfluence("insufficient_cells", None, None)
    unit_weights = jnp.ones(n_rows, dtype=scores.dtype)
    full_moment = jnp.einsum("np,nq->pq", scores, scores) / n_rows

    # A supplied direction that the sample loses makes the d-dimensional
    # target undefined. ``information_report`` would project it out and report
    # the ratio on the surviving subspace, which is how the library returns 1
    # on the singular sample of CE-O7-UNIT-RETENTION-SINGULAR-SAMPLE-001 while
    # the plug-in's own convention gives 0; neither is an estimate of the
    # frozen rule's retention, so the target is refused instead of replaced.
    try:
        transform = fisher_transform(full_moment, whiten=True, rank_rtol=rank_rtol)
    except ContractError:
        return _RetentionInfluence("singular_full_information", None, None)
    if transform.rank < dimension:
        return _RetentionInfluence("singular_full_information", None, None)
    basis = jnp.asarray(transform.matrix, dtype=scores.dtype)
    coordinates = scores @ basis
    cells = scatter_bin_statistics(labels, unit_weights, coordinates, n_bins)
    cell_coordinates = cells.means
    # Empty cells have zero weight and a zero mean, so they add nothing here.
    whitened_retained = (
        jnp.einsum("b,bp,bq->pq", cells.weights, cell_coordinates, cell_coordinates) / n_rows
    )
    whitened_retained = 0.5 * (whitened_retained + whitened_retained.T)

    # Decided before any determinant root or solve: on an exactly deficient
    # between-cell matrix the smallest eigenvalue is rounding noise, and its
    # d-th root is not (the audit measured 3.5e-9 in d = 2), so a determinant
    # ratio would report a regular-looking endpoint that the theorem excludes.
    # At the dtype default, never the caller's ``rank_rtol``, exactly as
    # ``information_report`` judges the same between-cell matrix.
    if is_numerically_singular(whitened_retained):
        return _RetentionInfluence("singular_retained_information", 0.0, None)

    eigenvalues = np.asarray(jnp.linalg.eigvalsh(whitened_retained), dtype=np.float64)
    estimate = float(np.exp(np.mean(np.log(eigenvalues))))

    cell_solutions = jnp.linalg.solve(whitened_retained, cell_coordinates.T).T
    cell_norms = jnp.sum(cell_coordinates * cell_solutions, axis=1)
    cross = jnp.sum(coordinates * cell_solutions[labels], axis=1)
    own = jnp.sum(coordinates * coordinates, axis=1)
    bracket = 2.0 * cross - cell_norms[labels] - own
    influence = (estimate / dimension) * bracket

    # The three terms of the bracket sum to zero over the sample and cancel
    # row by row on any law whose cells sit on their O7 ellipsoids
    # (CE-O7-ELLIPSOID-ZERO-VARIANCE-001). What survives such a cancellation is
    # rounding noise proportional to the magnitude the terms carried, so the
    # variance is declared degenerate when the root-mean-square bracket is
    # below the dtype's rank threshold times the root-mean-square magnitude.
    # This is a numerical guard on the sample; it does not decide whether the
    # population variance is zero, and a perturbation of the ellipsoid law
    # smaller than the threshold is absorbed by it. The user's ``rank_rtol``
    # does not move this threshold.
    magnitude = 2.0 * jnp.abs(cross) + cell_norms[labels] + own
    bracket_scale = float(np.sqrt(np.asarray(jnp.mean(bracket * bracket))))
    magnitude_scale = float(np.sqrt(np.asarray(jnp.mean(magnitude * magnitude))))
    if bracket_scale <= _default_rank_rtol(scores.dtype) * magnitude_scale:
        return _RetentionInfluence("degenerate_variance", estimate, influence)
    return _RetentionInfluence("ok", estimate, influence)


@execution_scope
def retention_uncertainty(
    scores: ArrayLike,
    assignments: ArrayLike,
    *,
    n_bins: int | None = None,
    confidence_level: float = 0.95,
    rank_rtol: float | None = None,
    execution: ExecutionConfig | None = None,
) -> RetentionUncertainty:
    r"""Estimate the sampling uncertainty of a frozen rule's held-out retention.

    The rows are an evaluation sample the rule never saw: independent,
    equally weighted draws from the reference law at which the model's true
    scores are evaluated, whose ``scores`` are those true scores and whose
    ``assignments`` are the frozen rule's labels for them. The estimate
    is the uncentred plug-in geometric-mean retention
    \(\hat\eta_D=(\det\hat I_Z/\det\hat V)^{1/d}\), the same number
    ``information_report`` reports on a full-rank sample. Its standard error
    is the influence-function estimate of the conditional central limit
    theorem ``RETENTION-PLUGIN-CLT-FROZEN-VECTOR`` (scalar case
    ``RETENTION-PLUGIN-CLT-FROZEN-SCALAR``),

    .. math::

        \psi_i=\frac{\hat\eta_D}{d}\Big[2s_i^\top\hat I_Z^{-1}\hat c_{z_i}
        -\hat c_{z_i}^\top\hat I_Z^{-1}\hat c_{z_i}-s_i^\top\hat V^{-1}s_i\Big],
        \qquad
        \widehat{\mathrm{SE}}=\sqrt{\tfrac{1}{N}\cdot\tfrac{1}{N}\sum_i\psi_i^2},

    and the interval is the untruncated two-sided Wald interval
    \(\hat\eta_D\pm z_{1-\alpha/2}\widehat{\mathrm{SE}}\).

    Parameters
    ----------
    scores
        Finite true-score matrix with shape ``[N, P]``, ``N >= 2``. Rows are
        never centered: the score-space origin has statistical meaning.
    assignments
        Integer label of the frozen rule for every row, with shape ``[N]``.
    n_bins
        Total number of cells, including empty ones. Inferred when omitted.
        Empty cells contribute nothing.
    confidence_level
        Two-sided nominal level, strictly between zero and one.
    rank_rtol
        Relative eigenvalue threshold for the informative directions of the
        full moment. A dtype-aware default is used when omitted. It moves
        neither the between-cell singularity test, which is judged at the
        dtype default as in ``information_report``, nor the cancellation
        threshold of the ``degenerate_variance`` guard.

    Returns
    -------
    RetentionUncertainty
        Estimate, standard error, interval and the status that says which of
        them are available.

    Notes
    -----
    The interval covers sampling variability of the evaluation draw for a
    frozen rule under the theorem's conditions: positive cell probabilities,
    finite fourth moments of the score, positive definite full and
    between-cell moments, and positive asymptotic variance. None of these can
    be verified from an array, and a positive empirical rank does not
    certify a population eigenvalue floor. The theorem is asymptotic: an
    infinite fourth moment puts a law outside it, and heavy tails, including
    laws that satisfy every condition, can cause material undercoverage at
    moderate sample sizes. Reading the number as the model's Fisher
    retention needs the evaluation law to be the reference law, where the
    true score has mean zero; under another law the same arithmetic
    estimates an uncentred determinant ratio that is not that retention.
    Weighted samples, rules refitted on the evaluation rows, profiled
    \(D_s\) retention and any statement without true scores are outside
    this diagnostic (``OPEN-RETENTION-UNCERTAINTY``).

    Scores from a classifier or another estimator are not the true scores.
    Their reported retention carries a separate proxy bias that the
    score-error budget ``SCORE-ERROR-RETENTION-BUDGET`` bounds only under
    truth-dependent error and conditioning assumptions; AUC or calibration
    alone does not certify that bias, and this error bar never measures it.

    Four outcomes withhold the interval rather than report an unsupported
    one. ``insufficient_cells``: the rule declares at most \(d\) cells, so
    ``FI-RANK-CEILING`` makes the population retention zero at the reference
    law and the plug-in is the upward-biased endpoint estimator; nothing is
    reported. ``singular_full_information``: the full moment is rank
    deficient at the threshold. ``singular_retained_information``: the
    between-cell moment is numerically rank deficient on this sample, so
    the diagnostic reports zero by its numerical-rank convention and
    withholds the interval. The exact plug-in can still be positive below
    the threshold; this sample-side refusal does not identify the population
    rank. Population singularity is a separate boundary of the Wald theorem
    (``RETENTION-PLUGIN-SINGULAR-ENDPOINT-RATE``).
    ``degenerate_variance``: the influence values cancel to rounding noise,
    the sample-side face of
    ``CE-O7-ELLIPSOID-ZERO-VARIANCE-001``. The test is
    \(\mathrm{rms}(B)\le\tau\,\mathrm{rms}(M)\) with \(B_i\) the bracket
    of \(\psi_i\), \(M_i\) the sum of the magnitudes of its three terms and
    \(\tau\) the dtype's rank threshold (\(10^{-10}\) in float64,
    about \(1.2\times10^{-5}\) in float32, not moved by ``rank_rtol``); it is a numerical
    guard that can suppress small positive influence variance and never
    asserts that the population variance is zero. The endpoints of an
    ``ok`` interval are not clipped to ``[0, 1]``.
    """
    del execution
    _validate_finite("confidence_level", confidence_level, positive=True)
    if confidence_level >= 1:
        raise ContractError("confidence_level must be strictly between zero and one")
    validate_rank_rtol(rank_rtol)
    sample = validate_sample(scores)
    if sample.n_effective < 2:
        raise ContractError("retention_uncertainty requires at least two observations")
    labels, resolved_n_bins = validate_assignments(sample, assignments, n_bins)
    outcome = _retention_influence(sample, labels, resolved_n_bins, rank_rtol)
    n_rows = sample.n_effective
    standard_error: float | None = None
    interval: tuple[float, float] | None = None
    if outcome.status == "degenerate_variance":
        standard_error = 0.0
    elif outcome.status == "ok":
        assert outcome.influence is not None and outcome.estimate is not None
        variance = float(np.asarray(jnp.mean(outcome.influence * outcome.influence)))
        standard_error = float(np.sqrt(variance / n_rows))
        # Evaluated from the tail: 1 - level is exact near one, while
        # 0.5 + 0.5 * level rounds to one for a level within an ulp of one.
        quantile = -NormalDist().inv_cdf(0.5 * (1.0 - confidence_level))
        interval = (
            outcome.estimate - quantile * standard_error,
            outcome.estimate + quantile * standard_error,
        )
    return canonicalize_public(
        RetentionUncertainty(
            estimate=outcome.estimate,
            standard_error=standard_error,
            confidence_interval=interval,
            confidence_level=float(confidence_level),
            n_observations=n_rows,
            status=outcome.status,
        )
    )
