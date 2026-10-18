"""Shared logic and committed evidence for the door3-classifier documentation page.

A two-component Gaussian mixture (`signal_pdf`, `background_pdf`) is separated
by a small `sklearn.linear_model.LogisticRegression` classifier whose
calibrated posteriors become component density ratios and then scores through
`scorequant.DensityRatioScore.from_classifier`. `exact_provider` builds the
analytic Bayes-optimal score for the same mixture, which is what makes the
retention ladder in `run_ladder` an honest comparison instead of a guess, and
what lets `run_study` add an exact-provider *ceiling* row: the same fit at
the same bin budget, with no classifier in the loop at all.

This is also the one library example that measures what an *estimated*
density-ratio score reports about itself (`information_kind ==
"supplied_score_surrogate"`, `ScoreProvenance.exact_fisher is False`) against
what it actually achieves relative to the known analytic score -- the
surrogate-information caveat. `run_study` records that measurement as committed evidence at
`examples/outputs/door3-classifier.json`, alongside the existing figure
`examples/outputs/door3-classifier.png`. It backs the ratios walkthrough.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.linear_model import LogisticRegression

import scorequant as sq
from examples._env import example_scale

ASSET_PATH = Path("examples/outputs/door3-classifier.png")
METRICS_PATH = Path("examples/outputs/door3-classifier.json")

SIGNAL_MU, SIGNAL_SIGMA = 1.0, 0.5
BACKGROUND_MU, BACKGROUND_SIGMA = 0.0, 1.5
REFERENCE_FRACTIONS = (0.3, 0.7)

#: Reference-mixture sample sizes of the full study; `run_study` shrinks
#: these under `SCOREQUANT_EXAMPLE_FAST` through `example_scale`.
N_TRAIN = 400
N_TEST = 600
#: Size of the dedicated training-measure closure sample. Deliberately much
#: larger than `N_TRAIN`: this sample never enters a fit, so it is cheap,
#: and `ratio_closure_report` needs a tight noise floor to tell classifier
#: bias apart from sampling noise -- at `N_TRAIN`-sized samples the two are
#: not reliably distinguishable for the best-trained ladder rung.
N_CLOSURE = 50_000
#: Labeled per-component training sizes swept by the retention ladder,
#: smallest first, shared by `run_ladder`, `run_study`, and the notebook.
N_PER_CLASS_VALUES: tuple[int, ...] = (15, 60, 300)
#: Bin budget shared by every rung of the ladder and the exact-provider
#: ceiling, so all four fits are compared at the same budget.
N_BINS = 4
#: Deterministic seeds: the reference-mixture draws, the closure-measure
#: draw, the solver, and the per-rung classifier training draws
#: (`CLASSIFIER_SEED_BASE + rung index`).
TRAIN_SEED = 2026
TEST_SEED = 999
CLOSURE_SEED = 4242
SOLVER_SEED = 7
CLASSIFIER_SEED_BASE = 101


def signal_pdf(x: np.ndarray) -> np.ndarray:
    """Evaluate the signal component's normal density."""
    scale = SIGNAL_SIGMA * np.sqrt(2 * np.pi)
    return np.exp(-0.5 * ((x - SIGNAL_MU) / SIGNAL_SIGMA) ** 2) / scale


def background_pdf(x: np.ndarray) -> np.ndarray:
    """Evaluate the background component's normal density."""
    scale = BACKGROUND_SIGMA * np.sqrt(2 * np.pi)
    return np.exp(-0.5 * ((x - BACKGROUND_MU) / BACKGROUND_SIGMA) ** 2) / scale


def exact_posteriors(x: np.ndarray) -> np.ndarray:
    """Return the true Bayes posterior `[N, 2]` under equal training priors."""
    values = np.asarray(x)[:, 0]
    joint = np.stack([signal_pdf(values), background_pdf(values)], axis=1)
    return joint / joint.sum(axis=1, keepdims=True)


def exact_provider() -> sq.ScoreFunction:
    """Build the analytic (non-estimated) score provider for this mixture.

    Returns
    -------
    scorequant.ScoreFunction
        A provider declaring `ScoreProvenance(kind="exact")`, the ceiling
        that `run_ladder`'s classifier-based providers approach as training
        data grows.
    """
    return sq.ScoreFunction(
        lambda x: sq.mixture_scores_from_ratios(
            sq.ratios_from_posteriors(exact_posteriors(x), [0.5, 0.5]),
            list(REFERENCE_FRACTIONS),
        ),
        provenance=sq.ScoreProvenance(kind="exact", description="analytic Bayes posterior"),
    )


def make_features(x: np.ndarray) -> np.ndarray:
    """Build the quadratic feature the classifier needs to separate unequal variances."""
    return np.column_stack([x, x**2])


def draw_reference_mixture(seed: int, n: int) -> np.ndarray:
    """Draw `n` events from the mixture at `REFERENCE_FRACTIONS`.

    Parameters
    ----------
    seed
        Deterministic seed.
    n
        Number of events to draw.

    Returns
    -------
    numpy.ndarray
        Observations with shape ``[n, 1]``.
    """
    rng = np.random.default_rng(seed)
    is_signal = rng.random(n) < REFERENCE_FRACTIONS[0]
    x = np.where(
        is_signal,
        rng.normal(SIGNAL_MU, SIGNAL_SIGMA, n),
        rng.normal(BACKGROUND_MU, BACKGROUND_SIGMA, n),
    )
    return x[:, None]


def draw_training_mixture(seed: int, n: int) -> np.ndarray:
    """Draw `n` events from the balanced 0.5/0.5 mixture the ratios are defined against.

    `ratios_from_posteriors` and `DensityRatioScore.from_classifier` both
    declare `class_priors = [0.5, 0.5]` as the ratio denominator's measure
    (`src/scorequant/ratios.py`); `ratio_closure_report` must be evaluated on
    a sample from *this* mixture, never on a sample from
    `draw_reference_mixture`'s `REFERENCE_FRACTIONS` mixture -- conflating
    the two is the measure mismatch `closure_measure_mismatch` in
    `run_study`'s record measures on purpose, as a labelled contrast.

    Parameters
    ----------
    seed
        Deterministic seed.
    n
        Number of events to draw.

    Returns
    -------
    numpy.ndarray
        Observations with shape ``[n, 1]``.
    """
    rng = np.random.default_rng(seed)
    is_signal = rng.random(n) < 0.5
    x = np.where(
        is_signal,
        rng.normal(SIGNAL_MU, SIGNAL_SIGMA, n),
        rng.normal(BACKGROUND_MU, BACKGROUND_SIGMA, n),
    )
    return x[:, None]


def train_classifier(seed: int, n_per_class: int) -> LogisticRegression:
    """Train a quadratic-feature logistic regression on balanced component samples.

    Parameters
    ----------
    seed
        Deterministic seed for the labeled training draw.
    n_per_class
        Number of labeled events drawn from each component.

    Returns
    -------
    sklearn.linear_model.LogisticRegression
        A fitted classifier; class 0 is signal, class 1 is background.
    """
    rng = np.random.default_rng(seed)
    x_signal = rng.normal(SIGNAL_MU, SIGNAL_SIGMA, n_per_class)
    x_background = rng.normal(BACKGROUND_MU, BACKGROUND_SIGMA, n_per_class)
    x = np.concatenate([x_signal, x_background])
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])
    return LogisticRegression(max_iter=1000).fit(make_features(x), y)


def classifier_provider(model: LogisticRegression, *, description: str) -> sq.DensityRatioScore:
    """Wrap a fitted classifier as a classifier-derived ratio provider.

    Parameters
    ----------
    model
        A classifier exposing `predict_proba` in `[signal, background]` order.
    description
        Human-readable label recorded on the provider's provenance.

    Returns
    -------
    scorequant.DensityRatioScore
        Always carries `kind="estimated_ratio"` provenance with the training
        priors and mixture parameterization recorded.
    """

    def predict(x: np.ndarray) -> np.ndarray:
        return model.predict_proba(make_features(np.asarray(x)[:, 0]))

    return sq.DensityRatioScore.from_classifier(
        predict,
        [0.5, 0.5],
        sq.MixtureParameterization(list(REFERENCE_FRACTIONS)),
        description=description,
    )


def _equal_frequency_labels(score: np.ndarray, n_bins: int) -> np.ndarray:
    """Return equal-frequency (quantile) cell labels for a one-dimensional score.

    The standard default binning a reader reaches for first: `n_bins`
    quantile cells of `score`, with no notion of Fisher information at all.

    Parameters
    ----------
    score
        Score column, any shape that flattens to `[N]` (e.g. `[N, 1]`).
    n_bins
        Requested number of cells; repeated quantile values can yield fewer
        distinct labels.

    Returns
    -------
    numpy.ndarray
        Integer cell index per row.
    """
    values = np.asarray(score).reshape(-1)
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    return np.digitize(values, edges)


def _equal_width_labels(score: np.ndarray, n_bins: int) -> np.ndarray:
    """Return equal-width cell labels for a one-dimensional score over its observed range.

    Parameters
    ----------
    score
        Score column, any shape that flattens to `[N]` (e.g. `[N, 1]`).
    n_bins
        Requested number of cells, spanning `[score.min(), score.max()]`.

    Returns
    -------
    numpy.ndarray
        Integer cell index per row.
    """
    values = np.asarray(score).reshape(-1)
    lo, hi = float(values.min()), float(values.max())
    edges = np.linspace(lo, hi, n_bins + 1)[1:-1]
    return np.digitize(values, edges)


@dataclass(frozen=True, slots=True)
class NaiveBinningRow:
    """What a reader's default one-dimensional binning of a score retains.

    Every field is scored the same way `LadderStep.true_retention` is --
    against the *exact* score at the labels the naive rule assigns -- so
    `true_retention` here is directly comparable to `equal_frequency_retention`
    and `equal_width_retention`, and to the corresponding `LadderStep`.

    Attributes
    ----------
    n_per_class
        Labeled training events per component for this rung, or `None` for
        the exact-provider ceiling row (which has no classifier).
    equal_frequency_retention
        D-efficiency of equal-frequency (quantile) cells of the score.
    equal_width_retention
        D-efficiency of equal-width cells of the score over its observed
        range.
    true_retention
        The fitted D-optimal partition's own true retention, carried
        alongside for direct comparison.
    """

    n_per_class: int | None
    equal_frequency_retention: float
    equal_width_retention: float
    true_retention: float


def naive_binning_row(
    *,
    scores: np.ndarray,
    oracle_test_scores: np.ndarray,
    n_bins: int,
    true_retention: float,
    n_per_class: int | None,
) -> NaiveBinningRow:
    """Score the two naive one-dimensional binnings of `scores` against the exact score.

    Parameters
    ----------
    scores
        The one-dimensional estimated (or exact) score to bin naively --
        the same vectors the fitted partition in this rung was built from.
    oracle_test_scores
        The exact mixture-fraction score at the same held-out observations,
        the denominator every retention number here is stated against.
    n_bins
        The shared bin budget.
    true_retention
        The fitted partition's own true retention, carried through unchanged.
    n_per_class
        Labeled training size of this rung, or `None` for the exact-provider
        ceiling row.

    Returns
    -------
    NaiveBinningRow
        The naive rules' retention, alongside the fitted rule's for contrast.
    """
    equal_frequency_labels = _equal_frequency_labels(scores, n_bins)
    equal_width_labels = _equal_width_labels(scores, n_bins)
    equal_frequency_retention = float(
        sq.information_report(
            oracle_test_scores, equal_frequency_labels, n_bins=n_bins
        ).geometric_mean_retention
    )
    equal_width_retention = float(
        sq.information_report(
            oracle_test_scores, equal_width_labels, n_bins=n_bins
        ).geometric_mean_retention
    )
    return NaiveBinningRow(
        n_per_class=n_per_class,
        equal_frequency_retention=equal_frequency_retention,
        equal_width_retention=equal_width_retention,
        true_retention=true_retention,
    )


@dataclass(frozen=True, slots=True)
class LadderStep:
    """One rung of the classifier-quality retention ladder.

    Attributes
    ----------
    n_per_class
        Labeled training events per component used to fit the classifier.
    surrogate_retention
        `QuantizerResult.evaluate_scores(...)` D-efficiency reported on the
        estimated scores themselves -- what the library can honestly claim
        about the vectors it was given.
    true_retention
        D-efficiency of the *exact* score, evaluated at the labels the
        estimated-score quantizer actually produced. This is what the
        surrogate number is a proxy for, and the two can disagree sharply.
    closure_residual
        `ratio_closure_report(...).max_residual` of the classifier-derived
        ratios, evaluated on a sample from the balanced 0.5/0.5 *training*
        mixture the ratios are actually defined against (never on
        `train_observations`, which comes from the reference mixture and
        measures a fixed offset rather than estimator bias -- see
        `closure_measure_mismatch` in `run_study`): exact ratios integrate to
        one under their own declared measure, so a large residual here flags
        estimator bias before any quantizer is fitted.
    """

    n_per_class: int
    surrogate_retention: float
    true_retention: float
    closure_residual: float


@dataclass(frozen=True, slots=True)
class _RungContext:
    """One ladder rung's fitted step plus what naive binning needs to reuse it.

    Kept private: `scores` and `oracle_test_scores` let `run_study` build the
    rung's `NaiveBinningRow` without re-fitting the classifier or refitting
    the quantizer -- both already done once, here.
    """

    step: LadderStep
    scores: np.ndarray
    oracle_test_scores: np.ndarray


def _run_rungs(
    *,
    n_per_class_values: tuple[int, ...],
    n_train: int,
    n_test: int,
    n_bins: int,
    n_closure: int | None,
) -> list[_RungContext]:
    """Fit a quantizer from an estimated score at each rung of classifier quality.

    The shared implementation behind `run_ladder` (which discards the extra
    context) and `run_study`'s naive-binning rows (which reuse it).
    """
    oracle = exact_provider()
    test_observations = draw_reference_mixture(TEST_SEED, n_test)
    oracle_test_scores = np.asarray(oracle.score(test_observations))
    train_observations = draw_reference_mixture(TRAIN_SEED, n_train)
    # A dedicated sample from the *training* mixture -- the measure every
    # ratio here is actually defined against -- never `train_observations`,
    # which is drawn from the reference mixture instead.
    closure_observations = draw_training_mixture(CLOSURE_SEED, n_closure or N_CLOSURE)

    contexts: list[_RungContext] = []
    for index, n_per_class in enumerate(n_per_class_values):
        model = train_classifier(CLASSIFIER_SEED_BASE + index, n_per_class)
        provider = classifier_provider(
            model, description=f"logistic regression, {n_per_class} events per class"
        )
        result = sq.fit_quantizer(
            sq.ObservationSample(train_observations),
            provider=provider,
            n_bins=n_bins,
            criterion=sq.DOptimality(),
            config=sq.DExchangeConfig(seed=SOLVER_SEED),
        )
        provider_test_scores = np.asarray(provider.score(test_observations))
        surrogate = float(result.evaluate_scores(provider_test_scores).geometric_mean_retention)
        labels = np.asarray(result.predict_scores(provider_test_scores))
        true_retention = float(
            sq.information_report(
                oracle_test_scores, labels, n_bins=result.n_bins
            ).geometric_mean_retention
        )
        closure = sq.ratio_closure_report(
            provider.ratio(closure_observations), np.ones(closure_observations.shape[0])
        )
        step = LadderStep(n_per_class, surrogate, true_retention, closure.max_residual)
        contexts.append(_RungContext(step, provider_test_scores, oracle_test_scores))
    return contexts


def run_ladder(
    *,
    n_per_class_values: tuple[int, ...],
    n_train: int,
    n_test: int,
    n_bins: int = 4,
    n_closure: int | None = None,
) -> list[LadderStep]:
    """Fit a quantizer from an estimated score at each rung of classifier quality.

    Parameters
    ----------
    n_per_class_values
        Labeled training-set sizes (per component) to try, smallest first.
    n_train, n_test
        Reference-mixture sample sizes used to fit and evaluate each quantizer.
    n_bins
        Requested bin budget, shared across every rung.
    n_closure
        Size of the dedicated training-measure sample `closure_residual` is
        evaluated on. Defaults to `N_CLOSURE`.

    Returns
    -------
    list of LadderStep
        One entry per `n_per_class_values` entry, in order.
    """
    contexts = _run_rungs(
        n_per_class_values=n_per_class_values,
        n_train=n_train,
        n_test=n_test,
        n_bins=n_bins,
        n_closure=n_closure,
    )
    return [context.step for context in contexts]


@dataclass(frozen=True, slots=True)
class ProviderProvenance:
    """The `ScoreProvenance` facts a fitted result reports about its own score.

    Attributes
    ----------
    provenance_kind
        `ScoreProvenance.kind` recorded on the provider, e.g. `"exact"` or
        `"estimated_ratio"`.
    exact_fisher
        `ScoreProvenance.exact_fisher`, *derived* from `provenance_kind`
        rather than accepted as an independent flag
        (`src/scorequant/sources.py`): only `"exact"` and `"autodiff"` set it.
    information_kind
        `QuantizerResult.information_kind`: `"exact_fisher"` when
        `exact_fisher` holds, else `"supplied_score_surrogate"` -- the label
        every classifier-backed result carries no matter how good the
        classifier is.
    """

    provenance_kind: str
    exact_fisher: bool
    information_kind: str


@dataclass(frozen=True, slots=True)
class ExactCeiling:
    """The exact-provider ceiling: the same fit, the same bin budget, no classifier.

    Attributes
    ----------
    retention
        D-efficiency of a quantizer fit directly from the analytic Bayes
        score, evaluated on the same held-out sample as every ladder rung --
        the ceiling the ladder climbs toward, and the page's honest reference
        point.
    closure_residual
        `ratio_closure_report(...).max_residual` of the *exact* ratios,
        evaluated on the same training-measure sample `run_ladder`'s rungs
        use (never `train_observations` -- see `closure_measure_mismatch`
        for what evaluating on the wrong measure looks like). Under the
        correct measure this is sampling noise on an unbiased estimator, not
        estimator bias, and should sit far below every classifier rung's
        residual.
    provenance
        `ProviderProvenance` of the exact fit: always
        `provenance_kind="exact"`, `exact_fisher=True`,
        `information_kind="exact_fisher"`.
    naive_binning
        `NaiveBinningRow` of the two naive one-dimensional binnings of the
        exact score itself (`n_per_class=None`), for the same comparison the
        ladder rungs get.
    """

    retention: float
    closure_residual: float
    provenance: ProviderProvenance
    naive_binning: NaiveBinningRow


def run_exact_ceiling(
    *,
    train_observations: np.ndarray,
    test_observations: np.ndarray,
    closure_observations: np.ndarray,
    n_bins: int,
) -> ExactCeiling:
    """Fit and evaluate the exact-provider ceiling at the ladder's bin budget.

    Parameters
    ----------
    train_observations, test_observations
        The same reference-mixture samples every ladder rung is fitted and
        evaluated on.
    closure_observations
        A sample from the balanced 0.5/0.5 training mixture -- the measure
        the exact ratios are defined against -- to evaluate
        `ratio_closure_report` on. Passing `train_observations` here would
        reproduce the reference-versus-training measure mismatch
        `closure_measure_mismatch` records deliberately, not by mistake.
    n_bins
        The shared bin budget.

    Returns
    -------
    ExactCeiling
        The ceiling retention, its ratio-closure residual under the correct
        measure, the exact provider's provenance facts, and its naive-binning
        comparison row.
    """
    oracle = exact_provider()
    result = sq.fit_quantizer(
        sq.ObservationSample(train_observations),
        provider=oracle,
        n_bins=n_bins,
        criterion=sq.DOptimality(),
        config=sq.DExchangeConfig(seed=SOLVER_SEED),
    )
    oracle_test_scores = np.asarray(oracle.score(test_observations))
    retention = float(result.evaluate_scores(oracle_test_scores).geometric_mean_retention)
    # `ScoreFunction` has no `.ratio()` -- the exact ratios are recomputed the
    # same way `exact_provider()` builds its score, so this is the same
    # closure check `run_ladder` runs on the classifier's ratios, on the same
    # training-measure sample.
    exact_ratios = sq.ratios_from_posteriors(exact_posteriors(closure_observations), [0.5, 0.5])
    closure = sq.ratio_closure_report(exact_ratios, np.ones(closure_observations.shape[0]))
    naive = naive_binning_row(
        scores=oracle_test_scores,
        oracle_test_scores=oracle_test_scores,
        n_bins=n_bins,
        true_retention=retention,
        n_per_class=None,
    )
    return ExactCeiling(
        retention=retention,
        closure_residual=float(closure.max_residual),
        provenance=ProviderProvenance(
            provenance_kind=result.provenance.kind,
            exact_fisher=bool(result.provenance.exact_fisher),
            information_kind=result.information_kind,
        ),
        naive_binning=naive,
    )


@dataclass(frozen=True, slots=True)
class ClosureMeasureMismatch:
    """The wrong-measure closure residual, kept as a deliberate, labelled contrast.

    `ratio_closure_report`'s contract (`src/scorequant/ratios.py`) is that
    `weights` must carry the reference measure of the ratio *denominator* --
    here the balanced 0.5/0.5 training mixture, never the 0.3/0.7 reference
    mixture events are actually drawn from. This record shows what the check
    reports when handed that wrong measure anyway, so a reader who makes the
    same mistake can recognise it rather than mistake it for a working
    diagnostic.

    Attributes
    ----------
    description
        Plain-language statement of what this number is and is not.
    wrong_measure_residual
        `ratio_closure_report(...).max_residual` of the *exact* provider's
        ratios, evaluated on `train_observations` (the reference mixture)
        instead of the training mixture its ratios are defined against.
    analytic_reference_measure_ratio_means
        The exact, quadrature (not sampled) reference-measure mean of each
        component ratio: the fixed offset present even with a perfect
        estimator, computed independently of any random draw.
    """

    description: str
    wrong_measure_residual: float
    analytic_reference_measure_ratio_means: list[float]


def analytic_reference_measure_ratio_means(
    *, half_width: float = 12.0, n_grid: int = 400_001
) -> list[float]:
    """Return the quadrature reference-measure mean of each exact ratio component.

    `ratios_from_posteriors` builds ``r_k(x) = phi_k(x) / p_train(x)`` against
    the balanced 0.5/0.5 training mixture, so its mean under *that* measure
    is 1 by construction. This instead integrates ``r_k`` against the
    *reference* mixture (`REFERENCE_FRACTIONS`) by quadrature -- the fixed,
    estimator-free offset `ratio_closure_report` reports when evaluated on
    the wrong measure, carrying no Monte Carlo noise of its own.

    Parameters
    ----------
    half_width
        Quadrature grid half-width around 0, in units of `x`.
    n_grid
        Number of quadrature nodes.

    Returns
    -------
    list of float
        One entry per mixture component, in `exact_posteriors` order.
    """
    x = np.linspace(-half_width, half_width, n_grid)
    step = x[1] - x[0]
    p_reference = REFERENCE_FRACTIONS[0] * signal_pdf(x) + REFERENCE_FRACTIONS[1] * background_pdf(
        x
    )
    ratios = np.asarray(sq.ratios_from_posteriors(exact_posteriors(x[:, None]), [0.5, 0.5]))
    return [float(np.sum(p_reference * ratios[:, k]) * step) for k in range(ratios.shape[1])]


def run_closure_measure_mismatch(train_observations: np.ndarray) -> ClosureMeasureMismatch:
    """Record the wrong-measure closure residual as a deliberate, labelled contrast.

    Parameters
    ----------
    train_observations
        The reference-mixture sample every ladder rung fits on -- the wrong
        measure for a closure check, used here on purpose to show what that
        mistake looks like.

    Returns
    -------
    ClosureMeasureMismatch
        The wrong-measure residual and its analytic explanation.
    """
    ratios = sq.ratios_from_posteriors(exact_posteriors(train_observations), [0.5, 0.5])
    closure = sq.ratio_closure_report(ratios, np.ones(train_observations.shape[0]))
    return ClosureMeasureMismatch(
        description=(
            "ratio_closure_report evaluated on train_observations (the 0.3/0.7 "
            "reference mixture) instead of the 0.5/0.5 training mixture the "
            "exact provider's ratios are actually defined against. This "
            "residual is not estimator error: it is present even for a "
            "perfect (exact) score, and it is what the check reports when "
            "handed the wrong measure. Compare exact_ceiling.closure_residual, "
            "which evaluates the same ratios on the correct measure instead."
        ),
        wrong_measure_residual=float(closure.max_residual),
        analytic_reference_measure_ratio_means=analytic_reference_measure_ratio_means(),
    )


def _classifier_provenance(
    *, train_observations: np.ndarray, n_per_class: int, seed: int, n_bins: int
) -> ProviderProvenance:
    """Return the classifier provider's provenance facts from one representative fit.

    `ScoreProvenance.kind` is `"estimated_ratio"` for every classifier-backed
    provider regardless of classifier quality, so any rung is representative;
    this uses the largest rung's own labeled draw.
    """
    model = train_classifier(seed, n_per_class)
    provider = classifier_provider(
        model, description=f"logistic regression, {n_per_class} events per class"
    )
    result = sq.fit_quantizer(
        sq.ObservationSample(train_observations),
        provider=provider,
        n_bins=n_bins,
        criterion=sq.DOptimality(),
        config=sq.DExchangeConfig(seed=SOLVER_SEED),
    )
    return ProviderProvenance(
        provenance_kind=result.provenance.kind,
        exact_fisher=bool(result.provenance.exact_fisher),
        information_kind=result.information_kind,
    )


@dataclass(frozen=True, slots=True)
class Study:
    """Everything the door3 page and figure need from one deterministic run."""

    metrics: dict[str, object]
    steps: list[LadderStep] = field(repr=False)


def run_study(
    *,
    n_per_class_values: tuple[int, ...] | None = None,
    n_train: int | None = None,
    n_test: int | None = None,
    n_closure: int | None = None,
    n_bins: int = N_BINS,
) -> Study:
    """Run the whole door3 classifier-ratio study and return its metrics and steps.

    Parameters
    ----------
    n_per_class_values
        Labeled per-component training sizes for the retention ladder,
        smallest first. Shrinks under `SCOREQUANT_EXAMPLE_FAST` when omitted.
    n_train, n_test
        Reference-mixture sample sizes. Shrink under `SCOREQUANT_EXAMPLE_FAST`
        when omitted.
    n_closure
        Size of the dedicated training-measure closure sample, shared by
        every ladder rung and the exact-provider ceiling so their residuals
        are directly comparable. Shrinks under `SCOREQUANT_EXAMPLE_FAST` when
        omitted; never enters a fit, so it can stay far larger than `n_train`
        without materially slowing the study.
    n_bins
        Bin budget shared by every ladder rung and the exact-provider
        ceiling.

    Returns
    -------
    Study
        The exact structure written to
        ``examples/outputs/door3-classifier.json``, together with the
        ladder steps the figure draws.
    """
    n_per_class_values = (
        example_scale(N_PER_CLASS_VALUES, (5, 15, 40))
        if n_per_class_values is None
        else n_per_class_values
    )
    n_train = example_scale(N_TRAIN, 120) if n_train is None else n_train
    n_test = example_scale(N_TEST, 200) if n_test is None else n_test
    n_closure = example_scale(N_CLOSURE, 5_000) if n_closure is None else n_closure

    rung_contexts = _run_rungs(
        n_per_class_values=n_per_class_values,
        n_train=n_train,
        n_test=n_test,
        n_bins=n_bins,
        n_closure=n_closure,
    )
    steps = [context.step for context in rung_contexts]
    naive_rows = [
        naive_binning_row(
            scores=context.scores,
            oracle_test_scores=context.oracle_test_scores,
            n_bins=n_bins,
            true_retention=context.step.true_retention,
            n_per_class=context.step.n_per_class,
        )
        for context in rung_contexts
    ]

    train_observations = draw_reference_mixture(TRAIN_SEED, n_train)
    test_observations = draw_reference_mixture(TEST_SEED, n_test)
    closure_observations = draw_training_mixture(CLOSURE_SEED, n_closure)
    ceiling = run_exact_ceiling(
        train_observations=train_observations,
        test_observations=test_observations,
        closure_observations=closure_observations,
        n_bins=n_bins,
    )
    mismatch = run_closure_measure_mismatch(train_observations)
    classifier_provenance = _classifier_provenance(
        train_observations=train_observations,
        n_per_class=n_per_class_values[-1],
        seed=CLASSIFIER_SEED_BASE + len(n_per_class_values) - 1,
        n_bins=n_bins,
    )

    gaps = [step.surrogate_retention - step.true_retention for step in steps]
    largest_index = int(np.argmax(gaps))

    # The largest gap the fitted D-optimal partition opens up over the
    # better of the two naive rules, at any rung. In one dimension this can
    # be small or even negative (a naive rule matching or beating the fit) --
    # reported as measured, not assumed.
    naive_gaps = [
        row.true_retention - max(row.equal_frequency_retention, row.equal_width_retention)
        for row in naive_rows
    ]
    largest_naive_gap_index = int(np.argmax(naive_gaps))

    metrics: dict[str, object] = {
        "problem": "door3_classifier",
        "mixture": {
            "signal_mu": SIGNAL_MU,
            "signal_sigma": SIGNAL_SIGMA,
            "background_mu": BACKGROUND_MU,
            "background_sigma": BACKGROUND_SIGMA,
            "reference_fractions": list(REFERENCE_FRACTIONS),
        },
        "n_train": n_train,
        "n_test": n_test,
        "n_closure": n_closure,
        "n_bins": n_bins,
        "seeds": {
            "train_observations": TRAIN_SEED,
            "test_observations": TEST_SEED,
            "closure_observations": CLOSURE_SEED,
            "solver": SOLVER_SEED,
            "classifier_base": CLASSIFIER_SEED_BASE,
        },
        "ladder": [asdict(step) for step in steps],
        "surrogate_gap": {
            "rows": [
                {"n_per_class": step.n_per_class, "gap": gap}
                for step, gap in zip(steps, gaps, strict=True)
            ],
            "largest_n_per_class": steps[largest_index].n_per_class,
            "largest_gap": gaps[largest_index],
        },
        "exact_ceiling": {
            "retention": ceiling.retention,
            "closure_residual": ceiling.closure_residual,
        },
        "closure_measure_mismatch": asdict(mismatch),
        "naive_binning": {
            "description": (
                "What a reader's default one-dimensional binning of the same "
                "estimated (or, for the ceiling row, exact) score retains, "
                "scored the same way true_retention is: equal-frequency "
                "(quantile) and equal-width cells, no Fisher information "
                "involved. Measured, not assumed -- in one dimension the "
                "quantile rule can be close to, or beat, the fitted partition."
            ),
            "rows": [asdict(row) for row in naive_rows],
            "exact_ceiling": asdict(ceiling.naive_binning),
            "largest_gap": naive_gaps[largest_naive_gap_index],
            "largest_gap_n_per_class": naive_rows[largest_naive_gap_index].n_per_class,
        },
        "provenance": {
            "classifier": asdict(classifier_provenance),
            "exact": asdict(ceiling.provenance),
        },
    }
    return Study(metrics=metrics, steps=steps)


def make_figure(steps: list[LadderStep]) -> Figure:
    """Render the surrogate-versus-true retention ladder.

    Parameters
    ----------
    steps
        Ladder rungs from `run_ladder`, smallest training set first.

    Returns
    -------
    matplotlib.figure.Figure
        A single-panel bar comparison of surrogate and true retention.
    """
    figure, ax = plt.subplots(figsize=(7, 4))
    labels = [f"{step.n_per_class}/class" for step in steps]
    x = np.arange(len(steps))
    width = 0.35
    surrogate_values = [s.surrogate_retention for s in steps]
    true_values = [s.true_retention for s in steps]
    ax.bar(x - width / 2, surrogate_values, width, label="surrogate (self-reported)")
    ax.bar(x + width / 2, true_values, width, label="true (exact scores, estimated labels)")
    ax.set(
        xticks=x,
        xticklabels=labels,
        ylabel="D-efficiency",
        xlabel="classifier training set size",
        title="Door 3: the surrogate-information caveat",
        ylim=(0.0, 1.05),
    )
    ax.legend()
    return figure


def main() -> None:
    """Run the study, then write the committed JSON and figure."""
    study = run_study()
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as stream:
        json.dump(study.metrics, stream, indent=2)
        stream.write("\n")
    figure = make_figure(study.steps)
    ASSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(ASSET_PATH, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
