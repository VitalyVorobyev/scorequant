"""Deterministic synthetic problems used by scripts and tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from scorequant.ratios import IntensityParameterization


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    observations: np.ndarray
    scores: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True, slots=True)
class SyntheticProblem:
    name: str
    title: str
    n_bins: int
    train: SyntheticDataset
    validation: SyntheticDataset
    test: SyntheticDataset


def _three_splits(
    generator: Callable[[np.random.Generator, int], SyntheticDataset],
    *,
    seed: int,
    sizes: tuple[int, int, int],
) -> tuple[SyntheticDataset, SyntheticDataset, SyntheticDataset]:
    generators = [np.random.default_rng(value) for value in np.random.SeedSequence(seed).spawn(3)]
    return tuple(generator(rng, size) for rng, size in zip(generators, sizes, strict=True))  # type: ignore[return-value]


def gaussian_location(
    *, seed: int = 10, sizes: tuple[int, int, int] = (2_000, 1_000, 10_000)
) -> SyntheticProblem:
    """A one-parameter Gaussian location problem with analytic score ``s(x)=x``."""

    def generate(rng: np.random.Generator, size: int) -> SyntheticDataset:
        observations = rng.normal(size=(size, 1))
        return SyntheticDataset(observations, observations.copy(), np.ones(size))

    train, validation, test = _three_splits(generate, seed=seed, sizes=sizes)
    return SyntheticProblem("gaussian_location", "Gaussian location", 4, train, validation, test)


def _spectral_components(x: np.ndarray) -> np.ndarray:
    first = 0.08 + np.exp(-0.5 * ((x - 0.22) / 0.055) ** 2)
    first += 0.65 * np.exp(-0.5 * ((x - 0.78) / 0.075) ** 2)
    second = 0.10 + 0.9 * np.exp(-0.5 * ((x - 0.48) / 0.09) ** 2)
    second += 0.75 * np.exp(-0.5 * ((x - 0.90) / 0.035) ** 2)
    return np.column_stack([first, second])


def spectral_templates(
    *, seed: int = 20, sizes: tuple[int, int, int] = (4_000, 2_000, 15_000)
) -> SyntheticProblem:
    """A weighted two-parameter non-monotonic template-intensity problem."""

    coefficients = np.asarray([1.0, 0.8])

    def generate(rng: np.random.Generator, size: int) -> SyntheticDataset:
        x = rng.uniform(0, 1, size)
        components = _spectral_components(x)
        intensity = components @ coefficients
        scores = np.asarray(IntensityParameterization(coefficients).scores(components))
        return SyntheticDataset(x[:, None], scores, intensity)

    train, validation, test = _three_splits(generate, seed=seed, sizes=sizes)
    return SyntheticProblem(
        "spectral_templates", "Overlapping spectral templates", 8, train, validation, test
    )


def _spatial_components(xy: np.ndarray) -> np.ndarray:
    x, y = xy[:, 0], xy[:, 1]

    def gaussian(cx: float, cy: float, width: float) -> np.ndarray:
        return np.exp(-0.5 * ((x - cx) ** 2 + (y - cy) ** 2) / width**2)

    first = 0.025 + gaussian(-0.48, -0.18, 0.24) + 0.55 * gaussian(0.46, 0.50, 0.17)
    second = 0.025 + gaussian(0.40, -0.30, 0.27) + 0.60 * gaussian(-0.38, 0.53, 0.19)
    return np.column_stack([first, second])


def spatial_sources(
    *, seed: int = 30, sizes: tuple[int, int, int] = (6_000, 3_000, 20_000)
) -> SyntheticProblem:
    """A two-dimensional importance-weighted overlapping-source problem."""

    coefficients = np.asarray([1.0, 0.9])

    def generate(rng: np.random.Generator, size: int) -> SyntheticDataset:
        observations = rng.uniform(-1, 1, size=(size, 2))
        components = _spatial_components(observations)
        intensity = components @ coefficients
        scores = np.asarray(IntensityParameterization(coefficients).scores(components))
        return SyntheticDataset(observations, scores, intensity)

    train, validation, test = _three_splits(generate, seed=seed, sizes=sizes)
    return SyntheticProblem(
        "spatial_sources", "Overlapping spatial sources", 16, train, validation, test
    )


def _mixture_components(xy: np.ndarray, *, separation: float, width: float) -> np.ndarray:
    x, y = xy[:, 0], xy[:, 1]

    def gaussian(cx: float) -> np.ndarray:
        return np.exp(-0.5 * ((x - cx) ** 2 + y**2) / width**2)

    first = 0.02 + gaussian(-separation / 2)
    second = 0.02 + gaussian(separation / 2)
    return np.column_stack([first, second])


def two_parameter_gaussian_mixture(
    *,
    seed: int = 40,
    sizes: tuple[int, int, int] = (4_000, 2_000, 15_000),
    n_bins: int = 16,
    separation: float = 1.4,
    width: float = 0.55,
) -> SyntheticProblem:
    """Two overlapping 2D Gaussian bumps with an exact linear component score.

    A minimal two-parameter problem for comparing every solver against every
    baseline (the "solver shootout"): the reference measure is uniform over a
    bounded square, the two components are isotropic Gaussian bumps centered
    symmetrically on the x-axis, and each event's exact score is the linear
    component score ``phi_k(x) / lambda(x; c0)``, the intensity map of
    `scorequant.ratios.IntensityParameterization`.

    Parameters
    ----------
    seed
        Deterministic seed spawned into three independent generators, one
        per split.
    sizes
        Row counts for the train, validation, and test splits.
    n_bins
        Requested bin budget recorded on the returned problem. Defaults to a
        perfect square so the equal-width-grid baseline can use it directly
        in two dimensions.
    separation
        Distance between the two bump centers; smaller values increase
        component overlap.
    width
        Shared isotropic Gaussian width; larger values increase overlap.

    Returns
    -------
    SyntheticProblem
        Train, validation, and test splits with ``[N, 2]`` observations and
        ``[N, 2]`` exact scores.
    """

    coefficients = np.asarray([0.5, 0.5])

    def generate(rng: np.random.Generator, size: int) -> SyntheticDataset:
        observations = rng.uniform(-2.5, 2.5, size=(size, 2))
        components = _mixture_components(observations, separation=separation, width=width)
        intensity = components @ coefficients
        scores = np.asarray(IntensityParameterization(coefficients).scores(components))
        return SyntheticDataset(observations, scores, intensity)

    train, validation, test = _three_splits(generate, seed=seed, sizes=sizes)
    return SyntheticProblem(
        "two_parameter_gaussian_mixture",
        "Two-parameter Gaussian mixture",
        n_bins,
        train,
        validation,
        test,
    )


@dataclass(frozen=True, slots=True)
class SignalBackgroundDataset:
    """One split of a signal-fraction plus background-shape mixture sample.

    Attributes
    ----------
    observations
        Reference-measure coordinates with shape ``[N, 1]``.
    scores
        Exact linear component scores with shape ``[N, 1 + n_background]``.
        Column 0 is the signal-fraction (interest) score; the remaining
        columns are the background-shape (nuisance) scores, in the order
        declared by `SignalBackgroundProblem.nuisance`.
    weights
        The reference mixture intensity at `observations`.
    """

    observations: np.ndarray
    scores: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True, slots=True)
class SignalBackgroundProblem:
    """A signal-fraction interest parameter with background-shape nuisances.

    Every component is an exact, normalized density on ``[0, 1]``, and the
    reference coefficients sum to one, so `intensity` is itself a probability
    density: `coefficients[0]` is exactly the reference signal fraction.
    `evaluate_components` and `intensity` expose the exact densities so a
    later example can build an `scorequant.IntegrationSource` directly,
    rather than only consuming the precomputed `train`/`validation`/`test`
    splits below.

    Attributes
    ----------
    name, title, n_bins
        Problem identity and requested bin budget.
    interest
        Score-column indices for the interest parameter (signal fraction).
    nuisance
        Score-column indices for the background-shape nuisance parameters,
        suitable for `scorequant.ProfiledDOptimality`.
    bounds
        The one-dimensional domain box, shape ``[1, 2]``, for
        `scorequant.IntegrationSource`.
    coefficients
        Reference coefficients aligned with `component_names`: signal
        fraction first, then each background weight.
    component_names
        Names aligned with `coefficients` and the columns of
        `evaluate_components`.
    train, validation, test
        Precomputed splits with matching exact scores.
    """

    name: str
    title: str
    n_bins: int
    interest: tuple[int, ...]
    nuisance: tuple[int, ...]
    bounds: np.ndarray
    coefficients: np.ndarray
    component_names: tuple[str, ...]
    signal_density: Callable[[np.ndarray], np.ndarray] = field(repr=False)
    background_densities: tuple[Callable[[np.ndarray], np.ndarray], ...] = field(repr=False)
    train: SignalBackgroundDataset = field(repr=False)
    validation: SignalBackgroundDataset = field(repr=False)
    test: SignalBackgroundDataset = field(repr=False)

    def evaluate_components(self, observations: np.ndarray) -> np.ndarray:
        """Return the exact component densities at `observations`.

        Parameters
        ----------
        observations
            Coordinates with shape ``[N, 1]`` or ``[N]``.

        Returns
        -------
        numpy.ndarray
            Component density matrix with shape ``[N, 1 + n_background]``,
            columns aligned with `component_names`.
        """
        x = np.asarray(observations, dtype=float).reshape(-1)
        columns = [self.signal_density(x)]
        columns.extend(density(x) for density in self.background_densities)
        return np.column_stack(columns)

    def intensity(self, observations: np.ndarray) -> np.ndarray:
        """Return the reference mixture density at `observations`."""
        return self.evaluate_components(observations) @ self.coefficients


def _signal_shape_density(x: np.ndarray, *, center: float, width: float) -> np.ndarray:
    """Approximately normalized Gaussian bump on ``[0, 1]``."""
    return np.exp(-0.5 * ((x - center) / width) ** 2) / (width * np.sqrt(2 * np.pi))


def _background_shape_density(x: np.ndarray, *, rate: float) -> np.ndarray:
    """Exactly normalized truncated-exponential density on ``[0, 1]``."""
    normalizer = rate / (1.0 - np.exp(-rate))
    return normalizer * np.exp(-rate * x)


def signal_background_shape(
    *,
    seed: int = 50,
    sizes: tuple[int, int, int] = (4_000, 2_000, 15_000),
    n_bins: int = 8,
    signal_fraction: float = 0.35,
    background_rates: Sequence[float] = (1.0, 4.0),
    signal_center: float = 0.5,
    signal_width: float = 0.06,
) -> SignalBackgroundProblem:
    """A signal fraction (interest) plus background-shape nuisances on ``[0, 1]``.

    The reference density is a mixture of one signal peak and one or more
    truncated-exponential background shapes, ``lambda(x; c) = sum_k c_k
    phi_k(x)``, with every ``phi_k`` an exact, normalized density on
    ``[0, 1]`` and every ``c_k`` summing to one. The exact linear component
    score at the reference point is ``s_k(x) = phi_k(x) / lambda(x; c0)``
    (`scorequant.ratios.IntensityParameterization`); column 0 is the signal-fraction score and the
    remaining columns are the background-shape scores, matching
    `scorequant.ProfiledDOptimality`'s ``interest``/nuisance convention
    directly.

    Parameters
    ----------
    seed
        Deterministic seed spawned into three independent generators, one
        per split.
    sizes
        Row counts for the train, validation, and test splits.
    n_bins
        Requested bin budget recorded on the returned problem.
    signal_fraction
        Reference coefficient of the signal component, in ``(0, 1)``. The
        remaining ``1 - signal_fraction`` mass is split evenly across the
        background shapes.
    background_rates
        One truncated-exponential rate per background-shape nuisance
        direction; each entry adds one nuisance score column.
    signal_center, signal_width
        Location and width of the signal Gaussian bump.

    Returns
    -------
    SignalBackgroundProblem
        Train, validation, and test splits, exact component densities, and
        the interest/nuisance score-column layout.

    Raises
    ------
    ValueError
        If `signal_fraction` is not in ``(0, 1)`` or `background_rates` is
        empty.
    """
    if not 0.0 < signal_fraction < 1.0:
        raise ValueError("signal_fraction must be strictly between 0 and 1")
    if len(background_rates) < 1:
        raise ValueError("background_rates must declare at least one nuisance direction")

    n_background = len(background_rates)
    coefficients = np.concatenate(
        [[signal_fraction], np.full(n_background, (1.0 - signal_fraction) / n_background)]
    )
    component_names = ("signal", *(f"background_{index + 1}" for index in range(n_background)))

    def signal_density(x: np.ndarray) -> np.ndarray:
        return _signal_shape_density(x, center=signal_center, width=signal_width)

    background_densities = tuple(
        (lambda x, rate=rate: _background_shape_density(x, rate=rate)) for rate in background_rates
    )

    def evaluate(x: np.ndarray) -> np.ndarray:
        columns = [signal_density(x)]
        columns.extend(density(x) for density in background_densities)
        return np.column_stack(columns)

    def generate(rng: np.random.Generator, size: int) -> SignalBackgroundDataset:
        x = rng.uniform(0.0, 1.0, size)
        components = evaluate(x)
        intensity = components @ coefficients
        scores = np.asarray(IntensityParameterization(coefficients).scores(components))
        return SignalBackgroundDataset(x[:, None], scores, intensity)

    train, validation, test = _three_splits(generate, seed=seed, sizes=sizes)  # type: ignore[arg-type]
    return SignalBackgroundProblem(
        name="signal_background_shape",
        title="Signal fraction with background-shape nuisance",
        n_bins=n_bins,
        interest=(0,),
        nuisance=tuple(range(1, 1 + n_background)),
        bounds=np.asarray([[0.0, 1.0]]),
        coefficients=coefficients,
        component_names=component_names,
        signal_density=signal_density,
        background_densities=background_densities,
        train=train,
        validation=validation,
        test=test,
    )


def separable_1d_direction(scores: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Return the leading weighted-variance direction of a score matrix.

    `separable_1d_projection` is exactly ``scores @ direction`` for this
    direction. Callers that must project a second split (a held-out sample)
    onto the *same* axis need the direction itself, not just one split's
    projected column, so this helper exposes it.

    Parameters
    ----------
    scores
        Finite score matrix with shape ``[N, P]``.
    weights
        Nonnegative weights with shape ``[N]``. Uniform weights are used
        when omitted.

    Returns
    -------
    numpy.ndarray
        A unit-norm direction with shape ``[P]``. The sign is whatever
        `numpy.linalg.eigh` returns, which is deterministic for a given
        input but carries no meaning.

    Raises
    ------
    ValueError
        If `scores` is empty or malformed, or `weights` does not sum to a
        positive value.
    """
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("scores must have non-empty shape [N, P]")
    w = np.ones(values.shape[0]) if weights is None else np.asarray(weights, dtype=float)
    total = float(w.sum())
    if total <= 0:
        raise ValueError("weights must have a positive sum")
    mean = (w[:, None] * values).sum(axis=0) / total
    centered = values - mean
    covariance = (centered * w[:, None]).T @ centered / total
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return eigenvectors[:, np.argmax(eigenvalues)]


def separable_1d_projection(scores: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Project multi-column scores onto their leading weighted-variance direction.

    A simple, deterministic dimensionality reduction for the scalar dynamic
    program baseline (`scorequant.ScalarDPConfig`), which requires a single
    score coordinate. This is a plain weighted-PCA projection, not an
    information-theoretic guarantee: it is a baseline input, not a claim
    about retained Fisher information.

    Parameters
    ----------
    scores
        Finite score matrix with shape ``[N, P]``.
    weights
        Nonnegative weights with shape ``[N]``. Uniform weights are used
        when omitted.

    Returns
    -------
    numpy.ndarray
        A single-column score matrix with shape ``[N, 1]``.
    """
    values = np.asarray(scores, dtype=float)
    return (values @ separable_1d_direction(values, weights))[:, None]


PROBLEMS = {
    "gaussian_location": gaussian_location,
    "spectral_templates": spectral_templates,
    "spatial_sources": spatial_sources,
    "two_parameter_gaussian_mixture": two_parameter_gaussian_mixture,
}
