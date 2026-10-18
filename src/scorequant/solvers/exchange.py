"""Exact exchange engine shared by both finite D criteria.

A *scan* is one complete evaluation of every admissible single-row relocation;
a scan either accepts work or certifies exchange stability. With
``batch_moves`` a single scan may relocate many rows at once, so accepted moves
and scans are different quantities. The criterion algebra lives in
``scorequant._exchange``; this module owns seeding, scanning, acceptance and
stopping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scorequant._chunking import chunk_rows_for
from scorequant._errors import ContractError, RefusalError
from scorequant._exchange import ExchangeObjective, ExchangeState, cell_statistics, relocation
from scorequant._execution import random_permutation
from scorequant._execution import xp as jnp
from scorequant._predict import assign_nearest
from scorequant.config import DExchangeConfig, KMeansConfig, PartitionConfig

from .kmeans import weighted_kmeans

# Strict positive-gain acceptance cannot revisit a labeling, so the exchange
# terminates without any scan cap. This bound only stops a run that a numerical
# pathology would otherwise keep alive; it is never reached by a healthy fit.
_SAFETY_SCAN_LIMIT = 100_000

# A guarded batch halves its size on rejection, so this bounds the retries of a
# single scan for any sample that fits in memory.
_MAX_BATCH_SHRINKS = 32

# Smallest improving set worth a guarded batch, and the largest share of a
# cell's weight that one batch may move in or out of it.
_MIN_BATCH_ROWS = 8
_BATCH_MASS_FRACTION = 0.25


@dataclass(frozen=True, slots=True)
class Move:
    """One candidate relocation of a row into another cell."""

    row: int
    destination: int
    gain: float


@dataclass(frozen=True, slots=True)
class ScanOutcome:
    """Result of one complete candidate scan."""

    best: Move | None
    row_gains: np.ndarray | None
    row_destinations: np.ndarray | None

    @property
    def best_gain(self) -> float:
        """Return the largest admissible gain, or ``-inf`` when no move exists."""
        return float("-inf") if self.best is None else self.best.gain


@dataclass(frozen=True, slots=True)
class ExchangeRun:
    """Terminal state and diagnostics of one exchange or guarded-batch restart."""

    labels: jnp.ndarray
    state: ExchangeState
    objective_history: tuple[float, ...]
    accepted_moves: int
    scans: int
    exchange_stable: bool
    best_remaining_gain: float
    lloyd_iterations: int = 0
    accepted_lloyd_steps: int = 0


def gain_tolerance(config: PartitionConfig) -> float:
    """Return the gain tolerance of a configuration the solver entry has resolved."""
    if config.gain_tolerance is None:
        raise ContractError("gain_tolerance must be resolved for the working dtype before solving")
    return config.gain_tolerance


def optimize_exchange(
    *,
    points: jnp.ndarray,
    coordinates: jnp.ndarray,
    weights: jnp.ndarray,
    n_bins: int,
    objective: ExchangeObjective,
    config: DExchangeConfig,
    initial_labels: jnp.ndarray | None = None,
) -> ExchangeRun:
    """Run every seeded exchange restart and keep the best exact objective.

    Supplied labels replace the seeding of the first restart only, so ``init``
    and ``initializer_restarts`` still govern restarts one and above and an
    initializer can be compared against ordinary seeding inside a single call.
    """
    best: ExchangeRun | None = None
    for restart in range(config.solver_restarts):
        labels = (
            initial_labels
            if restart == 0 and initial_labels is not None
            else _initial_labels(coordinates, weights, n_bins, config, restart)
        )
        run = run_exchange(points, weights, labels, n_bins, objective, config)
        # Strict comparison makes the earliest restart win exact ties.
        if best is None or run.state.objective > best.state.objective:
            best = run
    if best is None:
        raise ContractError("solver_restarts must be at least one")
    return best


def _initial_labels(
    coordinates: jnp.ndarray,
    weights: jnp.ndarray,
    n_bins: int,
    config: DExchangeConfig,
    restart: int,
) -> jnp.ndarray:
    seed = config.seed + restart
    if config.init == "random":
        # A permuted balanced labeling keeps every requested cell nonempty.
        permutation = random_permutation(seed, coordinates.shape[0])
        return (jnp.arange(coordinates.shape[0]) % n_bins).astype(jnp.int32)[permutation]
    return kmeans_labels(
        coordinates,
        weights,
        n_bins,
        rank_rtol=config.rank_rtol,
        seed=seed,
        initializer_restarts=config.initializer_restarts,
    )


def kmeans_labels(
    coordinates: jnp.ndarray,
    weights: jnp.ndarray,
    n_bins: int,
    *,
    rank_rtol: float | None,
    seed: int,
    initializer_restarts: int,
) -> jnp.ndarray:
    """Seed one solver with deterministic weighted k-means++ labels."""
    initializer = weighted_kmeans(
        coordinates,
        weights,
        n_bins,
        KMeansConfig(
            whiten=False,
            rank_rtol=rank_rtol,
            seed=seed,
            solver_restarts=initializer_restarts,
            max_iter=100,
            tolerance=1e-8,
            record_every=100,
        ),
    )
    return jnp.asarray(np.asarray(assign_nearest(coordinates, initializer.centers), dtype=np.int32))


def run_exchange(
    points: jnp.ndarray,
    weights: jnp.ndarray,
    labels: jnp.ndarray,
    n_bins: int,
    objective: ExchangeObjective,
    config: DExchangeConfig,
) -> ExchangeRun:
    """Scan, accept exact positive-gain work, and stop at exchange stability."""
    label_array = np.asarray(labels, dtype=np.int32).copy()
    state = objective.init_state(cell_statistics(points, weights, labels, n_bins))
    history = [state.objective]
    accepted_moves = 0
    scans = 0
    exchange_stable = False
    best_remaining_gain = float("-inf")
    tolerance = gain_tolerance(config)
    # first_improvement deliberately stops each scan early, so it never produces
    # the complete per-row table a guarded batch needs.
    batched = config.batch_moves and not config.first_improvement
    batch_size = label_array.shape[0]
    limit = _SAFETY_SCAN_LIMIT if config.max_scans is None else config.max_scans
    # True once a single move has updated the state by rank-two algebra; a
    # guarded batch is accepted on an exactly rebuilt state and resets it.
    incremental = False

    while scans < limit:
        scans += 1
        outcome = scan(points, weights, label_array, state, objective, config, rows=batched)
        best_remaining_gain = outcome.best_gain
        if outcome.best is None or outcome.best.gain <= tolerance:
            exchange_stable = True
            break
        accepted = (
            _accept_batch(
                points,
                weights,
                label_array,
                n_bins,
                state,
                objective,
                config,
                outcome=outcome,
                batch_size=batch_size,
            )
            if batched
            else None
        )
        if accepted is None:
            move = outcome.best
            state = objective.apply_move(
                state, relocation(points, weights, label_array, state, move.row, move.destination)
            )
            label_array[move.row] = move.destination
            accepted_moves += 1
            incremental = True
        else:
            state, label_array, moved = accepted
            accepted_moves += moved
            batch_size = min(2 * moved, label_array.shape[0])
            incremental = False
        history.append(state.objective)

    if not exchange_stable:
        scans += 1
        outcome = scan(points, weights, label_array, state, objective, config, rows=False)
        best_remaining_gain = outcome.best_gain
        exchange_stable = best_remaining_gain <= tolerance

    if incremental:
        # Chained rank-two updates accumulate rounding in the cell means, the
        # inverse and the log determinant. The reported objective, centres and
        # metric must be those of the returned labels, so they are rebuilt
        # exactly from them; the last history entry then states the same value.
        state = objective.init_state(
            cell_statistics(points, weights, jnp.asarray(label_array), n_bins)
        )
        history[-1] = state.objective

    return ExchangeRun(
        labels=jnp.asarray(label_array),
        state=state,
        objective_history=tuple(history),
        accepted_moves=accepted_moves,
        scans=scans,
        exchange_stable=exchange_stable,
        best_remaining_gain=best_remaining_gain,
    )


def scan(
    points: jnp.ndarray,
    weights: jnp.ndarray,
    labels: np.ndarray,
    state: ExchangeState,
    objective: ExchangeObjective,
    config: DExchangeConfig,
    *,
    rows: bool,
) -> ScanOutcome:
    """Evaluate every admissible relocation in memory-bounded row chunks."""
    n_rows = points.shape[0]
    chunk_rows = chunk_rows_for(points, state.weights.shape[0])
    label_array = jnp.asarray(labels)
    row_gains = np.full(n_rows, -np.inf) if rows else None
    row_destinations = np.zeros(n_rows, dtype=np.int64) if rows else None
    best: Move | None = None
    for start in range(0, n_rows, chunk_rows):
        stop = min(start + chunk_rows, n_rows)
        gains = np.asarray(
            objective.chunk_gains(
                state, points[start:stop], weights[start:stop], label_array[start:stop]
            )
        )
        if config.first_improvement:
            improving = gains > gain_tolerance(config)
            improving_rows = np.flatnonzero(improving.any(axis=1))
            if improving_rows.size:
                row = int(improving_rows[0])
                destination = int(np.argmax(improving[row]))
                return ScanOutcome(
                    Move(start + row, destination, float(gains[row, destination])), None, None
                )
        destinations = np.argmax(gains, axis=1)
        best_gains = gains[np.arange(stop - start), destinations]
        if row_gains is not None and row_destinations is not None:
            row_gains[start:stop] = best_gains
            row_destinations[start:stop] = destinations
        row = int(np.argmax(best_gains))
        gain = float(best_gains[row])
        if np.isfinite(gain) and (best is None or gain > best.gain):
            best = Move(start + row, int(destinations[row]), gain)
    return ScanOutcome(best, row_gains, row_destinations)


def _accept_batch(
    points: jnp.ndarray,
    weights: jnp.ndarray,
    labels: np.ndarray,
    n_bins: int,
    state: ExchangeState,
    objective: ExchangeObjective,
    config: DExchangeConfig,
    *,
    outcome: ScanOutcome,
    batch_size: int,
) -> tuple[ExchangeState, np.ndarray, int] | None:
    """Relocate many rows at once, accepting only a verified exact improvement.

    Every candidate batch is scored by rebuilding the criterion state exactly
    from the proposed labels, so acceptance never relies on accumulated
    increments and the exchange stays strictly monotone. A rejected batch is
    halved by gain rank and retried; when nothing survives, the caller applies
    the exact single best relocation instead.
    """
    if outcome.row_gains is None or outcome.row_destinations is None:
        return None
    improving = np.flatnonzero(outcome.row_gains > gain_tolerance(config))
    # Below this floor a batch cannot amortize its exact rebuild, and the few
    # surviving rows are strongly interacting boundary atoms whose certified
    # greedy relocation is both cheaper and better conditioned.
    if improving.size <= _MIN_BATCH_ROWS:
        return None
    order = improving[np.argsort(-outcome.row_gains[improving], kind="stable")]
    destinations = outcome.row_destinations[order]
    size = min(
        batch_size,
        _mass_budget_cut(
            np.asarray(weights)[order],
            labels[order],
            destinations,
            np.asarray(state.weights),
        ),
    )
    for _ in range(_MAX_BATCH_SHRINKS):
        if size <= 1:
            # The single best relocation has an exact positive gain; the caller
            # applies it through the cheaper rank-two path.
            return None
        trial_labels = labels.copy()
        trial_labels[order[:size]] = destinations[:size]
        trial = trial_state(points, weights, trial_labels, n_bins, objective)
        if trial is not None and trial.objective > state.objective + gain_tolerance(config):
            return trial, trial_labels, size
        size //= 2
    return None


def _mass_budget_cut(
    row_weights: np.ndarray,
    sources: np.ndarray,
    destinations: np.ndarray,
    cell_weights: np.ndarray,
) -> int:
    """Return how many gain-ranked rows a batch may relocate at once.

    The determinant lemma is exact for one relocation. A simultaneous batch
    displaces every touched cell mean in proportion to the mass it moves, so
    the batch is cut before any cell would gain or lose more than a fixed
    fraction of its weight. Without this cut the guarded batch still stays
    monotone, but it repeatedly overshoots into worse exchange-stable states.
    """
    cut = int(row_weights.shape[0])
    for cell in range(int(cell_weights.shape[0])):
        budget = _BATCH_MASS_FRACTION * float(cell_weights[cell])
        for touching in (sources == cell, destinations == cell):
            index = np.flatnonzero(touching)
            if index.size == 0:
                continue
            over = np.flatnonzero(np.cumsum(row_weights[index]) > budget)
            if over.size:
                cut = min(cut, int(index[over[0]]))
    return cut


def trial_state(
    points: jnp.ndarray,
    weights: jnp.ndarray,
    labels: np.ndarray,
    n_bins: int,
    objective: ExchangeObjective,
) -> ExchangeState | None:
    """Rebuild the exact state of a proposed labeling, or reject an infeasible one."""
    try:
        return objective.init_state(cell_statistics(points, weights, jnp.asarray(labels), n_bins))
    except (ContractError, RefusalError):
        # An emptied cell or a singular proposal is an ordinary batch rejection,
        # regardless of whether ``init_state`` reports it as a malformed input
        # (``ContractError``) or a theorem-backed refusal on this trial labeling
        # (``RefusalError``, e.g. a profiled trial whose binned information
        # would be degenerate): a discarded trial is never a request outcome,
        # so neither should surface past this internal rejection.
        return None
