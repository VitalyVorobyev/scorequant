"""Framework-neutral observation-to-score providers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ._errors import ContractError
from ._execution import canonical_array, execution_scope
from ._execution import xp as jnp
from ._typing import ArrayLike
from .config import ExecutionConfig
from .sources import ScoreProvenance, ScoreSchema


@dataclass(frozen=True, slots=True)
class ScoreFunction:
    """Wrap an exact or estimated callable ``[N, D] -> [N, P]``."""

    function: Callable[[ArrayLike], ArrayLike]
    provenance: ScoreProvenance = ScoreProvenance()
    schema: ScoreSchema | None = None

    def __post_init__(self) -> None:
        """Reject non-callable providers at construction time."""
        if not callable(self.function):
            raise TypeError("function must be callable")
        if self.schema is not None and not isinstance(self.schema, ScoreSchema):
            raise TypeError("schema must be a ScoreSchema")

    @execution_scope
    def score(
        self,
        observations: ArrayLike,
        *,
        execution: ExecutionConfig | None = None,
    ) -> np.ndarray:
        """Evaluate and validate a finite score matrix."""
        del execution
        values = jnp.asarray(self.function(observations))
        if values.ndim != 2 or values.shape[0] != jnp.asarray(observations).shape[0]:
            raise ContractError("score function must return shape [N, P]")
        if values.shape[1] == 0 or not bool(np.asarray(jnp.all(jnp.isfinite(values)))):
            raise ContractError("score function must return finite non-empty scores")
        return canonical_array(values)


@runtime_checkable
class ScoreProvider(Protocol):
    """The observation-to-score contract every fitting route goes through.

    ScoreQuant does not care how a score was obtained -- an analytic
    derivative, autodiff, a component model, a density-ratio estimator, or an
    external inference package -- only that observations can be mapped to a
    finite score matrix and that the mapping says what it is. Those are the two
    members here, so a caller can supply their own object rather than wrapping
    it:

    .. code-block:: python

        class MyExternalScore:
            provenance = ScoreProvenance(kind="estimated_ratio")

            def score(self, observations):
                return my_package.evaluate(observations)

    The built-in providers -- :class:`ScoreFunction`,
    :class:`LinearComponentScore`, and the ratio providers in
    :mod:`scorequant.ratios` -- are convenience implementations of this
    protocol, not a closed list of what is allowed.

    A provider may additionally expose ``schema``, a
    :class:`~scorequant.ScoreSchema` naming its score columns; it is used when
    present and is not part of the required contract.

    Notes
    -----
    ``score`` takes observations alone. The execution backend is ambient
    context established by the public task, not an argument a provider has to
    thread through, so an external implementation needs no knowledge of
    :class:`~scorequant.ExecutionConfig`.
    """

    @property
    def provenance(self) -> ScoreProvenance:
        """Describe how this score representation was obtained."""
        ...

    def score(self, observations: ArrayLike) -> ArrayLike:
        """Map observations with shape ``[N, D]`` to scores with shape ``[N, P]``."""
        ...


def validate_provider(provider: object) -> None:
    """Reject an object that cannot act as a score provider.

    ``runtime_checkable`` only proves the attributes exist, so the provenance
    type is checked here too: a provider whose ``provenance`` is not a
    :class:`~scorequant.ScoreProvenance` would otherwise reach the result and
    silently decide whether exact-Fisher language is justified.
    """
    if not isinstance(provider, ScoreProvider):
        missing = [name for name in ("score", "provenance") if not hasattr(provider, name)]
        raise TypeError(
            "provider must implement the ScoreProvider protocol; "
            f"{type(provider).__name__} is missing {', '.join(missing)}"
        )
    if not isinstance(provider.provenance, ScoreProvenance):
        raise TypeError(
            "provider.provenance must be a ScoreProvenance, not "
            f"{type(provider.provenance).__name__}"
        )
    if not callable(provider.score):
        raise TypeError("provider.score must be callable")
