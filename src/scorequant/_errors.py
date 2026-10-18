"""Library exception hierarchy. Stdlib-only; imported by every module that raises."""

from __future__ import annotations


class ScoreQuantError(Exception):
    """Base of every exception ScoreQuant raises deliberately."""


class ContractError(ScoreQuantError, ValueError):
    """The caller violated an input, shape, range, or pairing contract.

    Detectable from the arguments alone; the remedy is to change the call.
    Subclasses ``ValueError`` so existing ``except ValueError`` handlers keep working.
    """


class RefusalError(ScoreQuantError, RuntimeError):
    """The library declines a valid request because a theorem-backed condition fails on the data.

    Parameters
    ----------
    message
        Plain-English refusal, unchanged from the pre-hierarchy message text.
    counterexample
        Id of the exact counterexample that forces the refusal; the repository
        keeps it as ``tests/fixtures/counterexamples/<id>.json``. Always a
        string literal at the raise site.
    """

    counterexample: str

    def __init__(self, message: str, counterexample: str) -> None:
        super().__init__(message, counterexample)  # positional args keep pickling/copy intact
        self.counterexample = counterexample

    def __str__(self) -> str:
        return f"{self.args[0]} [{self.counterexample}]"
