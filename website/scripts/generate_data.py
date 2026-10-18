"""Write the criterion/solver matrix into the Markdown files that show it."""

from __future__ import annotations

import re
from pathlib import Path

# ``_SOLVER_TABLE`` is a private module-level constant, not part of ``scorequant``'s public
# surface. The generator is deliberately coupled to it anyway: it is the single source of
# truth for which (config, criterion) pairs ``optimize_partition`` and ``fit_quantizer``
# accept, and being coupled to the registry -- rather than to a hand-copied Markdown table --
# is the entire point of generating the matrix instead of maintaining it by hand.
from scorequant.api import _SOLVER_TABLE

ROOT = Path(__file__).resolve().parents[2]

# The generated criterion/solver matrix marks its home in three Markdown files with these
# markers; content between them is rewritten on every run and must never be hand-edited.
_SOLVER_MATRIX_START = (
    "<!-- generated: solver-matrix (do not edit by hand; run `pnpm generate:data`) -->"
)
_SOLVER_MATRIX_END = "<!-- /generated: solver-matrix -->"

# One-sentence contract prose per configuration. This is not derivable from ``_SOLVER_TABLE``
# -- the registry only knows which criteria a config accepts, not how its solver behaves -- so
# it is hand-maintained here, keyed by configuration class name, and carried into the
# generated README table's trailing "Contract" column.
_SOLVER_CONTRACTS: dict[str, str] = {
    "DExchangeConfig": (
        "Exact positive-gain relocations; monotone objective; terminates exchange-stable"
    ),
    "MahalanobisLloydConfig": (
        "A batch is adopted only if the exactly rebuilt objective improves; "
        "optional exact-exchange guard"
    ),
    "SoftVoronoiConfig": (
        "Differentiable soft optimization then hardening, with the hardening gap reported"
    ),
    "KMeansConfig": "Weighted $k$-means in whitened score space",
    "ScalarDPConfig": "The exact global interval solution for rank-one score space",
}


def _solver_matrix() -> list[dict[str, object]]:
    """Read the criterion/solver compatibility matrix from ``scorequant.api._SOLVER_TABLE``.

    This is the single generated source for the table that is otherwise hand-copied in
    ``docs/guide.md``, ``docs/api.md`` and ``README.md``: one row per
    configuration type, in the registry's own declaration order, naming which criteria each
    of the two public tasks accepts for it.
    """
    return [
        {
            "configuration": config_type.__name__,
            "partitionCriteria": [criterion.__name__ for criterion in spec.partition_criteria],
            "quantizerCriteria": [criterion.__name__ for criterion in spec.quantizer_criteria],
            "contract": _SOLVER_CONTRACTS[config_type.__name__],
        }
        for config_type, spec in _SOLVER_TABLE.items()
    ]


def _format_criteria(names: list[str]) -> str:
    return ", ".join(f"`{name}`" for name in names) if names else "—"


def _render_solver_matrix(rows: list[dict[str, object]], *, with_contract: bool) -> str:
    """Render ``rows`` (as produced by :func:`_solver_matrix`) as a Markdown table."""
    headers = ["Configuration", "`optimize_partition`", "`fit_quantizer`"]
    if with_contract:
        headers.append("Contract")
    lines = [
        f"| {' | '.join(headers)} |",
        f"| {' | '.join(['---'] * len(headers))} |",
    ]
    for row in rows:
        cells = [
            f"`{row['configuration']}`",
            _format_criteria(list(row["partitionCriteria"])),  # type: ignore[arg-type]
            _format_criteria(list(row["quantizerCriteria"])),  # type: ignore[arg-type]
        ]
        if with_contract:
            cells.append(str(row["contract"]))
        lines.append(f"| {' | '.join(cells)} |")
    return "\n".join(lines)


def _rewrite_generated_region(path: Path, table: str) -> None:
    """Replace the content between the solver-matrix markers in ``path`` with ``table``."""
    text = path.read_text()
    pattern = re.compile(
        re.escape(_SOLVER_MATRIX_START) + r".*?" + re.escape(_SOLVER_MATRIX_END),
        flags=re.DOTALL,
    )
    replacement = f"{_SOLVER_MATRIX_START}\n{table}\n{_SOLVER_MATRIX_END}"
    new_text, count = pattern.subn(replacement, text)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one solver-matrix generated region in {path}, found {count}"
        )
    path.write_text(new_text)


def _rewrite_solver_matrix_docs(rows: list[dict[str, object]]) -> None:
    """Rewrite the generated solver-matrix region in every consuming Markdown file."""
    plain_table = _render_solver_matrix(rows, with_contract=False)
    contract_table = _render_solver_matrix(rows, with_contract=True)
    _rewrite_generated_region(ROOT / "docs" / "guide.md", plain_table)
    _rewrite_generated_region(ROOT / "README.md", contract_table)
    _rewrite_generated_region(ROOT / "docs" / "api.md", contract_table)


def main() -> None:
    """Rewrite every generated solver-matrix region from the registry."""
    _rewrite_solver_matrix_docs(_solver_matrix())


if __name__ == "__main__":
    main()
