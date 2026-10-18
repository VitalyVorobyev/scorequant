"""Recompute, in exact rational arithmetic, every number the paper quotes from a fixture.

Standard library only. Run from the repository root::

    python research/paper/check_numbers.py
"""

from __future__ import annotations

import json
import math
from fractions import Fraction as F
from itertools import product
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "counterexamples"

Vector = list[F]
Matrix = list[list[F]]


def load(name: str) -> tuple[Matrix, Vector, dict[str, object]]:
    """Return the exact scores, weights and raw record of one fixture."""
    data = json.loads((FIXTURES / f"{name}.json").read_text())
    key = "scores_centered_exact" if "scores_centered_exact" in data else "scores"
    scores = [[F(str(x)) for x in row] for row in data[key]]
    return scores, [F(x) for x in data["weights"]], data


def cells(scores: Matrix, weights: Vector, labels: list[int], k: int) -> tuple[Matrix, Matrix]:
    """Return the cell centroids and the binned information ``sum_c W_c mu_c mu_c^T``."""
    d = len(scores[0])
    mass = [
        sum((w for w, z in zip(weights, labels, strict=True) if z == c), F(0)) for c in range(k)
    ]
    mean = [
        [
            sum(
                (w * s[j] for s, w, z in zip(scores, weights, labels, strict=True) if z == c),
                F(0),
            )
            / mass[c]
            for j in range(d)
        ]
        for c in range(k)
    ]
    info = [
        [sum(mass[c] * mean[c][i] * mean[c][j] for c in range(k)) for j in range(d)]
        for i in range(d)
    ]
    return mean, info


def det(a: Matrix) -> F:
    """Return the determinant of a 1x1 or 2x2 matrix."""
    return a[0][0] if len(a) == 1 else a[0][0] * a[1][1] - a[0][1] * a[1][0]


def inverse(a: Matrix) -> Matrix:
    """Return the inverse of a 1x1 or 2x2 matrix."""
    if len(a) == 1:
        return [[1 / a[0][0]]]
    return [[a[1][1] / det(a), -a[0][1] / det(a)], [-a[1][0] / det(a), a[0][0] / det(a)]]


def dist(h: Matrix, s: Vector, mu: Vector) -> F:
    """Return the quadratic distance ``(s - mu)^T h (s - mu)``."""
    u = [x - y for x, y in zip(s, mu, strict=True)]
    return sum((u[i] * h[i][j] * u[j] for i in range(len(u)) for j in range(len(u))), F(0))


def nearest(h: Matrix, s: Vector, means: Matrix) -> int:
    """Return the index of the nearest centroid in the metric ``h``."""
    return min(range(len(means)), key=lambda c: dist(h, s, means[c]))


def lloyd() -> dict[str, object]:
    """Section 5.1: one I^{-1}-Lloyd step from the fixture's labels."""
    scores, weights, fx = load("CE-D-LLOYD-001")
    z = list(fx["labels_before"])
    mu, info = cells(scores, weights, z, 3)
    h = inverse(info)
    z1 = [nearest(h, s, mu) for s in scores]
    mu1, info1 = cells(scores, weights, z1, 3)
    distortion = [
        sum(w * dist(h, s, m[c]) for s, w, c in zip(scores, weights, lab, strict=True))
        for lab, m in ((z, mu), (z1, mu1))
    ]
    return {
        "labels_after": z1,
        "det_before": det(info),
        "det_after": det(info1),
        "logdet_before": math.log(det(info)),
        "logdet_after": math.log(det(info1)),
        "changed": sum(a != b for a, b in zip(z, z1, strict=True)),
        "distortion": distortion,
        "tangent": distortion[0] - distortion[1],
    }


def voronoi_converse() -> dict[str, object]:
    """Section 5.2: a strict Voronoi labelling that is not exchange stable."""
    scores, weights, fx = load("CE-D-VORONOI-CONVERSE-001")
    z = list(fx["labels_before"])
    mu, info = cells(scores, weights, z, 2)
    _, info1 = cells(scores, weights, list(fx["labels_after_or_optimum"]), 2)
    h = inverse(info)
    return {
        "centroids": [m[0] for m in mu],
        "strict_voronoi": all(
            dist(h, s, mu[c]) < dist(h, s, mu[1 - c]) for s, c in zip(scores, z, strict=True)
        ),
        "info_before": info[0][0],
        "info_after": info1[0][0],
    }


def profiled_optimum() -> dict[str, object]:
    """Section 7: the global profiled optimum against its own efficient semimetric."""
    scores, weights, _ = load("CE-DS-GLOBAL-GEOMETRY-001")

    def profiled(labels: list[int]) -> F | None:
        info = cells(scores, weights, labels, 3)[1]
        return None if info[1][1] == 0 else info[0][0] - info[0][1] ** 2 / info[1][1]

    partitions = [
        list(z)
        for z in product(range(3), repeat=8)
        if z[0] == 0 and set(z) == {0, 1, 2} and all(z[i] <= max(z[:i]) + 1 for i in range(1, 8))
    ]
    values = [(profiled(z), z) for z in partitions]
    ranked = sorted(((v, z) for v, z in values if v is not None), reverse=True)
    best, z = ranked[0]
    mu, info = cells(scores, weights, z, 3)
    h = inverse(info)
    g = [[h[0][0], h[0][1]], [h[1][0], h[1][1] - 1 / info[1][1]]]
    margins = {
        i: dist(g, s, mu[z[i]]) - min(dist(g, s, m) for m in mu) for i, s in enumerate(scores)
    }
    return {
        "partitions": len(partitions),
        "regular": len(ranked),
        "labels": z,
        "best": best,
        "runner_up": ranked[1][0],
        "info": info,
        "projection": -info[0][1] / info[1][1],
        "violations": {i: m for i, m in margins.items() if m > 0},
    }


def retention() -> list[F]:
    """Section 8: same labels and ranking, different reported retention."""
    reported = []
    for proxy in ([-2, -1, 1, 2], [F(-11, 10), F(-9, 10), F(9, 10), F(11, 10)], [-2, -1, 1, 100]):
        s = [[F(x)] for x in proxy]
        full = sum(x[0] ** 2 for x in s) / 4
        reported.append(cells(s, [F(1, 4)] * 4, [0, 0, 1, 1], 2)[1][0][0] / full)
    return reported


if __name__ == "__main__":
    for name, result in (
        ("Lloyd", lloyd()),
        ("Voronoi converse", voronoi_converse()),
        ("D_s", profiled_optimum()),
        ("retention", retention()),
    ):
        print(f"{name}: {result}")
