"""Independent audit instrument for O7 (RETENTION-PLUGIN-CLT-FROZEN-VECTOR).

Built by the AUDIT-RETENTION-PLUGIN-VECTOR session without reading the
researcher's ``py/retention_plugin_vector.py``; the researcher's
``popref.json`` was opened, after the audit's own Hermite references existed,
only to copy the parameters of the bounded-mixture law and of the
library-fitted rule (recorded in ``popref.json`` of this audit). Stages:

``exact``
    ``fractions.Fraction`` identities in d = 1, 2, 3 on adversarial samples
    (ties, duplicate atoms, a singleton cell, a declared-empty cell, K = d on a
    centred sample, K < d, singular V_hat): the within-cell-scatter form of
    V_hat - I_hat_Z, 0 <= ratio <= 1, sum psi_hat = 0, the order-four
    within-cell-moment expansion of sigma_hat^2, exact Gateaux quotients of the
    determinant ratio against r * B with a halving remainder, the d = 1
    reduction to O6, affine invariance, library agreement and the two
    ``rank_rtol`` caveats; then random rational atomic laws: E[psi] = 0,
    E[N_1] = E[N_2] = d, the population Gateaux reduction, and the ellipsoid
    polynomial identity behind O7.4(a).
``degenerate``
    The sigma^2 = 0 laws: the fixture's two-atom law and its atomless arc law,
    a d = 3 sphere law of the same kind, the eta_D = 1 law with its singular
    samples (boundary failure H1), and exhaustive sample compositions on the
    two-atom law.
``popref``
    Population references by two independent quadrature routes (QUADPACK and
    cut-aligned composite Gauss-Legendre) for the Hermite laws, the endpoint
    laws with their null-direction geometry, the bounded mixture and the
    library-fitted rule.
``coverage``
    Fresh-seed Monte Carlo of the Wald interval on every regular law and the
    arc law.
``singular``
    The O7.4(b) limit law simulated directly from its Gaussian ingredients
    against n^{(d-r)/d} eta_hat_D at finite n.
``fixtures``
    Serialises the boundary counterexample to ``COUNTEREXAMPLES/``.

Every stage writes a provenance-stamped JSON record under
``AUDITS/artifacts/AUDIT-RETENTION-PLUGIN-VECTOR-001/``. Run with
``JAX_ENABLE_X64=1`` for the library stages.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

RESEARCH = Path(__file__).resolve().parents[1]
ROOT = RESEARCH.parent
ARTIFACTS = RESEARCH / "AUDITS" / "artifacts" / "AUDIT-RETENTION-PLUGIN-VECTOR-001"
FIXTURE_ID = "CE-O7-UNIT-RETENTION-SINGULAR-SAMPLE-001"
Z975 = 1.959963984540054

# Recorded by the researcher (KNOWN_RESULTS/10-oracle.md, O7.7); six decimals.
RECORDED = {
    "hermite2": {"eta": 0.690666, "sigma": 0.673175, "fourth": 83.0},
    "hermite3": {"eta": 0.629969, "sigma": 1.624099, "fourth": 4259.0},
    "mixture": {"eta": 0.330317, "sigma": 0.621795, "fourth": 22.2},
    "library": {"eta": 0.743363, "sigma": 0.557001},
}


# --------------------------------------------------------------------------
# provenance and serialisation


def provenance(mode: str, parameters: dict[str, object]) -> dict[str, object]:
    """Return the reproducibility record required by the numerical protocol."""
    script = Path(__file__)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, check=True, text=True
    ).stdout.strip()
    return {
        "mode": mode,
        "parameters": parameters,
        "git_revision": revision,
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def fr(value: Fraction | int) -> str:
    return str(Fraction(value))


def frm(matrix: list[list[Fraction]]) -> list[list[str]]:
    return [[fr(x) for x in row] for row in matrix]


def write_record(name: str, record: dict[str, object]) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / f"{name}.json"
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------
# exact linear algebra (small d)

Matrix = list[list[Fraction]]
Vector = list[Fraction]


def zeros(d: int) -> Matrix:
    return [[Fraction(0)] * d for _ in range(d)]


def identity(d: int) -> Matrix:
    return [[Fraction(int(i == j)) for j in range(d)] for i in range(d)]


def det(a: Matrix) -> Fraction:
    """Determinant by fraction-free elimination on a copy."""
    n = len(a)
    m = [row[:] for row in a]
    result = Fraction(1)
    for col in range(n):
        pivot = next((r for r in range(col, n) if m[r][col] != 0), None)
        if pivot is None:
            return Fraction(0)
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
            result = -result
        result *= m[col][col]
        for r in range(col + 1, n):
            factor = m[r][col] / m[col][col]
            for c in range(col, n):
                m[r][c] -= factor * m[col][c]
    return result


def inv(a: Matrix) -> Matrix:
    """Inverse by Gauss-Jordan elimination; raises on a singular matrix."""
    n = len(a)
    m = [row[:] + identity(n)[i] for i, row in enumerate(a)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if m[r][col] != 0), None)
        if pivot is None:
            raise ZeroDivisionError("singular matrix")
        m[col], m[pivot] = m[pivot], m[col]
        scale = m[col][col]
        m[col] = [x / scale for x in m[col]]
        for r in range(n):
            if r != col and m[r][col] != 0:
                factor = m[r][col]
                m[r] = [x - factor * y for x, y in zip(m[r], m[col], strict=True)]
    return [row[n:] for row in m]


def matvec(a: Matrix, x: Vector) -> Vector:
    return [sum(a[i][j] * x[j] for j in range(len(x))) for i in range(len(a))]


def dot(x: Vector, y: Vector) -> Fraction:
    return sum(u * v for u, v in zip(x, y, strict=True))


def quad(a: Matrix, x: Vector, y: Vector) -> Fraction:
    return dot(x, matvec(a, y))


def outer(x: Vector, y: Vector) -> Matrix:
    return [[u * v for v in y] for u in x]


def madd(a: Matrix, b: Matrix, scale: Fraction = Fraction(1)) -> Matrix:
    return [[u + scale * v for u, v in zip(ra, rb, strict=True)] for ra, rb in zip(a, b, strict=True)]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def transpose(a: Matrix) -> Matrix:
    return [list(col) for col in zip(*a, strict=True)]


def rank(a: Matrix) -> int:
    n = len(a)
    m = [row[:] for row in a]
    r = 0
    for col in range(len(a[0])):
        pivot = next((i for i in range(r, n) if m[i][col] != 0), None)
        if pivot is None:
            continue
        m[r], m[pivot] = m[pivot], m[r]
        for i in range(n):
            if i != r and m[i][col] != 0:
                factor = m[i][col] / m[r][col]
                m[i] = [x - factor * y for x, y in zip(m[i], m[r], strict=True)]
        r += 1
        if r == n:
            break
    return r


# --------------------------------------------------------------------------
# the O7 functional on a weighted atom sample / atomic law

Atom = tuple[Vector, int, Fraction]  # (score, label, weight)


def law_moments(atoms: list[Atom], n_bins: int) -> dict[str, object]:
    """Cell probabilities, cell moments, V and I_Z (0/0 := 0 on empty cells)."""
    d = len(atoms[0][0])
    total = sum(w for _, _, w in atoms)
    p = [sum(w for _, z, w in atoms if z == b) / total for b in range(n_bins)]
    m = [
        [sum(w * s[i] for s, z, w in atoms if z == b) / total for i in range(d)]
        for b in range(n_bins)
    ]
    v = zeros(d)
    for s, _, w in atoms:
        v = madd(v, outer(s, s), w / total)
    i_z = zeros(d)
    for b in range(n_bins):
        if p[b] > 0:
            i_z = madd(i_z, outer(m[b], m[b]), 1 / p[b])
    c = [[x / p[b] for x in m[b]] if p[b] > 0 else [Fraction(0)] * d for b in range(n_bins)]
    return {"d": d, "p": p, "m": m, "c": c, "V": v, "I_Z": i_z}


def det_ratio(mom: dict[str, object]) -> Fraction:
    """The everywhere-defined functional r = phi^d: 0 when det V = 0 or ratio <= 0."""
    dv = det(mom["V"])
    if dv == 0:
        return Fraction(0)
    ratio = det(mom["I_Z"]) / dv
    return ratio if ratio > 0 else Fraction(0)


def bracket(mom: dict[str, object], s: Vector, b: int) -> Fraction:
    """B(s, b) = 2 s^T I_Z^{-1} c_b - c_b^T I_Z^{-1} c_b - s^T V^{-1} s.

    psi = (eta_D / d) B and psi_r = r B for the determinant ratio r = eta_D^d.
    """
    iz_inv, v_inv = inv(mom["I_Z"]), inv(mom["V"])
    c = mom["c"][b]
    return 2 * quad(iz_inv, s, c) - quad(iz_inv, c, c) - quad(v_inv, s, s)


def law_summary(atoms: list[Atom], n_bins: int) -> dict[str, object]:
    """Exact population summary of an atomic law: r, B at every atom, E B^2."""
    mom = law_moments(atoms, n_bins)
    total = sum(w for _, _, w in atoms)
    r = det_ratio(mom)
    b_values = [bracket(mom, s, z) for s, z, _ in atoms]
    mean_b = sum(w * bv for (_, _, w), bv in zip(atoms, b_values, strict=True)) / total
    mean_b2 = sum(w * bv * bv for (_, _, w), bv in zip(atoms, b_values, strict=True)) / total
    iz_inv, v_inv = inv(mom["I_Z"]), inv(mom["V"])
    n1 = [2 * quad(iz_inv, s, mom["c"][z]) - quad(iz_inv, mom["c"][z], mom["c"][z]) for s, z, _ in atoms]
    n2 = [quad(v_inv, s, s) for s, _, _ in atoms]
    mean_n1 = sum(w * x for (_, _, w), x in zip(atoms, n1, strict=True)) / total
    mean_n2 = sum(w * x for (_, _, w), x in zip(atoms, n2, strict=True)) / total
    var_n1 = sum(w * (x - mean_n1) ** 2 for (_, _, w), x in zip(atoms, n1, strict=True)) / total
    var_n2 = sum(w * (x - mean_n2) ** 2 for (_, _, w), x in zip(atoms, n2, strict=True)) / total
    cov = (
        sum(w * (x - mean_n1) * (y - mean_n2) for (_, _, w), x, y in zip(atoms, n1, n2, strict=True))
        / total
    )
    return {
        "moments": mom,
        "ratio": r,
        "B": b_values,
        "E_B": mean_b,
        "E_B2": mean_b2,
        "E_N1": mean_n1,
        "E_N2": mean_n2,
        "covariance_form": var_n1 - 2 * cov + var_n2,
    }


def sample_atoms(scores: list[Vector], labels: list[int], weights: list[Fraction] | None = None) -> list[Atom]:
    n = len(scores)
    if weights is None:
        weights = [Fraction(1, n)] * n
    return [(s, z, w) for s, z, w in zip(scores, labels, weights, strict=True)]


def sigma2_hat_from_moments(atoms: list[Atom], n_bins: int, mom: dict[str, object]) -> Fraction:
    """E B^2 written as a polynomial in within-cell moments of order <= 4.

    B = 2 s^T a_b - k_b - s^T Q s with a_b = I_Z^{-1} c_b, k_b = c_b^T I_Z^{-1} c_b,
    Q = V^{-1}; so B^2 = 4 (s^T a)^2 - 4 k s^T a - 4 (s^T a)(s^T Q s) + k^2
    + 2 k s^T Q s + (s^T Q s)^2, and every term averages to a contraction of
    the within-cell moment tensors M_b^{(1..4)} with a, k and Q.
    """
    d = mom["d"]
    total = sum(w for _, _, w in atoms)
    iz_inv, q = inv(mom["I_Z"]), inv(mom["V"])
    result = Fraction(0)
    idx = range(d)
    for b in range(n_bins):
        cell = [(s, w / total) for s, z, w in atoms if z == b]
        if not cell:
            continue
        a = matvec(iz_inv, mom["c"][b])
        k = quad(iz_inv, mom["c"][b], mom["c"][b])
        m0 = sum(w for _, w in cell)
        m1 = [sum(w * s[i] for s, w in cell) for i in idx]
        m2 = [[sum(w * s[i] * s[j] for s, w in cell) for j in idx] for i in idx]
        m3 = {
            (i, j, l): sum(w * s[i] * s[j] * s[l] for s, w in cell)
            for i in idx
            for j in idx
            for l in idx
        }
        m4 = {
            (i, j, l, o): sum(w * s[i] * s[j] * s[l] * s[o] for s, w in cell)
            for i in idx
            for j in idx
            for l in idx
            for o in idx
        }
        term_aa = sum(a[i] * a[j] * m2[i][j] for i in idx for j in idx)
        term_a = sum(a[i] * m1[i] for i in idx)
        term_aq = sum(a[i] * q[j][l] * m3[(i, j, l)] for i in idx for j in idx for l in idx)
        term_q = sum(q[i][j] * m2[i][j] for i in idx for j in idx)
        term_qq = sum(
            q[i][j] * q[l][o] * m4[(i, j, l, o)] for i in idx for j in idx for l in idx for o in idx
        )
        result += 4 * term_aa - 4 * k * term_a - 4 * term_aq + k * k * m0 + 2 * k * term_q + term_qq
    return result


def gateaux_check(
    atoms: list[Atom], n_bins: int, index: int, exponents: tuple[int, ...] = (10, 11, 12)
) -> dict[str, object]:
    """Exact difference quotients of r along the tilt towards one atom.

    theta_eps = (1 - eps) theta + eps delta_atom; the quotient (r(theta_eps) - r) / eps
    must approach r * B(atom) with an O(eps) remainder that halves per halving.
    """
    base = law_summary(atoms, n_bins)
    r0, target = base["ratio"], base["ratio"] * base["B"][index]
    total = sum(w for _, _, w in atoms)
    gaps = []
    for k in exponents:
        eps = Fraction(1, 2**k)
        tilted = [
            (s, z, (1 - eps) * w / total + (eps if i == index else 0))
            for i, (s, z, w) in enumerate(atoms)
        ]
        r_eps = det_ratio(law_moments(tilted, n_bins))
        gaps.append((r_eps - r0) / eps - target)
    ratios = [float(a / b) if b != 0 else None for a, b in zip(gaps[:-1], gaps[1:], strict=True)]
    return {
        "target": fr(target),
        "gaps": [fr(g) for g in gaps],
        "halving_ratios": ratios,
        "ok": all(g == 0 for g in gaps) or all(rt is not None and abs(rt - 2) < 0.25 for rt in ratios),
    }


def ellipsoid_identity(mom: dict[str, object], s: Vector, b: int) -> Fraction:
    """B(s, b) + [(s - V I^{-1} c)^T V^{-1} (s - V I^{-1} c) - c^T I^{-1} (V - I) I^{-1} c]; must be 0."""
    iz_inv, v_inv = inv(mom["I_Z"]), inv(mom["V"])
    c = mom["c"][b]
    centre = matvec(mom["V"], matvec(iz_inv, c))
    diff = [x - y for x, y in zip(s, centre, strict=True)]
    v_minus_i = madd(mom["V"], mom["I_Z"], Fraction(-1))
    rhs = quad(v_minus_i, matvec(iz_inv, c), matvec(iz_inv, c))
    return bracket(mom, s, b) + quad(v_inv, diff, diff) - rhs


class Lcg:
    """Tiny deterministic generator so the exact stage needs no numpy state."""

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF

    def next(self, bound: int) -> int:
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return self.state % bound


def random_law(rng: Lcg, d: int, n_bins: int, atoms_per_cell: int) -> list[Atom]:
    atoms: list[Atom] = []
    for b in range(n_bins):
        for _ in range(atoms_per_cell):
            s = [Fraction(rng.next(13) - 6, 1 + rng.next(3)) for _ in range(d)]
            atoms.append((s, b, Fraction(1 + rng.next(5))))
    total = sum(w for _, _, w in atoms)
    return [(s, z, w / total) for s, z, w in atoms]


def library_retention(atoms: list[Atom], n_bins: int, rank_rtol: float | None = None) -> float:
    import scorequant as sq

    report = sq.information_report(
        np.array([[float(x) for x in s] for s, _, _ in atoms]),
        np.array([z for _, z, _ in atoms]),
        np.array([float(w) for _, _, w in atoms]),
        n_bins=n_bins,
        rank_rtol=rank_rtol,
    )
    return float(report.geometric_mean_retention), int(report.effective_rank)


# --------------------------------------------------------------------------
# stage: exact


def _vec(*xs: int | Fraction) -> Vector:
    return [Fraction(x) for x in xs]


def adversarial_samples() -> dict[str, tuple[list[Atom], int]]:
    """Adversarial finite samples in d = 1, 2, 3 (uniform weights)."""
    out: dict[str, tuple[list[Atom], int]] = {}
    # d = 2: ties, a duplicate atom, a singleton cell (cell 3) and a declared-empty cell (cell 4)
    rows2 = [(2, 1), (-1, 3), (0, -2), (3, 3), (-2, -1), (1, 0), (1, 0), (-3, 2), (2, -3), (0, 1), (4, -1), (-1, -1), (2, 1)]
    labels2 = [0, 1, 2, 0, 1, 2, 2, 1, 3, 0, 0, 1, 2]
    out["d2_ties_duplicate_singleton_empty"] = (
        sample_atoms([_vec(*r) for r in rows2], labels2),
        5,
    )
    # d = 3: duplicate atoms across cells (same score, different labels) and unequal cells
    rows3 = [(1, 0, 2), (1, 0, 2), (-2, 1, 0), (0, -1, 3), (3, 2, -1), (-1, -1, -1), (2, 2, 2), (0, 3, -2), (-3, 0, 1), (1, -2, 0), (2, -1, 1), (0, 0, 1), (-1, 2, 2), (1, 1, -3)]
    labels3 = [0, 1, 1, 2, 0, 3, 2, 1, 3, 0, 2, 0, 1, 3]
    out["d3_duplicates_across_cells"] = (sample_atoms([_vec(*r) for r in rows3], labels3), 4)
    # d = 2, K = d = 2 on an exactly centred sample: I_hat_Z has rank <= 1, ratio exactly 0
    rows_c = [(1, 2), (-1, -2), (2, -1), (-2, 1), (3, 0), (-3, 0)]
    out["d2_K2_centred_rank_deficient"] = (sample_atoms([_vec(*r) for r in rows_c], [0, 1, 0, 1, 0, 1]), 2)
    # d = 3, K = 2 < d: rank I_hat_Z <= 2, ratio exactly 0
    out["d3_K2_below_dimension"] = (
        sample_atoms([_vec(*r) for r in rows3[:8]], [0, 1, 0, 1, 1, 0, 0, 1]),
        2,
    )
    # d = 2 sample with singular V_hat (all scores on a line): phi := 0 by convention
    rows_l = [(1, 2), (2, 4), (-1, -2), (3, 6), (-2, -4)]
    out["d2_singular_V_hat"] = (sample_atoms([_vec(*r) for r in rows_l], [0, 1, 0, 1, 2]), 3)
    # d = 1 with ties and an empty cell: the O6 reduction
    rows_1 = [(2,), (-1,), (0,), (3,), (-2,), (1,), (1,), (-3,), (2,), (0,)]
    out["d1_ties_empty"] = (sample_atoms([_vec(*r) for r in rows_1], [0, 1, 2, 0, 1, 2, 2, 1, 0, 3]), 5)
    return out


def within_scatter(atoms: list[Atom], mom: dict[str, object]) -> Matrix:
    d = mom["d"]
    total = sum(w for _, _, w in atoms)
    out = zeros(d)
    for s, z, w in atoms:
        diff = [x - y for x, y in zip(s, mom["c"][z], strict=True)]
        out = madd(out, outer(diff, diff), w / total)
    return out


def stage_exact() -> dict[str, object]:
    record: dict[str, object] = {"samples": {}, "laws": {}}
    for name, (atoms, n_bins) in adversarial_samples().items():
        mom = law_moments(atoms, n_bins)
        d = mom["d"]
        r = det_ratio(mom)
        entry: dict[str, object] = {
            "d": d,
            "K": n_bins,
            "n": len(atoms),
            "ratio": fr(r),
            "rank_I_hat_Z": rank(mom["I_Z"]),
            "det_V_hat": fr(det(mom["V"])),
        }
        scatter = within_scatter(atoms, mom)
        entry["within_scatter_identity"] = madd(mom["V"], mom["I_Z"], Fraction(-1)) == scatter
        entry["ratio_in_unit_interval"] = 0 <= r <= 1
        regular = det(mom["V"]) != 0 and det(mom["I_Z"]) != 0
        entry["regular"] = regular
        if regular:
            summary = law_summary(atoms, n_bins)
            entry["sum_psi_hat_zero"] = summary["E_B"] == 0
            entry["sigma2_hat_moment_expansion_exact"] = (
                sigma2_hat_from_moments(atoms, n_bins, mom) == summary["E_B2"]
            )
            entry["E_B2"] = fr(summary["E_B2"])
            eta = float(r) ** (1.0 / d)
            entry["sigma_hat"] = eta / d * math.sqrt(float(summary["E_B2"]))
            gate = [gateaux_check(atoms, n_bins, i) for i in range(len(atoms))]
            entry["gateaux_all_ok"] = all(g["ok"] for g in gate)
            entry["gateaux_first"] = gate[0]
            if d == 1:
                v = mom["V"][0][0]
                eta1 = r
                o6 = [
                    ((1 - eta1) * s[0] ** 2 - (s[0] - mom["c"][z][0]) ** 2) / v
                    for s, z, _ in atoms
                ]
                entry["d1_reduction_to_O6_exact"] = [eta1 * bv for bv in summary["B"]] == o6
            # affine invariance S -> A S
            a_mat = [[Fraction(2), Fraction(1)], [Fraction(-1), Fraction(3)]] if d == 2 else None
            if d == 3:
                a_mat = [[Fraction(1), Fraction(2), Fraction(0)], [Fraction(0), Fraction(1), Fraction(-1)], [Fraction(1), Fraction(0), Fraction(1)]]
            if a_mat is not None:
                moved = [(matvec(a_mat, s), z, w) for s, z, w in atoms]
                moved_summary = law_summary(moved, n_bins)
                entry["affine_invariance_exact"] = (
                    moved_summary["ratio"] == r and moved_summary["B"] == summary["B"]
                )
        lib, eff_rank = library_retention(atoms, n_bins)
        entry["library_geometric_mean_retention"] = lib
        entry["library_effective_rank"] = eff_rank
        entry["exact_plugin_eta"] = float(r) ** (1.0 / d)
        entry["library_gap"] = abs(lib - float(r) ** (1.0 / d))
        record["samples"][name] = entry
    # random rational atomic laws
    rng = Lcg(20260906)
    laws: list[dict[str, object]] = []
    all_ok = True
    for d, n_bins, per_cell in [(2, 3, 2), (2, 4, 3), (2, 5, 2), (3, 4, 2), (3, 5, 3), (3, 4, 3)] * 5:
        atoms = random_law(rng, d, n_bins, per_cell)
        mom = law_moments(atoms, n_bins)
        if det(mom["V"]) == 0 or det(mom["I_Z"]) == 0:
            continue
        summary = law_summary(atoms, n_bins)
        gate = [gateaux_check(atoms, n_bins, i, (8, 9, 10)) for i in range(0, len(atoms), max(1, len(atoms) // 4))]
        ell = all(ellipsoid_identity(mom, s, z) == 0 for s, z, _ in atoms)
        # and at generic rational points off the support, every cell
        probe = [_vec(Fraction(k, 3), Fraction(1 - k, 2), *([Fraction(k, 5)] if d == 3 else [])) for k in range(-3, 4)]
        ell_generic = all(ellipsoid_identity(mom, s, b) == 0 for s in probe for b in range(n_bins))
        ok = (
            summary["E_B"] == 0
            and summary["E_N1"] == d
            and summary["E_N2"] == d
            and summary["covariance_form"] == summary["E_B2"]
            and sigma2_hat_from_moments(atoms, n_bins, mom) == summary["E_B2"]
            and all(g["ok"] for g in gate)
            and ell
            and ell_generic
        )
        all_ok &= ok
        laws.append(
            {
                "d": d,
                "K": n_bins,
                "atoms": len(atoms),
                "ratio": fr(summary["ratio"]),
                "E_B_zero": summary["E_B"] == 0,
                "E_N1": fr(summary["E_N1"]),
                "E_N2": fr(summary["E_N2"]),
                "covariance_form_matches": summary["covariance_form"] == summary["E_B2"],
                "moment_expansion_matches": sigma2_hat_from_moments(atoms, n_bins, mom) == summary["E_B2"],
                "gateaux_ok": all(g["ok"] for g in gate),
                "ellipsoid_identity_on_support": ell,
                "ellipsoid_identity_generic_points": ell_generic,
                "ok": ok,
            }
        )
    record["laws"] = {"count": len(laws), "all_ok": all_ok, "entries": laws}
    record["provenance"] = provenance("exact", {"laws": len(laws), "lcg_seed": 20260906})
    return record


# --------------------------------------------------------------------------
# stage: degenerate


def fixture_two_atom_law() -> tuple[list[Atom], int]:
    """CE-O7-ELLIPSOID-ZERO-VARIANCE-001 rebuilt from its description."""
    base = [(3, 4), (3, -4)]
    atoms: list[Atom] = []
    for b in range(4):
        for x, y in base:
            for _ in range(b):
                x, y = -y, x  # quarter turn
            atoms.append((_vec(x, y), b, Fraction(1, 8)))
    return atoms, 4


def sphere_law_d3() -> tuple[list[Atom], int]:
    """d = 3, K = 6 (cells +-e_i), two atoms per cell on the sphere |s|^2 = 25."""
    atoms: list[Atom] = []
    cells = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]
    for b, axis in enumerate(cells):
        i = axis.index(next(x for x in axis if x))
        sign = 1 if sum(axis) > 0 else -1
        j = (i + 1) % 3
        for t in (4, -4):
            s = [Fraction(0)] * 3
            s[i] = Fraction(3 * sign)
            s[j] = Fraction(t)
            atoms.append((s, b, Fraction(1, 12)))
    return atoms, 6


def unit_retention_law() -> tuple[list[Atom], int]:
    """eta_D = 1: S = c_Z a.s., c_0 = (1, 0), c_1 = (0, 1), p = 1/2."""
    return [(_vec(1, 0), 0, Fraction(1, 2)), (_vec(0, 1), 1, Fraction(1, 2))], 2


def compositions(n: int, parts: int):
    for cuts in itertools.combinations(range(n + parts - 1), parts - 1):
        prev, out = -1, []
        for c in cuts:
            out.append(c - prev - 1)
            prev = c
        out.append(n + parts - 2 - prev)
        yield tuple(out)


def arc_law_numeric(alpha: float, nodes: int = 4000) -> dict[str, object]:
    """Uniform arc law of the fixture: V, I_Z, eta_D and sigma^2 by Gauss-Legendre in the angle."""
    x, w = np.polynomial.legendre.leggauss(nodes)
    theta = math.pi + alpha * x
    w = w / 2.0  # uniform density on [pi - alpha, pi + alpha]
    centre, radius = np.array([25.0 / 3.0, 0.0]), 20.0 / 3.0
    s0 = centre[None, :] + radius * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    rot = np.array([[0.0, -1.0], [1.0, 0.0]])
    cells = [s0]
    for _ in range(3):
        cells.append(cells[-1] @ rot.T)
    p = np.full(4, 0.25)
    m = np.stack([0.25 * (w[:, None] * s).sum(axis=0) for s in cells])
    v = sum(0.25 * (s.T * w) @ s for s in cells)
    i_z = sum(np.outer(m[b], m[b]) / p[b] for b in range(4))
    ratio = np.linalg.det(i_z) / np.linalg.det(v)
    iz_inv, v_inv = np.linalg.inv(i_z), np.linalg.inv(v)
    b_max, e_b2 = 0.0, 0.0
    for b, s in enumerate(cells):
        c = m[b] / p[b]
        bv = 2 * s @ iz_inv @ c - c @ iz_inv @ c - np.einsum("ni,ij,nj->n", s, v_inv, s)
        b_max = max(b_max, float(np.abs(bv).max()))
        e_b2 += 0.25 * float(w @ bv**2)
    return {
        "alpha": alpha,
        "sin_alpha_over_alpha": math.sin(alpha) / alpha,
        "cell_mean_0": m[0].tolist(),
        "second_moment_cell_0": float(w @ np.einsum("ni,ni->n", s0, s0)),
        "V": v.tolist(),
        "I_Z": i_z.tolist(),
        "det_ratio": float(ratio),
        "eta_D": float(math.sqrt(ratio)),
        "max_abs_B_on_support": b_max,
        "sigma2": (ratio / 4.0) * e_b2,
    }


def stage_degenerate() -> dict[str, object]:
    record: dict[str, object] = {}
    # (b) the fixture's two-atom law
    atoms, n_bins = fixture_two_atom_law()
    summary = law_summary(atoms, n_bins)
    mom = summary["moments"]
    record["fixture_two_atom_law"] = {
        "V": frm(mom["V"]),
        "I_Z": frm(mom["I_Z"]),
        "det_ratio": fr(summary["ratio"]),
        "E_S": [fr(sum(w * s[i] for s, _, w in atoms)) for i in range(2)],
        "B_at_atoms": [fr(x) for x in summary["B"]],
        "sigma2": fr(summary["E_B2"]),
        "ellipsoid_identity_on_support": all(ellipsoid_identity(mom, s, z) == 0 for s, z, _ in atoms),
        "cell0_centre": [fr(x) for x in matvec(mom["V"], matvec(inv(mom["I_Z"]), mom["c"][0]))],
        "cell0_rhs": fr(
            quad(
                madd(mom["V"], mom["I_Z"], Fraction(-1)),
                matvec(inv(mom["I_Z"]), mom["c"][0]),
                matvec(inv(mom["I_Z"]), mom["c"][0]),
            )
        ),
        "library_geometric_mean_retention": library_retention(atoms, n_bins)[0],
    }
    # the atomless arc law: alpha solves sin(alpha)/alpha = 4/5
    lo, hi = 0.5, 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if math.sin(mid) / mid > 0.8:
            lo = mid
        else:
            hi = mid
    alpha = 0.5 * (lo + hi)
    record["arc_law"] = arc_law_numeric(alpha)
    # (c) a d = 3 sphere law
    atoms3, k3 = sphere_law_d3()
    s3 = law_summary(atoms3, k3)
    m3 = s3["moments"]
    record["sphere_law_d3"] = {
        "K": k3,
        "V": frm(m3["V"]),
        "I_Z": frm(m3["I_Z"]),
        "det_ratio": fr(s3["ratio"]),
        "eta_D": fr(Fraction(9, 25)) if s3["ratio"] == Fraction(9, 25) ** 3 else float(s3["ratio"]) ** (1 / 3),
        "E_S": [fr(sum(w * s[i] for s, _, w in atoms3)) for i in range(3)],
        "B_at_atoms": [fr(x) for x in s3["B"]],
        "sigma2": fr(s3["E_B2"]),
        "cell0_ellipsoid_centre": [fr(x) for x in matvec(m3["V"], matvec(inv(m3["I_Z"]), m3["c"][0]))],
        "circle_s1_eq_3_on_ellipsoid": all(
            ellipsoid_identity(m3, _vec(3, x, y), 0) == 0 and bracket(m3, _vec(3, x, y), 0) == 0
            for x, y in [(4, 0), (0, 4), (Fraction(12, 5), Fraction(16, 5)), (Fraction(-12, 5), Fraction(16, 5))]
        ),
    }
    # (d) the eta_D = 1 law and its singular samples (H1)
    atoms1, k1 = unit_retention_law()
    su = law_summary(atoms1, k1)
    h1: dict[str, object] = {
        "population": {
            "V": frm(su["moments"]["V"]),
            "I_Z": frm(su["moments"]["I_Z"]),
            "det_ratio": fr(su["ratio"]),
            "B_at_atoms": [fr(x) for x in su["B"]],
        },
        "compositions": [],
    }
    for n in range(1, 7):
        for n0 in range(n + 1):
            scores = [_vec(1, 0)] * n0 + [_vec(0, 1)] * (n - n0)
            labels = [0] * n0 + [1] * (n - n0)
            sample = sample_atoms(scores, labels)
            mom_s = law_moments(sample, k1)
            r_s = det_ratio(mom_s)
            lib, eff = library_retention(sample, k1)
            h1["compositions"].append(
                {
                    "n": n,
                    "counts": [n0, n - n0],
                    "det_V_hat": fr(det(mom_s["V"])),
                    "exact_plugin_ratio": fr(r_s),
                    "library": lib,
                    "library_effective_rank": eff,
                }
            )
    failures = [c for c in h1["compositions"] if c["exact_plugin_ratio"] == "0"]
    h1["singular_sample_count"] = len(failures)
    h1["all_regular_samples_give_one"] = all(
        c["exact_plugin_ratio"] == "1" for c in h1["compositions"] if c["det_V_hat"] != "0"
    )
    h1["library_returns_one_on_singular_samples"] = all(c["library"] == 1.0 for c in failures)
    record["unit_retention_law_H1"] = h1
    # (e) exhaustive compositions on the two-atom fixture law
    r_pop = summary["ratio"]
    enum: dict[str, object] = {"per_n": [], "first_below": None}
    below_total = 0
    for n in range(1, 7):
        count = regular = below = 0
        for comp in compositions(n, len(atoms)):
            scores, labels = [], []
            for (s, z, _), k in zip(atoms, comp, strict=True):
                scores.extend([s] * k)
                labels.extend([z] * k)
            sample = sample_atoms(scores, labels)
            mom_s = law_moments(sample, n_bins)
            count += 1
            if det(mom_s["V"]) == 0 or det(mom_s["I_Z"]) == 0:
                continue
            regular += 1
            r_s = det_ratio(mom_s)
            if r_s < r_pop:
                below += 1
                if enum["first_below"] is None:
                    enum["first_below"] = {"n": n, "composition": list(comp), "ratio": fr(r_s)}
        below_total += below
        enum["per_n"].append({"n": n, "compositions": count, "regular": regular, "below_population": below})
    enum["plugin_can_fall_below_eta_D"] = below_total > 0
    record["two_atom_law_enumeration"] = enum
    record["provenance"] = provenance("degenerate", {"max_n": 6})
    return record


# --------------------------------------------------------------------------
# stage: popref — population references by two independent quadrature routes
#
# Every law is one-dimensional in x with a density f on the line, an interval
# partition of x (the frozen rule) and a vector score S(x). Cell moments are
# integrals of polynomial-in-S functions against f over the cell interval.

SQRT2PI = math.sqrt(2.0 * math.pi)


def phi(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / SQRT2PI


def hermite_scores(x: np.ndarray, d: int) -> np.ndarray:
    cols = [x, x * x - 1.0, x**3 - 3.0 * x]
    return np.stack(cols[:d], axis=-1)


class IntervalLaw:
    """x ~ f, cells = intervals between cuts, scores S(x) in R^d."""

    def __init__(self, name: str, d: int, cuts: list[float], density, scores, support: tuple[float, float]) -> None:
        self.name, self.d, self.cuts, self.density, self.scores, self.support = name, d, list(cuts), density, scores, support

    @property
    def n_bins(self) -> int:
        return len(self.cuts) + 1

    def cell_edges(self) -> list[tuple[float, float]]:
        edges = [self.support[0], *self.cuts, self.support[1]]
        return [(edges[i], edges[i + 1]) for i in range(self.n_bins)]

    def labels(self, x: np.ndarray) -> np.ndarray:
        return np.searchsorted(np.asarray(self.cuts), x, side="right")


def gauss_legendre_nodes(lo: float, hi: float, pieces: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(lo, hi, pieces + 1)
    half = 0.5 * (edges[1:] - edges[:-1])
    mid = 0.5 * (edges[1:] + edges[:-1])
    nodes = (mid[:, None] + half[:, None] * x[None, :]).ravel()
    weights = (half[:, None] * w[None, :]).ravel()
    return nodes, weights


def cell_moments_gl(law: IntervalLaw, pieces: int, order: int) -> dict[str, np.ndarray]:
    """Route 2: composite Gauss-Legendre with pieces aligned to the cuts."""
    d = law.d
    p = np.zeros(law.n_bins)
    m = np.zeros((law.n_bins, d))
    m2 = np.zeros((law.n_bins, d, d))
    fourth = 0.0
    nodes_all: list[tuple[int, np.ndarray, np.ndarray]] = []
    for b, (lo, hi) in enumerate(law.cell_edges()):
        n_pieces = max(1, int(round(pieces * (hi - lo) / (law.support[1] - law.support[0]))))
        xs, ws = gauss_legendre_nodes(lo, hi, n_pieces, order)
        fw = ws * law.density(xs)
        s = law.scores(xs)
        p[b] = fw.sum()
        m[b] = fw @ s
        m2[b] = (s.T * fw) @ s
        fourth += fw @ (np.einsum("ni,ni->n", s, s) ** 2)
        nodes_all.append((b, xs, fw))
    return {"p": p, "m": m, "M2": m2, "fourth": fourth, "nodes": nodes_all}


def cell_moments_quadpack(law: IntervalLaw) -> dict[str, np.ndarray]:
    """Route 1: adaptive QUADPACK per cell and per moment."""
    from scipy.integrate import quad as quadpack

    d = law.d
    p = np.zeros(law.n_bins)
    m = np.zeros((law.n_bins, d))
    m2 = np.zeros((law.n_bins, d, d))
    fourth = 0.0
    opts = {"epsabs": 1e-14, "epsrel": 1e-13, "limit": 500}

    def integrate(fn, lo, hi):
        return quadpack(fn, lo, hi, **opts)[0]

    for b, (lo, hi) in enumerate(law.cell_edges()):
        dens = lambda x: float(law.density(np.array([x]))[0])  # noqa: E731
        sc = lambda x: law.scores(np.array([x]))[0]  # noqa: E731
        p[b] = integrate(dens, lo, hi)
        for i in range(d):
            m[b, i] = integrate(lambda x: sc(x)[i] * dens(x), lo, hi)
            for j in range(i, d):
                m2[b, i, j] = m2[b, j, i] = integrate(lambda x: sc(x)[i] * sc(x)[j] * dens(x), lo, hi)
        fourth += integrate(lambda x: float(sc(x) @ sc(x)) ** 2 * dens(x), lo, hi)
    return {"p": p, "m": m, "M2": m2, "fourth": fourth}


def population_from_moments(law: IntervalLaw, mom: dict[str, np.ndarray], gl: dict[str, np.ndarray]) -> dict[str, object]:
    """eta_D, sigma^2 and the endpoint geometry from cell moments.

    sigma^2 = (eta_D/d)^2 E[B^2] is integrated on the Gauss-Legendre nodes of
    ``gl`` (B is a polynomial in S, so the nodes resolve it as well as the
    moments); the moments themselves come from ``mom`` (either route).
    """
    d = law.d
    p, m, m2 = mom["p"], mom["m"], mom["M2"]
    v = m2.sum(axis=0)
    i_z = sum(np.outer(m[b], m[b]) / p[b] for b in range(law.n_bins))
    eig = np.linalg.eigvalsh(i_z)
    rank_iz = int((eig > 1e-10 * eig.max()).sum())
    out: dict[str, object] = {
        "d": d,
        "K": law.n_bins,
        "p": p.tolist(),
        "m": m.tolist(),
        "E_S": m.sum(axis=0).tolist(),
        "V": v.tolist(),
        "I_Z": i_z.tolist(),
        "I_Z_eigenvalues": eig.tolist(),
        "rank_I_Z": rank_iz,
        "fourth_moment": float(mom["fourth"]),
    }
    if rank_iz == d:
        ratio = np.linalg.det(i_z) / np.linalg.det(v)
        eta = ratio ** (1.0 / d)
        iz_inv, v_inv = np.linalg.inv(i_z), np.linalg.inv(v)
        e_b, e_b2, e_n1 = 0.0, 0.0, 0.0
        for b, xs, fw in gl["nodes"]:
            s = law.scores(xs)
            c = m[b] / p[b]
            n1 = 2 * s @ iz_inv @ c - c @ iz_inv @ c
            bv = n1 - np.einsum("ni,ij,nj->n", s, v_inv, s)
            e_b += fw @ bv
            e_b2 += fw @ bv**2
            e_n1 += fw @ n1
        out.update(
            {
                "det_ratio": float(ratio),
                "eta_D": float(eta),
                "sigma2": float((eta / d) ** 2 * e_b2),
                "sigma": float(eta / d * math.sqrt(e_b2)),
                "E_B": float(e_b),
                "E_N1": float(e_n1),
            }
        )
    else:
        # endpoint geometry for O7.4(b): range R, null U, A, a_b, Lambda, null covariances
        w, q = np.linalg.eigh(i_z)
        u, r_basis = q[:, : d - rank_iz], q[:, d - rank_iz :]
        a_mat = r_basis.T @ i_z @ r_basis
        a_vec = m @ r_basis  # rows a_b^T
        lam = np.array(
            [[a_vec[bp] @ np.linalg.solve(a_mat, a_vec[b]) / p[bp] for b in range(law.n_bins)] for bp in range(law.n_bins)]
        )
        null_cov = [u.T @ m2[b] @ u for b in range(law.n_bins)]
        out.update(
            {
                "det_ratio": 0.0,
                "eta_D": 0.0,
                "null_basis_U": u.tolist(),
                "range_basis_R": r_basis.tolist(),
                "A": a_mat.tolist(),
                "det_A": float(np.linalg.det(a_mat)),
                "det_V": float(np.linalg.det(v)),
                "Lambda": lam.tolist(),
                "Lambda_is_projection": bool(np.allclose(lam @ lam, lam, atol=1e-12)),
                "Lambda_trace": float(np.trace(lam)),
                "null_covariances": [nc.tolist() for nc in null_cov],
                "U_m_b_max": float(np.abs(m @ u).max()),
            }
        )
    return out


def hermite_laws() -> dict[str, IntervalLaw]:
    sup = (-12.0, 12.0)
    return {
        "hermite2": IntervalLaw("hermite2", 2, [-1.0, 0.0, 1.0], phi, lambda x: hermite_scores(x, 2), sup),
        "hermite3": IntervalLaw("hermite3", 3, [-1.5, -0.5, 0.5, 1.5], phi, lambda x: hermite_scores(x, 3), sup),
        "endpoint2": IntervalLaw("endpoint2", 2, [0.0], phi, lambda x: hermite_scores(x, 2), sup),
        "endpoint3": IntervalLaw("endpoint3", 3, [-0.6, 0.6], phi, lambda x: hermite_scores(x, 3), sup),
    }


EXTRA_LAWS_PATH = ARTIFACTS / "extra_laws.json"


def mixture_law(params: dict[str, object]) -> IntervalLaw:
    """Bounded scores: mixture fractions as parameters of a location mixture.

    x ~ sum_k pi_k N(mu_k, sd_k^2); theta = (pi_1, pi_2) with pi_3 = 1 - pi_1 - pi_2,
    so the score is s_k(x) = (f_k(x) - f_3(x)) / f(x), bounded by 1/pi_k.
    """
    mus = np.asarray(params["means"], dtype=float)
    sds = np.asarray(params["sds"], dtype=float)
    fracs = np.asarray(params["fractions"], dtype=float)
    fr_all = np.append(fracs, 1.0 - fracs.sum())

    def comps(x: np.ndarray) -> np.ndarray:
        return np.stack([phi((x - mu) / sd) / sd for mu, sd in zip(mus, sds, strict=True)], axis=-1)

    def density(x: np.ndarray) -> np.ndarray:
        return comps(x) @ fr_all

    def scores(x: np.ndarray) -> np.ndarray:
        f = comps(x)
        tot = f @ fr_all
        return (f[:, :2] - f[:, 2:3]) / tot[:, None]

    return IntervalLaw("mixture", 2, [float(c) for c in params["cuts"]], density, scores, tuple(params["support"]))


def library_rule_law(params: dict[str, object]) -> tuple[IntervalLaw, dict[str, object]]:
    """Rebuild the library-fitted D-exchange rule on the d = 2 Hermite score and locate its x-cuts."""
    import scorequant as sq

    rng = np.random.default_rng(int(params["training_seed"]))
    x_train = rng.standard_normal(int(params["n_train"]))
    result = sq.optimize_partition(
        hermite_scores(x_train, 2),
        n_bins=int(params["K"]),
        criterion=sq.DOptimality(),
        config=sq.DExchangeConfig(seed=int(params["solver_seed"]), initializer_restarts=int(params["initializer_restarts"])),
    )
    rule = result.compile_quantizer()
    grid = np.linspace(-8.0, 8.0, 160_001)
    lab = rule.predict_scores(hermite_scores(grid, 2))
    change = np.flatnonzero(lab[1:] != lab[:-1])
    cuts = []
    for k in change:
        lo, hi = grid[k], grid[k + 1]
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if rule.predict_scores(hermite_scores(np.array([mid]), 2))[0] == lab[k]:
                lo = mid
            else:
                hi = mid
        cuts.append(0.5 * (lo + hi))
    labels_by_piece = [int(lab[0]), *[int(lab[k + 1]) for k in change]]
    law = IntervalLaw("library", 2, cuts, phi, lambda x: hermite_scores(x, 2), (-12.0, 12.0))
    info = {
        "cuts": cuts,
        "recorded_cuts_gap": float(np.abs(np.asarray(cuts) - np.asarray(params["recorded_cuts_in_x"])).max()) if len(cuts) == len(params["recorded_cuts_in_x"]) else None,
        "piece_labels": labels_by_piece,
        "pieces_equal_cells": len(set(labels_by_piece)) == len(labels_by_piece),
        "train_retention": float(result.train_report.geometric_mean_retention),
    }
    # the rule's own labels on the nodes must agree with the interval labelling
    xs, _ = gauss_legendre_nodes(-8.0, 8.0, 400, 24)
    info["label_disagreements_on_9600_nodes"] = int((rule.predict_scores(hermite_scores(xs, 2)) != np.array(labels_by_piece)[law.labels(xs)]).sum())
    return law, info


def all_laws() -> tuple[dict[str, IntervalLaw], dict[str, object]]:
    laws = hermite_laws()
    extra: dict[str, object] = {}
    if EXTRA_LAWS_PATH.is_file():
        params = json.loads(EXTRA_LAWS_PATH.read_text())
        laws["mixture"] = mixture_law(params["mixture"])
        law, info = library_rule_law(params["library"])
        laws["library"] = law
        extra = {"parameters": params, "library_rule": info}
    return laws, extra


def stage_popref() -> dict[str, object]:
    laws, extra = all_laws()
    record: dict[str, object] = {"laws": {}, "extra": extra}
    for name, law in laws.items():
        gl = cell_moments_gl(law, pieces=600, order=32)
        gl_fine = cell_moments_gl(law, pieces=900, order=40)
        qp = cell_moments_quadpack(law)
        pop_gl = population_from_moments(law, gl, gl)
        pop_fine = population_from_moments(law, gl_fine, gl_fine)
        pop_qp = population_from_moments(law, qp, gl_fine)
        entry = dict(pop_qp)
        entry["route_disagreement"] = {
            "eta_D_gl_vs_quadpack": abs(pop_gl["eta_D"] - pop_qp["eta_D"]),
            "eta_D_gl_vs_gl_fine": abs(pop_gl["eta_D"] - pop_fine["eta_D"]),
            "sigma_gl_vs_quadpack": abs(pop_gl.get("sigma", 0.0) - pop_qp.get("sigma", 0.0)),
            "p_max": float(np.abs(np.asarray(pop_gl["p"]) - np.asarray(pop_qp["p"])).max()),
            "V_max": float(np.abs(np.asarray(pop_gl["V"]) - np.asarray(pop_qp["V"])).max()),
            "fourth_moment": abs(pop_gl["fourth_moment"] - pop_qp["fourth_moment"]),
        }
        if name in RECORDED:
            entry["recorded"] = RECORDED[name]
            entry["recorded_gap"] = {
                "eta_D": abs(entry["eta_D"] - RECORDED[name]["eta"]),
                "sigma": abs(entry.get("sigma", 0.0) - RECORDED[name]["sigma"]),
            }
        record["laws"][name] = entry
        print(name, "eta_D", entry["eta_D"], "sigma", entry.get("sigma"), "rank", entry["rank_I_Z"])
    record["provenance"] = provenance("popref", {"gl": "600x32 and 900x40 cut-aligned", "quadpack": "epsabs 1e-14"})
    return record


# --------------------------------------------------------------------------
# stage: coverage — fresh-seed Monte Carlo of the Wald interval


def plugin_batch(s: np.ndarray, z: np.ndarray, n_bins: int) -> dict[str, np.ndarray]:
    """Vectorised plug-in on a batch: s [R, n, d], z [R, n] -> eta_hat, sigma_hat, empties."""
    big_r, n, d = s.shape
    onehot = np.zeros((big_r, n, n_bins))
    np.put_along_axis(onehot, z[..., None], 1.0, axis=2)
    p_hat = onehot.mean(axis=1)  # [R, K]
    m_hat = np.einsum("rnb,rni->rbi", onehot, s) / n  # [R, K, d]
    v_hat = np.einsum("rni,rnj->rij", s, s) / n
    safe_p = np.where(p_hat > 0, p_hat, 1.0)
    c_hat = m_hat / safe_p[..., None]
    i_hat = np.einsum("rb,rbi,rbj->rij", p_hat, c_hat, c_hat)
    det_v = np.linalg.det(v_hat)
    det_i = np.linalg.det(i_hat)
    ratio = np.where(det_v > 0, det_i / np.where(det_v > 0, det_v, 1.0), 0.0)
    eta = np.where(ratio > 0, np.abs(ratio) ** (1.0 / d), 0.0)
    regular = (ratio > 0) & (np.linalg.matrix_rank(i_hat) == d)
    i_inv = np.linalg.inv(np.where(regular[:, None, None], i_hat, np.eye(d)[None]))
    v_inv = np.linalg.inv(np.where(det_v[:, None, None] > 0, v_hat, np.eye(d)[None]))
    a_hat = np.einsum("rij,rbj->rbi", i_inv, c_hat)  # I^{-1} c_b
    k_hat = np.einsum("rbi,rbi->rb", c_hat, a_hat)
    a_z = np.take_along_axis(a_hat, z[..., None], axis=1)  # [R, n, d]
    k_z = np.take_along_axis(k_hat, z, axis=1)  # [R, n]
    bracket_i = 2 * np.einsum("rni,rni->rn", s, a_z) - k_z - np.einsum("rni,rij,rnj->rn", s, v_inv, s)
    sigma = eta / d * np.sqrt((bracket_i**2).mean(axis=1))
    return {
        "eta": eta,
        "sigma": sigma,
        "empty_cells": (p_hat == 0).sum(axis=1),
        "regular": regular,
        "psi_sum": (eta / d)[:, None] * bracket_i.sum(axis=1)[:, None],
    }


def sampler_for(name: str, laws: dict[str, IntervalLaw], popref: dict[str, object], extra: dict[str, object]):
    """Return (draw(rng, n) -> (S [n, d], Z [n]), eta_D, sigma, d, K)."""
    if name == "arc":
        alpha = popref["arc_alpha"]

        def draw_arc(rng: np.random.Generator, n: int):
            b = rng.integers(0, 4, size=n)
            theta = math.pi + alpha * (2 * rng.random(n) - 1)
            s0 = np.stack([25.0 / 3.0 + 20.0 / 3.0 * np.cos(theta), 20.0 / 3.0 * np.sin(theta)], axis=1)
            ang = b * (math.pi / 2)
            rot = np.stack([np.cos(ang) * s0[:, 0] - np.sin(ang) * s0[:, 1], np.sin(ang) * s0[:, 0] + np.cos(ang) * s0[:, 1]], axis=1)
            return rot, b

        return draw_arc, 9.0 / 25.0, 0.0, 2, 4
    law = laws[name]
    ref = popref["laws"][name]
    piece_labels = None
    if name == "library":
        piece_labels = np.asarray(extra["library_rule"]["piece_labels"])

    if name == "mixture":
        params = extra["parameters"]["mixture"]
        mus, sds = np.asarray(params["means"]), np.asarray(params["sds"])
        th = np.append(np.asarray(params["fractions"]), 1.0 - sum(params["fractions"]))

        def draw_x(rng: np.random.Generator, n: int) -> np.ndarray:
            k = rng.choice(3, size=n, p=th)
            return rng.normal(mus[k], sds[k])
    else:

        def draw_x(rng: np.random.Generator, n: int) -> np.ndarray:
            return rng.standard_normal(n)

    def draw(rng: np.random.Generator, n: int):
        x = draw_x(rng, n)
        z = law.labels(x)
        if piece_labels is not None:
            z = piece_labels[z]
        return law.scores(x), z

    return draw, ref["eta_D"], ref.get("sigma", 0.0), law.d, law.n_bins


def skewness(values: np.ndarray) -> float:
    centred = values - values.mean()
    return float((centred**3).mean() / (centred**2).mean() ** 1.5)


def replicate(draw, eta_pop: float, sigma_pop: float, d: int, n_bins: int, n: int, replicates: int, seed: list[int], chunk: int = 250) -> dict[str, object]:
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    etas, sigmas, empties, regs = [], [], [], []
    for start in range(0, replicates, chunk):
        size = min(chunk, replicates - start)
        s_all, z_all = [], []
        for _ in range(size):
            s, z = draw(rng, n)
            s_all.append(s)
            z_all.append(z)
        out = plugin_batch(np.stack(s_all), np.stack(z_all), n_bins)
        etas.append(out["eta"])
        sigmas.append(out["sigma"])
        empties.append(out["empty_cells"])
        regs.append(out["regular"])
    eta_hat, sigma_hat = np.concatenate(etas), np.concatenate(sigmas)
    empty, regular = np.concatenate(empties), np.concatenate(regs)
    half = Z975 * sigma_hat / math.sqrt(n)
    lo, hi = eta_hat - half, eta_hat + half
    covers = (lo <= eta_pop) & (eta_pop <= hi)
    cov = float(covers.mean())
    stud = (eta_hat - eta_pop) / np.where(sigma_hat > 0, sigma_hat, np.nan) * math.sqrt(n)
    rec: dict[str, object] = {
        "n": n,
        "replicates": replicates,
        "coverage": cov,
        "coverage_se": math.sqrt(cov * (1 - cov) / replicates),
        "mean_eta_hat": float(eta_hat.mean()),
        "n_bias": float(n * (eta_hat.mean() - eta_pop)),
        "sd_eta_hat": float(eta_hat.std(ddof=1)),
        "sigma_over_sqrt_n": sigma_pop / math.sqrt(n),
        "mean_sigma_hat_over_sqrt_n": float(sigma_hat.mean() / math.sqrt(n)),
        "median_half_width": float(np.median(half)),
        "plugin_studentized_skew": skewness(stud[np.isfinite(stud)]) if np.isfinite(stud).any() else None,
        "plugin_studentized_sd": float(np.nanstd(stud, ddof=1)),
        "intervals_below_zero": int((lo < 0).sum()),
        "intervals_above_one": int((hi > 1).sum()),
        "replicates_with_empty_cell": int((empty > 0).sum()),
        "irregular_replicates": int((~regular).sum()),
        "eta_hat_ge_population_fraction": float((eta_hat >= eta_pop).mean()),
        "n_sd_eta_hat": float(n * eta_hat.std(ddof=1)),
    }
    if sigma_pop > 0:
        oracle = (eta_hat - eta_pop) / sigma_pop * math.sqrt(n)
        rec["relative_rmse_sigma_hat"] = float(np.sqrt(((sigma_hat / sigma_pop - 1) ** 2).mean()))
        rec["mean_sigma_hat_over_sigma"] = float((sigma_hat / sigma_pop).mean())
        rec["oracle_studentized_sd"] = float(oracle.std(ddof=1))
        rec["oracle_studentized_skew"] = skewness(oracle)
    return rec


def load_popref() -> tuple[dict[str, object], dict[str, IntervalLaw], dict[str, object]]:
    popref = json.loads((ARTIFACTS / "popref.json").read_text())
    degenerate = json.loads((ARTIFACTS / "degenerate.json").read_text())
    popref["arc_alpha"] = degenerate["arc_law"]["alpha"]
    laws, extra = all_laws()
    return popref, laws, extra


COVERAGE_LAWS = ["mixture", "hermite2", "library", "hermite3", "arc"]
LAW_INDEX = {"hermite2": 1, "hermite3": 2, "mixture": 3, "library": 4, "endpoint2": 5, "endpoint3": 6, "arc": 7}
SIZES = (100, 300, 1000, 3000)


def stage_coverage(replicates: int = 4000) -> dict[str, object]:
    popref, laws, extra = load_popref()
    record: dict[str, object] = {"laws": {}}
    for name in COVERAGE_LAWS:
        draw, eta_pop, sigma_pop, d, n_bins = sampler_for(name, laws, popref, extra)
        rows = []
        for n in SIZES:
            rec = replicate(draw, eta_pop, sigma_pop, d, n_bins, n, replicates, [20260906, 1001, LAW_INDEX[name], n])
            rows.append(rec)
            print(name, n, "coverage", round(rec["coverage"], 4), "sd", round(rec["sd_eta_hat"], 5), "sigma/sqrt n", round(rec["sigma_over_sqrt_n"], 5), "sigma_hat ratio", rec.get("mean_sigma_hat_over_sigma"))
        record["laws"][name] = {"eta_D": eta_pop, "sigma": sigma_pop, "d": d, "K": n_bins, "rows": rows}
    record["provenance"] = provenance("coverage", {"replicates": replicates, "seed": "SeedSequence([20260906, 1001, law, n])", "sizes": list(SIZES)})
    return record


# --------------------------------------------------------------------------
# stage: singular — the O7.4(b) limit law against finite n


def limit_law_draws(ref: dict[str, object], draws: int, rng: np.random.Generator) -> np.ndarray:
    """(det A det(W P^{-1} W^T) / det V)^{1/d} with W = G (I - Lambda), G_b ~ N(0, N_b) independent."""
    d, n_bins = ref["d"], ref["K"]
    p = np.asarray(ref["p"])
    lam = np.asarray(ref["Lambda"])
    null_cov = [np.asarray(nc) for nc in ref["null_covariances"]]
    dr = null_cov[0].shape[0]
    g = np.zeros((draws, dr, n_bins))
    for b in range(n_bins):
        chol = np.linalg.cholesky(null_cov[b])
        g[:, :, b] = rng.standard_normal((draws, dr)) @ chol.T
    w = g @ (np.eye(n_bins) - lam)
    gram = np.einsum("mib,b,mjb->mij", w, 1.0 / p, w)
    det_gram = np.linalg.det(gram) if dr > 1 else gram[:, 0, 0]
    return (ref["det_A"] * det_gram / ref["det_V"]) ** (1.0 / d)


def stage_singular(replicates: int = 4000) -> dict[str, object]:
    popref, laws, extra = load_popref()
    record: dict[str, object] = {"laws": {}}
    for name in ("endpoint2", "endpoint3"):
        ref = popref["laws"][name]
        d, r = ref["d"], ref["rank_I_Z"]
        exponent = (d - r) / d
        rng = np.random.default_rng(np.random.SeedSequence([20260906, 1002, LAW_INDEX[name]]))
        limit = limit_law_draws(ref, 1_000_000, rng)
        entry: dict[str, object] = {
            "d": d,
            "rank": r,
            "exponent": exponent,
            "limit_law": {
                "mean": float(limit.mean()),
                "sd": float(limit.std()),
                "quantiles_10_50_90": [float(np.quantile(limit, q)) for q in (0.1, 0.5, 0.9)],
            },
            "rows": [],
        }
        if name == "endpoint2":
            entry["limit_law"]["closed_form_mean"] = 2.0 / math.pi  # E|N(0,2)| / sqrt(pi)
        draw, _, _, _, n_bins = sampler_for(name, laws, popref, extra)
        for n in (300, 1000, 3000, 10000):
            rec = replicate(draw, 0.0, 0.0, d, n_bins, n, replicates, [20260906, 1002, LAW_INDEX[name], n])
            scale = n**exponent
            rng_n = np.random.default_rng(np.random.SeedSequence([20260906, 1002, LAW_INDEX[name], n]))
            # recompute the scaled distribution from the same stream for quantiles
            etas, sigmas = [], []
            for start in range(0, replicates, 250):
                size = min(250, replicates - start)
                s_all, z_all = zip(*[draw(rng_n, n) for _ in range(size)], strict=True)
                out = plugin_batch(np.stack(s_all), np.stack(z_all), n_bins)
                etas.append(out["eta"])
                sigmas.append(out["sigma"])
            eta_hat, sigma_hat = np.concatenate(etas), np.concatenate(sigmas)
            entry["rows"].append(
                {
                    "n": n,
                    "coverage_of_zero": rec["coverage"],
                    "scaled_mean": float(scale * eta_hat.mean()),
                    "scaled_sd": float(scale * eta_hat.std(ddof=1)),
                    "scaled_quantiles_10_50_90": [float(scale * np.quantile(eta_hat, q)) for q in (0.1, 0.5, 0.9)],
                    "scaled_half_width_median": float(scale * np.median(Z975 * sigma_hat / math.sqrt(n))),
                    "half_width_over_eta_hat_median": float(np.median(Z975 * sigma_hat / math.sqrt(n) / np.where(eta_hat > 0, eta_hat, np.nan))),
                }
            )
            print(name, n, entry["rows"][-1])
        record["laws"][name] = entry
    record["provenance"] = provenance("singular", {"replicates": replicates, "limit_draws": 1_000_000})
    return record


# --------------------------------------------------------------------------
# stage: fixtures


def stage_fixtures() -> dict[str, object]:
    atoms, n_bins = unit_retention_law()
    summary = law_summary(atoms, n_bins)
    mom = summary["moments"]
    failing_scores = [[1, 0], [1, 0]]
    failing_labels = [0, 0]
    sample = sample_atoms([_vec(*s) for s in failing_scores], failing_labels)
    mom_s = law_moments(sample, n_bins)
    lib, eff = library_retention(sample, n_bins)
    fixture = {
        "id": FIXTURE_ID,
        "criterion": "D",
        "level": "information_accounting",
        "claim_falsified": "At eta_D = 1 (S = c_Z almost surely) the frozen-rule geometric-mean retention plug-in eta_hat_D = (det I_hat_Z / det V_hat)^(1/d) equals 1 on every sample.",
        "scores": failing_scores,
        "weights": ["1/2", "1/2"],
        "K": n_bins,
        "labels_before": failing_labels,
        "labels_after_or_optimum": None,
        "poi_indices": [0, 1],
        "nuisance_indices": [],
        "objective_before": "exact plug-in eta_hat_D = 0 (det V_hat = 0, phi := 0); library geometric_mean_retention = 1 on the retained rank-1 subspace",
        "objective_after": None,
        "population_law": {
            "description": "d = 2, K = 2, S = c_Z a.s. with c_0 = (1, 0), c_1 = (0, 1), p_b = 1/2",
            "atoms": [[fr(x) for x in s] for s, _, _ in atoms],
            "labels": [z for _, z, _ in atoms],
            "weights": [fr(w) for _, _, w in atoms],
            "V": frm(mom["V"]),
            "I_Z": frm(mom["I_Z"]),
            "det_ratio": fr(summary["ratio"]),
            "eta_D": "1",
            "psi_values": [fr(x) for x in summary["B"]],
            "sigma2": "0",
        },
        "exact_quantities": {
            "sample": "n = 2, both draws in cell 0 (probability 2^{-n} per one-cell composition, so P(V_hat singular) = 2^{1-n} > 0 for every n)",
            "V_hat": frm(mom_s["V"]),
            "I_hat_Z": frm(mom_s["I_Z"]),
            "det_V_hat": fr(det(mom_s["V"])),
            "exact_plugin": "0",
            "library_geometric_mean_retention": lib,
            "library_effective_rank": eff,
            "regular_samples": "every composition with both cells occupied has V_hat = I_hat_Z and plug-in exactly 1 (exhaustive to n = 6)",
        },
        "verification": {
            "method": "exact_formula",
            "notes": "V_hat = diag(1, 0) is singular, so the estimator's own everywhere-defined functional phi returns 0 by its det V_hat = 0 convention while the population value is 1; the library projects the null direction out (rank_rtol) and reports the ratio on the retained subspace, which is 1. Corrected statement: at eta_D = 1 the plug-in is 1 on every sample with V_hat positive definite, i.e. whenever the occupied cells' means span R^d; the singular event has probability at most sum_b (1 - p_b)^n ... P(fewer than d spanning cells occupied) -> 0 exponentially under A1, so the CLT statement is unaffected, only the 'every sample' sentence.",
        },
        "source": "AUDIT-RETENTION-PLUGIN-VECTOR independent audit",
        "date": "2026-09-06",
    }
    path = RESEARCH / "COUNTEREXAMPLES" / f"{FIXTURE_ID}.json"
    path.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")
    return {"fixture": fixture, "provenance": provenance("fixtures", {})}


STAGES = {
    "exact": stage_exact,
    "degenerate": stage_degenerate,
    "popref": stage_popref,
    "coverage": stage_coverage,
    "singular": stage_singular,
    "fixtures": stage_fixtures,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", choices=[*STAGES, "all"])
    parser.add_argument("--replicates", type=int, default=4000)
    args = parser.parse_args()
    names = list(STAGES) if args.stage == "all" else [args.stage]
    for name in names:
        fn = STAGES[name]
        record = fn(args.replicates) if name in ("coverage", "singular") else fn()
        write_record(name, record)


if __name__ == "__main__":
    main()
