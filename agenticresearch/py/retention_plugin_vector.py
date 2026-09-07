"""Frozen-rule vector retention uncertainty for RETENTION-PLUGIN-VECTOR (OP27, vector case).

Instrument for KNOWN_RESULTS/10-oracle.md section O7. Conditional on a frozen
label map Z = q(s_hat(X)) and a d-dimensional *true* score S, it studies the
plug-in geometric-mean retention

    eta_hat_D = (det I_hat_Z / det V_hat)^(1/d),
    V_hat = mean(S S^T),  I_hat_Z = sum_b m_hat_b m_hat_b^T / p_hat_b  (0/0 := 0),

on an independent iid equally weighted evaluation sample, and the delta-method
variance built from the matrix influence function

    psi = (eta_D / d) [ 2 S^T I_Z^{-1} c_Z - c_Z^T I_Z^{-1} c_Z - S^T V^{-1} S ],
    c_b = m_b / p_b.

Modes
-----
selftest   exact `fractions.Fraction` identities on deterministic integer
           samples in d = 2, 3: the within-cell-scatter form of V_hat - I_hat_Z,
           0 <= det ratio <= 1, sum psi_hat = 0, the influence function of the
           rational functional r = eta^d against exact Gateaux difference
           quotients (with the O(eps) remainder checked to shrink linearly),
           the exact zero of the plug-in when K <= d on an exactly centred
           sample, the reduction to the O6 scalar psi at d = 1, agreement with
           ``scorequant.information_report``, and the rank_rtol projection
           discontinuity of the library estimator.
fixtures   the exact sigma^2 = 0 law with 0 < eta_D < 1 in d = 2 (four cells
           related by quarter turns, two atoms per cell on a circle) and the
           polynomial identity that makes every law on that circle with the
           same cell mean -- atomless ones included -- a sigma^2 = 0 law;
           serializes COUNTEREXAMPLES/CE-O7-ELLIPSOID-ZERO-VARIANCE-001.json.
popref     closed-form population references for the Hermite laws
           S = (x, x^2 - 1) and S = (x, x^2 - 1, x^3 - 3x), x ~ N(0, 1), under
           frozen interval partitions of x (truncated-Gaussian moment
           recursion; second route by composite Gauss-Legendre), plus a
           library-fitted D-exchange rule on the d = 2 Hermite score whose
           cells in x are located by bisection; and the singular-I_Z endpoint
           law (K = d).
coverage   seeded Monte Carlo coverage of the 95% Wald interval at several n,
           variance-estimate consistency, studentized moments, the n^{-(d-r)/d}
           rate at the singular endpoint, and the conservative/liberal
           behaviour on the sigma^2 = 0 arc law.

Nothing here is a proof. Artifacts under WORK/artifacts/RETENTION-PLUGIN-VECTOR/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scorequant as sq  # noqa: E402

SEED_BASE = 20260906
WORKSPACE = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = WORKSPACE / "WORK" / "artifacts" / "RETENTION-PLUGIN-VECTOR"
FIXTURE_PATH = WORKSPACE / "COUNTEREXAMPLES" / "CE-O7-ELLIPSOID-ZERO-VARIANCE-001.json"
Z_95 = float(norm.ppf(0.975))


def provenance(mode: str, params: dict) -> dict:
    script = Path(__file__).resolve()
    rev = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=script.parent,
        check=False,
    ).stdout.strip()
    return {
        "mode": mode,
        "params": params,
        "git_revision": rev,
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scorequant": getattr(sq, "__version__", "unknown"),
    }


# ----------------------------------------------------------------------------
# Exact rational linear algebra (small d)
# ----------------------------------------------------------------------------

Mat = list[list[Fraction]]
Vec = list[Fraction]


def _zeros(d: int) -> Mat:
    return [[Fraction(0)] * d for _ in range(d)]


def _outer(u: Vec, v: Vec) -> Mat:
    return [[a * b for b in v] for a in u]


def _add(a: Mat, b: Mat, scale: Fraction = Fraction(1)) -> Mat:
    return [[x + scale * y for x, y in zip(ra, rb, strict=True)] for ra, rb in zip(a, b, strict=True)]


def _det(a: Mat) -> Fraction:
    n = len(a)
    m = [row[:] for row in a]
    det = Fraction(1)
    for col in range(n):
        pivot = next((r for r in range(col, n) if m[r][col] != 0), None)
        if pivot is None:
            return Fraction(0)
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
            det = -det
        det *= m[col][col]
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            if f:
                m[r] = [x - f * y for x, y in zip(m[r], m[col], strict=True)]
    return det


def _inv(a: Mat) -> Mat:
    n = len(a)
    m = [row[:] + [Fraction(int(i == j)) for j in range(n)] for i, row in enumerate(a)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if m[r][col] != 0), None)
        if pivot is None:
            raise ZeroDivisionError("singular matrix")
        m[col], m[pivot] = m[pivot], m[col]
        p = m[col][col]
        m[col] = [x / p for x in m[col]]
        for r in range(n):
            if r != col and m[r][col]:
                f = m[r][col]
                m[r] = [x - f * y for x, y in zip(m[r], m[col], strict=True)]
    return [row[n:] for row in m]


def _quad(a: Mat, u: Vec, v: Vec) -> Fraction:
    return sum(u[i] * a[i][j] * v[j] for i in range(len(u)) for j in range(len(v)))


@dataclass(frozen=True)
class ExactPlugIn:
    d: int
    n_bins: int
    p: Vec
    m: list[Vec]
    v: Mat
    i_z: Mat
    det_ratio: Fraction  # r = eta_D^d, exact
    psi_r: Vec  # influence function of r (= d * eta^{d-1} * psi), per row
    within_scatter: Mat


def exact_plugin(scores: list[Vec], labels: list[int], n_bins: int, weights: Vec | None = None) -> ExactPlugIn:
    """Exact plug-in on a weighted atom sample (weights default to 1/n)."""
    n = len(scores)
    d = len(scores[0])
    w = [Fraction(1, n)] * n if weights is None else list(weights)
    total = sum(w)
    w = [x / total for x in w]
    p = [Fraction(0)] * n_bins
    m = [[Fraction(0)] * d for _ in range(n_bins)]
    v = _zeros(d)
    for s, z, wi in zip(scores, labels, w, strict=True):
        p[z] += wi
        m[z] = [a + wi * b for a, b in zip(m[z], s, strict=True)]
        v = _add(v, _outer(s, s), wi)
    i_z = _zeros(d)
    for b in range(n_bins):
        if p[b] > 0:
            i_z = _add(i_z, _outer(m[b], m[b]), 1 / p[b])
    det_v = _det(v)
    det_iz = _det(i_z)
    ratio = det_iz / det_v if det_v != 0 else Fraction(0)
    within = _zeros(d)
    for s, z, wi in zip(scores, labels, w, strict=True):
        c = [x / p[z] for x in m[z]]
        resid = [a - b for a, b in zip(s, c, strict=True)]
        within = _add(within, _outer(resid, resid), wi)
    psi_r: Vec = []
    if det_iz != 0 and det_v != 0:
        iz_inv = _inv(i_z)
        v_inv = _inv(v)
        for s, z in zip(scores, labels, strict=True):
            c = [x / p[z] for x in m[z]]
            psi_r.append(ratio * (2 * _quad(iz_inv, s, c) - _quad(iz_inv, c, c) - _quad(v_inv, s, s)))
    return ExactPlugIn(d, n_bins, p, m, v, i_z, ratio, psi_r, within)


def exact_det_ratio_weighted(scores: list[Vec], labels: list[int], n_bins: int, weights: Vec) -> Fraction:
    return exact_plugin(scores, labels, n_bins, weights).det_ratio


# ----------------------------------------------------------------------------
# Float plug-in (vectorized, for coverage)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class PlugIn:
    eta: float
    sigma2: float
    det_ratio: float
    counts: np.ndarray
    psi: np.ndarray

    @property
    def n_empty(self) -> int:
        return int(np.sum(self.counts == 0))


def plugin_retention(scores: np.ndarray, labels: np.ndarray, n_bins: int) -> PlugIn:
    """Plug-in eta_hat_D, its influence-function variance estimate, and the pieces.

    Empty cells contribute nothing (0/0 := 0). If I_hat_Z or V_hat is singular
    the plug-in is 0 and psi is undefined (returned as zeros, sigma2 = 0): that
    is the endpoint outside the theorem, not a value of the influence function.
    """
    s = np.asarray(scores, dtype=float)
    z = np.asarray(labels).reshape(-1)
    n, d = s.shape
    counts = np.bincount(z, minlength=n_bins).astype(float)
    sums = np.zeros((n_bins, d))
    np.add.at(sums, z, s)
    means = np.divide(sums, counts[:, None], out=np.zeros((n_bins, d)), where=counts[:, None] > 0)
    v = s.T @ s / n
    i_z = (sums.T * np.divide(1.0, counts, out=np.zeros(n_bins), where=counts > 0)) @ sums / n
    sign_v, logdet_v = np.linalg.slogdet(v)
    sign_i, logdet_i = np.linalg.slogdet(i_z)
    if sign_v <= 0 or sign_i <= 0 or not np.isfinite(logdet_i):
        return PlugIn(0.0, 0.0, 0.0, counts, np.zeros(n))
    ratio = float(np.exp(logdet_i - logdet_v))
    eta = ratio ** (1.0 / d)
    iz_inv = np.linalg.inv(i_z)
    v_inv = np.linalg.inv(v)
    c = means[z]
    psi = (eta / d) * (
        2.0 * np.einsum("ni,ij,nj->n", s, iz_inv, c)
        - np.einsum("ni,ij,nj->n", c, iz_inv, c)
        - np.einsum("ni,ij,nj->n", s, v_inv, s)
    )
    return PlugIn(eta, float(np.mean(psi * psi)), ratio, counts, psi)


def wald_interval(fit: PlugIn, n: int, z: float = Z_95) -> tuple[float, float]:
    half = z * math.sqrt(fit.sigma2 / n)
    return fit.eta - half, fit.eta + half


# ----------------------------------------------------------------------------
# selftest: exact identities
# ----------------------------------------------------------------------------


def _fr(x: object) -> str:
    return str(x)


def _gateaux_check(scores: list[Vec], labels: list[int], n_bins: int) -> dict:
    """Exact Gateaux difference quotients of r = det ratio versus psi_r.

    For each row j, phi((1 - eps) P_n + eps delta_j) - phi(P_n) over eps at
    eps = 2^-10, 2^-11, 2^-12: the gap to psi_r,j must shrink by a factor 2
    per halving (the O(eps) remainder of a smooth functional).
    """
    n = len(scores)
    base = exact_plugin(scores, labels, n_bins)
    worst_gap = Fraction(0)
    worst_ratio_dev = Fraction(0)
    for j in range(n):
        gaps = []
        for k in (10, 11, 12):
            eps = Fraction(1, 2**k)
            w = [(1 - eps) / n] * n
            w[j] += eps
            r_eps = exact_det_ratio_weighted(scores, labels, n_bins, w)
            quotient = (r_eps - base.det_ratio) / eps
            gaps.append(quotient - base.psi_r[j])
        worst_gap = max(worst_gap, abs(gaps[0]))
        for a, b in ((gaps[0], gaps[1]), (gaps[1], gaps[2])):
            if b != 0:
                worst_ratio_dev = max(worst_ratio_dev, abs(a / b - 2))
            elif a != 0:
                worst_ratio_dev = max(worst_ratio_dev, Fraction(10**6))
    return {
        "worst_first_order_gap_at_eps_2^-10": float(worst_gap),
        "worst_deviation_of_remainder_halving_ratio_from_2": float(worst_ratio_dev),
        "pass": bool(worst_gap < Fraction(1, 100) and worst_ratio_dev < Fraction(1, 5)),
    }


def _sample_d2() -> tuple[list[Vec], list[int], int]:
    rows = [(2, 1), (-1, 3), (0, -2), (3, 3), (-2, -1), (1, 0), (1, 0), (-3, 2), (2, -3), (0, 1), (4, -1), (-1, -1)]
    labels = [0, 1, 2, 0, 1, 2, 2, 1, 3, 3, 0, 1]  # duplicate atoms (1,0) split across cells 2 and 2; cell 4 declared empty
    scores = [[Fraction(a), Fraction(b)] for a, b in rows]
    return scores, labels, 5


def _sample_d3() -> tuple[list[Vec], list[int], int]:
    rows = [
        (1, 2, 0), (-2, 1, 1), (0, -1, 3), (3, 0, -1), (1, 1, 1), (-1, -2, 2), (2, -1, -2),
        (0, 3, 1), (-3, 1, 0), (1, 0, -3), (2, 2, 2), (-1, 1, -1), (0, 0, 1), (1, -3, 0),
    ]
    labels = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 3, 3]
    scores = [[Fraction(a), Fraction(b), Fraction(c)] for a, b, c in rows]
    return scores, labels, 4


def run_selftest() -> dict:
    report: dict = {"provenance": provenance("selftest", {}), "cases": {}}
    ok = True
    for name, (scores, labels, n_bins) in (("d2", _sample_d2()), ("d3", _sample_d3())):
        fit = exact_plugin(scores, labels, n_bins)
        d = fit.d
        # (i) within-scatter identity: V_hat - I_hat_Z = within-cell scatter about cell means
        scatter_ok = _add(fit.v, fit.i_z, Fraction(-1)) == fit.within_scatter
        # (ii) 0 <= r <= 1
        range_ok = 0 <= fit.det_ratio <= 1
        # (iii) sum psi = 0 exactly
        zero_mean_ok = sum(fit.psi_r) == 0
        # (iv) Gateaux
        gateaux = _gateaux_check(scores, labels, n_bins)
        # (v) library agreement
        lib = sq.information_report(
            np.array([[float(x) for x in s] for s in scores]), np.array(labels), n_bins=n_bins
        )
        eta_exact = float(fit.det_ratio) ** (1.0 / d)
        lib_gap = abs(float(lib.geometric_mean_retention) - eta_exact)
        case_ok = scatter_ok and range_ok and zero_mean_ok and gateaux["pass"] and lib_gap < 1e-12
        ok = ok and case_ok
        report["cases"][name] = {
            "d": d,
            "n": len(scores),
            "K_declared": n_bins,
            "n_empty_cells": sum(1 for x in fit.p if x == 0),
            "det_ratio": _fr(fit.det_ratio),
            "eta_D": eta_exact,
            "within_scatter_identity": scatter_ok,
            "det_ratio_in_unit_interval": range_ok,
            "sum_psi_hat_zero": zero_mean_ok,
            "gateaux": gateaux,
            "library_geometric_mean_retention_gap": lib_gap,
            "library_effective_rank": int(lib.effective_rank),
            "pass": case_ok,
        }

    # (vi) K <= d on an exactly centred sample: sum_b m_hat_b = 0 forces rank <= K - 1 < d, r = 0 exactly
    scores, labels, _ = _sample_d2()
    total = [sum(s[i] for s in scores) for i in range(2)]
    centred = [[x - t / len(scores) for x, t in zip(s, total, strict=True)] for s in scores]
    two_cell = [z % 2 for z in labels]
    fit2 = exact_plugin(centred, two_cell, 2)
    lib2 = sq.information_report(np.array([[float(x) for x in s] for s in centred]), np.array(two_cell), n_bins=2)
    # The library takes the d-th root of a rounding-level determinant, so it reports about eps^(1/d),
    # not 0: an exactly singular I_hat_Z surfaces as a positive number of order 1e-9 at d = 2.
    k_le_d_ok = fit2.det_ratio == 0 and float(lib2.geometric_mean_retention) < 1e-7
    report["K_equals_d_centred_sample"] = {
        "det_ratio_exact": _fr(fit2.det_ratio),
        "library_geometric_mean_retention": float(lib2.geometric_mean_retention),
        "note": "exact 0; the library's value is the d-th root of a rounding-level determinant",
        "pass": k_le_d_ok,
    }
    ok = ok and k_le_d_ok

    # (vii) d = 1 reduction to the O6 influence function
    from agenticresearch.py import score_oracle_retention_uncertainty as o6

    rng = np.random.default_rng(SEED_BASE)
    s1 = rng.standard_t(df=5, size=60)
    z1 = rng.integers(0, 4, size=60)
    fit_o6 = o6.plugin_retention(s1, z1, 4)
    fit_o7 = plugin_retention(s1[:, None], z1, 4)
    d1_gap = float(np.max(np.abs(fit_o6.psi - fit_o7.psi))) + abs(fit_o6.eta - fit_o7.eta)
    report["d1_reduction_to_o6"] = {"max_gap": d1_gap, "pass": d1_gap < 1e-12}
    ok = ok and d1_gap < 1e-12

    # (viii) rank_rtol discontinuity: an exactly zero third coordinate is projected out by the
    # library (rank 2, exponent 1/2), while the theorem's (A3) excludes the law altogether.
    s3 = np.column_stack([np.array([[float(x) for x in s] for s in scores]), np.zeros(len(scores))])
    lib3 = sq.information_report(s3, np.array(labels), n_bins=5)
    fit_full = exact_plugin([s[:2] for s in scores], labels, 5)
    report["rank_rtol_projection"] = {
        "library_effective_rank": int(lib3.effective_rank),
        "library_geometric_mean_retention": float(lib3.geometric_mean_retention),
        "two_dimensional_plugin": float(fit_full.det_ratio) ** 0.5,
        "note": "the library reports the 2-D ratio with exponent 1/rank; the 3-D plug-in of the theorem is 0/0",
    }
    report["pass"] = ok
    return report


# ----------------------------------------------------------------------------
# fixtures: the sigma^2 = 0 ellipsoid law
# ----------------------------------------------------------------------------


def _rot90(v: Vec, k: int) -> Vec:
    x, y = v
    for _ in range(k % 4):
        x, y = -y, x
    return [x, y]


def ellipsoid_fixture_atoms() -> tuple[list[Vec], list[int], Vec]:
    """Four cells related by quarter turns; cell 0 carries (3, 4) and (3, -4) with weight 1/8 each."""
    base = [[Fraction(3), Fraction(4)], [Fraction(3), Fraction(-4)]]
    scores: list[Vec] = []
    labels: list[int] = []
    weights: Vec = []
    for b in range(4):
        for atom in base:
            scores.append(_rot90(atom, b))
            labels.append(b)
            weights.append(Fraction(1, 8))
    return scores, labels, weights


def run_fixtures(write: bool = True) -> dict:
    scores, labels, weights = ellipsoid_fixture_atoms()
    fit = exact_plugin(scores, labels, 4, weights)
    d = 2
    mean = [sum(w * s[i] for s, w in zip(scores, weights, strict=True)) for i in range(d)]
    eta_d = fit.det_ratio  # eta_D^2
    checks = {
        "E_S_zero": mean == [0, 0],
        "V": [[_fr(x) for x in row] for row in fit.v],
        "I_Z": [[_fr(x) for x in row] for row in fit.i_z],
        "cell_probabilities": [_fr(x) for x in fit.p],
        "cell_means": [[_fr(x / p) for x in m] for m, p in zip(fit.m, fit.p, strict=True)],
        "det_ratio": _fr(eta_d),
        "eta_D": _fr(Fraction(9, 25)) if eta_d == Fraction(81, 625) else float(eta_d) ** 0.5,
        "psi_values": [_fr(x) for x in fit.psi_r],
        "sigma2": _fr(sum(w * x * x for x, w in zip(fit.psi_r, weights, strict=True))),
    }
    v_ok = fit.v == [[Fraction(25, 2), Fraction(0)], [Fraction(0), Fraction(25, 2)]]
    iz_ok = fit.i_z == [[Fraction(9, 2), Fraction(0)], [Fraction(0), Fraction(9, 2)]]
    psi_ok = all(x == 0 for x in fit.psi_r)
    # Ellipsoid (here a circle) of cell 0: (s - centre)^T V^{-1} (s - centre) = c^T I_Z^{-1} (V - I_Z) I_Z^{-1} c
    # with centre V I_Z^{-1} c_0 = (25/3, 0) and right-hand side 32/9, i.e. Euclidean radius^2 = (32/9)(25/2) = 400/9
    iz_inv = _inv(fit.i_z)
    v_inv = _inv(fit.v)
    c0 = [x / fit.p[0] for x in fit.m[0]]
    centre = [sum(fit.v[i][j] * sum(iz_inv[j][k] * c0[k] for k in range(d)) for j in range(d)) for i in range(d)]
    vmi = _add(fit.v, fit.i_z, Fraction(-1))
    t = [sum(iz_inv[j][k] * c0[k] for k in range(d)) for j in range(d)]
    radius2 = _quad(vmi, t, t)
    euclid_radius2 = radius2 * fit.v[0][0]
    ellipsoid_ok = centre == [Fraction(25, 3), Fraction(0)] and radius2 == Fraction(32, 9) and euclid_radius2 == Fraction(400, 9)
    # Polynomial identity: psi_r(s) = -(2 r / 25) * (|s|^2 - (50/3) s_1 + 25) for every s in R^2,
    # so psi vanishes on the whole circle, not only at the two atoms. Checked on rational circle
    # points s(t) = (25/3 + (20/3)(1 - t^2)/(1 + t^2), (20/3) 2t/(1 + t^2)) and at generic points.
    def psi_r_at(s: Vec) -> Fraction:
        return eta_d * (2 * _quad(iz_inv, s, c0) - _quad(iz_inv, c0, c0) - _quad(v_inv, s, s))

    def circle_poly(s: Vec) -> Fraction:
        return s[0] ** 2 + s[1] ** 2 - Fraction(50, 3) * s[0] + 25

    identity_ok = True
    circle_zero_ok = True
    for num in range(-7, 8):
        t = Fraction(num, 3)
        s_t = [
            Fraction(25, 3) + Fraction(20, 3) * (1 - t * t) / (1 + t * t),
            Fraction(20, 3) * 2 * t / (1 + t * t),
        ]
        circle_zero_ok = circle_zero_ok and psi_r_at(s_t) == 0
        generic = [t + Fraction(1, 7), t * t - 2]
        identity_ok = identity_ok and psi_r_at(generic) + Fraction(2, 25) * eta_d * circle_poly(generic) == 0
    lib = sq.information_report(
        np.array([[float(x) for x in s] for s in scores]),
        np.array(labels),
        np.array([float(w) for w in weights]),
        n_bins=4,
    )
    lib_ok = abs(float(lib.geometric_mean_retention) - 0.36) < 1e-14
    # The atomless variant: theta uniform on [pi - alpha, pi + alpha] with sin(alpha)/alpha = 4/5 puts the
    # mean of the uniform-on-arc law at the cell mean (3, 0); alpha is transcendental, so it is a float.
    alpha = _solve_arc_half_width(0.8)
    arc = {
        "alpha": alpha,
        "mean_check": (25.0 / 3.0 + (20.0 / 3.0) * (-math.sin(alpha) / alpha), 0.0),
        "second_moment_check": 25.0,
        "note": "any law on the circle with mean (3, 0) has psi = 0 identically (polynomial identity above)",
    }
    ok = v_ok and iz_ok and psi_ok and ellipsoid_ok and identity_ok and circle_zero_ok and lib_ok and checks["E_S_zero"]
    fixture = {
        "id": "CE-O7-ELLIPSOID-ZERO-VARIANCE-001",
        "criterion": "D",
        "level": "information_accounting",
        "claim_falsified": (
            "The scalar characterisation of sigma^2 = 0 in RETENTION-PLUGIN-CLT-FROZEN-SCALAR (A4) lifts to "
            "vector scores: for 0 < eta_D < 1 the influence variance of the frozen-rule geometric-mean retention "
            "plug-in vanishes only for laws with at most two atoms per cell, so an atomless cell of positive "
            "probability implies sigma^2 > 0 whenever eta_D > 0."
        ),
        "scores": [[int(x) for x in s] for s in scores],
        "weights": [_fr(w) for w in weights],
        "K": 4,
        "labels_before": labels,
        "labels_after_or_optimum": None,
        "poi_indices": [0, 1],
        "nuisance_indices": [],
        "objective_before": "eta_D = 9/25 (det ratio 81/625)",
        "objective_after": None,
        "exact_quantities": {
            "cell_probabilities": checks["cell_probabilities"],
            "cell_means": checks["cell_means"],
            "V": checks["V"],
            "I_Z": checks["I_Z"],
            "det_ratio": checks["det_ratio"],
            "eta_D": "9/25",
            "psi_values": checks["psi_values"],
            "sigma2": checks["sigma2"],
            "zero_variance_ellipsoid_cell_0": {
                "centre": ["25/3", "0"],
                "V_inverse_metric_radius_squared": "32/9",
                "euclidean_radius_squared": "400/9",
            },
            "polynomial_identity": "psi_r(s) = -(2 r / 25) (s_1^2 + s_2^2 - (50/3) s_1 + 25), r = eta_D^2 = 81/625",
        },
        "atomless_variant": {
            "description": (
                "S | Z = 0 uniform on the arc of the same circle with polar angle in [pi - alpha, pi + alpha] "
                "about the centre (25/3, 0), alpha the root of sin(alpha)/alpha = 4/5; cells 1-3 by quarter turns"
            ),
            "alpha": alpha,
            "cell_mean": [3, 0],
            "second_moment": 25,
        },
        "verification": {
            "method": "exact_formula",
            "notes": (
                "E[S] = 0, p_b = 1/4, V = (25/2) I, I_Z = (9/2) I, det ratio 81/625, eta_D = 9/25. psi = (eta_D/2)[2 s^T I_Z^{-1} c_b - "
                "c_b^T I_Z^{-1} c_b - s^T V^{-1} s] = 0 at every atom, and as a polynomial in s it is a multiple of "
                "the circle |s - (25/3, 0)|^2 = (20/3)^2, the sigma^2 = 0 ellipsoid of O7.4. Hence every law on that "
                "circle with mean (3, 0) - the two-atom law and the atomless arc law alike - has sigma^2 = 0 at "
                "eta_D = 9/25. Correct statement: sigma^2 = 0 iff S | Z = b is supported on the cell's ellipsoid for "
                "every cell; absolutely continuous laws are excluded, atomless singular laws are not."
            ),
        },
        "source": "RETENTION-PLUGIN-VECTOR theorem investigation",
        "date": "2026-09-06",
    }
    if write and ok:
        FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")
    return {
        "provenance": provenance("fixtures", {}),
        "checks": {
            **checks,
            "V_is_25_over_2_identity": v_ok,
            "I_Z_is_9_over_2_identity": iz_ok,
            "psi_zero_at_every_atom": psi_ok,
            "ellipsoid_centre_and_radius": ellipsoid_ok,
            "psi_zero_on_15_rational_circle_points": circle_zero_ok,
            "polynomial_identity_at_15_generic_points": identity_ok,
            "library_geometric_mean_retention": float(lib.geometric_mean_retention),
        },
        "atomless_variant": arc,
        "fixture_written": bool(write and ok),
        "pass": ok,
    }


def _solve_arc_half_width(target: float) -> float:
    lo, hi = 1e-9, math.pi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if math.sin(mid) / mid > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------------
# popref: Hermite laws with closed-form truncated-Gaussian moments
# ----------------------------------------------------------------------------


def truncated_moments(a: float, b: float, kmax: int) -> np.ndarray:
    """M_k = int_a^b x^k phi(x) dx for k = 0..kmax (a, b may be +-inf)."""
    m = np.zeros(kmax + 1)
    pa = norm.pdf(a) if np.isfinite(a) else 0.0
    pb = norm.pdf(b) if np.isfinite(b) else 0.0
    m[0] = norm.cdf(b) - norm.cdf(a)
    if kmax >= 1:
        m[1] = pa - pb
    for k in range(2, kmax + 1):
        ta = a ** (k - 1) * pa if np.isfinite(a) else 0.0
        tb = b ** (k - 1) * pb if np.isfinite(b) else 0.0
        m[k] = (k - 1) * m[k - 2] + ta - tb
    return m


HERMITE = {
    1: np.array([0.0, 1.0]),
    2: np.array([-1.0, 0.0, 1.0]),
    3: np.array([0.0, -3.0, 0.0, 1.0]),
}


def hermite_scores(x: np.ndarray, d: int) -> np.ndarray:
    return np.column_stack([np.polynomial.polynomial.polyval(x, HERMITE[k]) for k in range(1, d + 1)])


@dataclass(frozen=True)
class IntervalLaw:
    name: str
    d: int
    cuts: tuple[float, ...]

    @property
    def n_bins(self) -> int:
        return len(self.cuts) + 1

    def labels(self, x: np.ndarray) -> np.ndarray:
        return np.searchsorted(np.asarray(self.cuts), x, side="right")

    def edges(self) -> list[tuple[float, float]]:
        pts = [-np.inf, *self.cuts, np.inf]
        return list(zip(pts[:-1], pts[1:], strict=True))


def _poly_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.polynomial.polynomial.polymul(a, b)


def _poly_expect(coef: np.ndarray, moments: np.ndarray) -> float:
    return float(np.dot(coef, moments[: len(coef)]))


def population_interval_law(law: IntervalLaw) -> dict:
    """Closed-form eta_D, sigma^2 and the pieces for a Hermite law under interval cells."""
    d = law.d
    pieces = law.edges()
    kmax = 4 * d + 2
    moms = [truncated_moments(a, b, kmax) for a, b in pieces]
    p = np.array([m[0] for m in moms])
    m_cells = np.array([[_poly_expect(HERMITE[k], mm) for k in range(1, d + 1)] for mm in moms])
    v = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            coef = _poly_mul(HERMITE[i + 1], HERMITE[j + 1])
            v[i, j] = sum(_poly_expect(coef, mm) for mm in moms)
    i_z = (m_cells.T / p) @ m_cells
    rank = int(np.linalg.matrix_rank(i_z, tol=1e-12))
    out = {
        "law": law.name,
        "d": d,
        "K": law.n_bins,
        "cuts": list(law.cuts),
        "p": p.tolist(),
        "cell_means": (m_cells / p[:, None]).tolist(),
        "V": v.tolist(),
        "I_Z": i_z.tolist(),
        "rank_I_Z": rank,
        "E_S": m_cells.sum(axis=0).tolist(),
    }
    if rank < d:
        out.update({"eta_D": 0.0, "sigma": None, "note": "singular I_Z: endpoint law, eta_D = 0"})
        return out
    ratio = float(np.linalg.det(i_z) / np.linalg.det(v))
    eta = ratio ** (1.0 / d)
    iz_inv = np.linalg.inv(i_z)
    v_inv = np.linalg.inv(v)
    # psi on cell b is a polynomial in x: (eta/d)[2 S^T A c_b - c_b^T A c_b - S^T B S]
    sigma2 = 0.0
    mean_psi = 0.0
    fourth = 0.0
    for b, (mm, cb) in enumerate(zip(moms, m_cells / p[:, None], strict=True)):
        poly = np.array([-(cb @ iz_inv @ cb)])
        for i in range(d):
            poly = np.polynomial.polynomial.polyadd(poly, 2.0 * float(iz_inv[i] @ cb) * HERMITE[i + 1])
            for j in range(d):
                poly = np.polynomial.polynomial.polyadd(poly, -v_inv[i, j] * _poly_mul(HERMITE[i + 1], HERMITE[j + 1]))
        poly = poly * (eta / d)
        mean_psi += _poly_expect(poly, mm)
        sigma2 += _poly_expect(_poly_mul(poly, poly), mm)
        del b
    # E||S||^4 (A2)
    norm2 = np.zeros(1)
    for i in range(d):
        norm2 = np.polynomial.polynomial.polyadd(norm2, _poly_mul(HERMITE[i + 1], HERMITE[i + 1]))
    fourth = sum(_poly_expect(_poly_mul(norm2, norm2), mm) for mm in moms)
    out.update(
        {
            "det_ratio": ratio,
            "eta_D": eta,
            "sigma2": sigma2,
            "sigma": math.sqrt(sigma2),
            "E_psi": mean_psi,
            "E_norm_S_4": fourth,
        }
    )
    return out


def _gauss_legendre_nodes(cuts: tuple[float, ...], half_range: float, panels_per_piece: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Composite Gauss-Legendre nodes whose pieces are aligned with the cut points.

    The integrands are polynomials times phi on each piece, discontinuous only
    at the cuts, so aligning the pieces makes the route converge spectrally;
    the tails beyond +-half_range are dropped (mass below 1e-40 at 14).
    """
    nodes, wts = np.polynomial.legendre.leggauss(order)
    pts = [-half_range, *cuts, half_range]
    xs = []
    ws = []
    for a, b in zip(pts[:-1], pts[1:], strict=True):
        edges = np.linspace(a, b, panels_per_piece + 1)
        mid = 0.5 * (edges[1:] + edges[:-1])
        half = 0.5 * (edges[1:] - edges[:-1])
        xs.append((mid[:, None] + half[:, None] * nodes[None, :]).ravel())
        ws.append((half[:, None] * wts[None, :]).ravel())
    x = np.concatenate(xs)
    w = np.concatenate(ws) * norm.pdf(x)
    return x, w


def _route_from_nodes(x: np.ndarray, w: np.ndarray, z: np.ndarray, d: int, K: int) -> dict:
    s = hermite_scores(x, d)
    p = np.bincount(z, weights=w, minlength=K)
    m = np.zeros((K, d))
    np.add.at(m, z, w[:, None] * s)
    v = (s * w[:, None]).T @ s
    i_z = (m.T / p) @ m
    ratio = float(np.linalg.det(i_z) / np.linalg.det(v))
    eta = ratio ** (1.0 / d)
    iz_inv = np.linalg.inv(i_z)
    v_inv = np.linalg.inv(v)
    c = (m / p[:, None])[z]
    psi = (eta / d) * (
        2.0 * np.einsum("ni,ij,nj->n", s, iz_inv, c)
        - np.einsum("ni,ij,nj->n", c, iz_inv, c)
        - np.einsum("ni,ij,nj->n", s, v_inv, s)
    )
    return {"eta_D": eta, "sigma2": float(np.sum(w * psi * psi)), "mass": float(np.sum(w))}


def _gauss_legendre_route(law: IntervalLaw, half_range: float, panels_per_piece: int, order: int) -> dict:
    """Independent quadrature route for eta_D and sigma^2 (labels evaluated at the nodes)."""
    x, w = _gauss_legendre_nodes(law.cuts, half_range, panels_per_piece, order)
    return _route_from_nodes(x, w, law.labels(x), law.d, law.n_bins)


@dataclass(frozen=True)
class MixtureLaw:
    """Three-component Gaussian location mixture with two free fractions: bounded d = 2 scores.

    lambda(x) = th_1 phi_1 + th_2 phi_2 + (1 - th_1 - th_2) phi_3 with phi_k = N(mu_k, s_k^2);
    the fraction scores s_k = (phi_k - phi_3) / lambda are bounded because every fraction is
    interior, so (A2) holds trivially - the library's mixture-fraction use case.
    """

    name: str
    cuts: tuple[float, ...]
    means: tuple[float, float, float] = (-1.5, 1.0, 0.0)
    sds: tuple[float, float, float] = (0.8, 0.6, 2.5)
    fractions: tuple[float, float] = (0.35, 0.25)
    d: int = 2

    @property
    def n_bins(self) -> int:
        return len(self.cuts) + 1

    def labels(self, x: np.ndarray) -> np.ndarray:
        return np.searchsorted(np.asarray(self.cuts), x, side="right")

    def components(self, x: np.ndarray) -> np.ndarray:
        return np.column_stack([norm.pdf(x, loc=m, scale=s) for m, s in zip(self.means, self.sds, strict=True)])

    def density(self, x: np.ndarray) -> np.ndarray:
        th = np.array([*self.fractions, 1.0 - sum(self.fractions)])
        return self.components(x) @ th

    def scores(self, x: np.ndarray) -> np.ndarray:
        comp = self.components(x)
        lam = self.density(x)
        return np.column_stack([(comp[:, 0] - comp[:, 2]) / lam, (comp[:, 1] - comp[:, 2]) / lam])

    def draw(self, rng: np.random.Generator, n: int) -> np.ndarray:
        th = np.array([*self.fractions, 1.0 - sum(self.fractions)])
        k = rng.choice(3, size=n, p=th)
        return rng.normal(np.asarray(self.means)[k], np.asarray(self.sds)[k])


def population_mixture_law(law: MixtureLaw, half_range: float, panels_per_piece: int, order: int) -> dict:
    """Quadrature route (pieces aligned with the cuts; integrands smooth on each piece)."""
    x, w = _gauss_legendre_nodes(law.cuts, half_range, panels_per_piece, order)
    w = w / norm.pdf(x) * law.density(x)  # reweight the standard-normal nodes to the mixture density
    s = law.scores(x)
    z = law.labels(x)
    K = law.n_bins
    p = np.bincount(z, weights=w, minlength=K)
    m = np.zeros((K, law.d))
    np.add.at(m, z, w[:, None] * s)
    v = (s * w[:, None]).T @ s
    i_z = (m.T / p) @ m
    ratio = float(np.linalg.det(i_z) / np.linalg.det(v))
    eta = ratio ** (1.0 / law.d)
    iz_inv = np.linalg.inv(i_z)
    v_inv = np.linalg.inv(v)
    c = (m / p[:, None])[z]
    psi = (eta / law.d) * (
        2.0 * np.einsum("ni,ij,nj->n", s, iz_inv, c)
        - np.einsum("ni,ij,nj->n", c, iz_inv, c)
        - np.einsum("ni,ij,nj->n", s, v_inv, s)
    )
    return {
        "law": law.name,
        "d": law.d,
        "K": K,
        "cuts": list(law.cuts),
        "mass": float(np.sum(w)),
        "p": p.tolist(),
        "E_S": m.sum(axis=0).tolist(),
        "cell_means": (m / p[:, None]).tolist(),
        "V": v.tolist(),
        "I_Z": i_z.tolist(),
        "rank_I_Z": int(np.linalg.matrix_rank(i_z, tol=1e-12)),
        "det_ratio": ratio,
        "eta_D": eta,
        "sigma2": float(np.sum(w * psi * psi)),
        "sigma": float(np.sqrt(np.sum(w * psi * psi))),
        "E_psi": float(np.sum(w * psi)),
        "E_norm_S_4": float(np.sum(w * np.sum(s * s, axis=1) ** 2)),
        "score_bound": float(np.max(np.abs(s))),
    }


MIXTURE = MixtureLaw("mixture_d2_K4_bounded", (-2.0, -0.5, 1.0))


LAWS = {
    "hermite_d2_K4": IntervalLaw("hermite_d2_K4", 2, (-1.0, 0.0, 1.0)),
    "hermite_d3_K5": IntervalLaw("hermite_d3_K5", 3, (-1.5, -0.5, 0.5, 1.5)),
    "hermite_d2_K2_singular": IntervalLaw("hermite_d2_K2_singular", 2, (0.0,)),
    "hermite_d3_K3_singular": IntervalLaw("hermite_d3_K3_singular", 3, (-0.6, 0.6)),
}


# --- library-fitted rule on the d = 2 Hermite score -------------------------


@dataclass(frozen=True)
class FittedRule:
    quantizer: object
    n_bins: int
    train_seed: int
    cuts: tuple[float, ...]  # label-change points in x located by bisection
    labels_between: tuple[int, ...]

    def labels(self, x: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(np.asarray(self.cuts), x, side="right")
        return np.asarray(self.labels_between)[idx]


def fit_library_rule(n_train: int = 4000, n_bins: int = 4, seed: int = 11) -> FittedRule:
    """Freeze a D-exchange rule on a training sample of the d = 2 Hermite score.

    The evaluation law then sees the *compiled* Mahalanobis rule; its cells in x
    are finite unions of intervals, located here by scanning a fine grid for
    label changes and bisecting each to 1e-13.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_train)
    s = hermite_scores(x, 2)
    result = sq.optimize_partition(s, n_bins=n_bins, config=sq.DExchangeConfig(seed=7, initializer_restarts=8))
    quantizer = result.compile_quantizer()

    def label(xx: np.ndarray) -> np.ndarray:
        return np.asarray(quantizer.predict_scores(hermite_scores(np.atleast_1d(xx), 2)))

    grid = np.linspace(-12.0, 12.0, 240_001)
    lg = label(grid)
    change = np.nonzero(lg[1:] != lg[:-1])[0]
    cuts = []
    for i in change:
        lo, hi = grid[i], grid[i + 1]
        llo = lg[i]
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if label(np.array([mid]))[0] == llo:
                lo = mid
            else:
                hi = mid
        cuts.append(0.5 * (lo + hi))
    labels_between = [int(lg[0])] + [int(lg[i + 1]) for i in change]
    return FittedRule(quantizer, n_bins, seed, tuple(cuts), tuple(labels_between))


def population_fitted_rule(rule: FittedRule) -> dict:
    """Closed-form moments per x-interval, then aggregated per label."""
    d = 2
    pts = [-np.inf, *rule.cuts, np.inf]
    kmax = 4 * d + 2
    K = rule.n_bins
    moms_by_label = [np.zeros(kmax + 1) for _ in range(K)]
    for (a, b), lab in zip(zip(pts[:-1], pts[1:], strict=True), rule.labels_between, strict=True):
        moms_by_label[lab] += truncated_moments(a, b, kmax)
    p = np.array([m[0] for m in moms_by_label])
    m_cells = np.array([[_poly_expect(HERMITE[k], mm) for k in range(1, d + 1)] for mm in moms_by_label])
    v = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            v[i, j] = sum(_poly_expect(_poly_mul(HERMITE[i + 1], HERMITE[j + 1]), mm) for mm in moms_by_label)
    i_z = (m_cells.T / p) @ m_cells
    ratio = float(np.linalg.det(i_z) / np.linalg.det(v))
    eta = ratio ** (1.0 / d)
    iz_inv = np.linalg.inv(i_z)
    v_inv = np.linalg.inv(v)
    sigma2 = 0.0
    for mm, cb in zip(moms_by_label, m_cells / p[:, None], strict=True):
        poly = np.array([-(cb @ iz_inv @ cb)])
        for i in range(d):
            poly = np.polynomial.polynomial.polyadd(poly, 2.0 * float(iz_inv[i] @ cb) * HERMITE[i + 1])
            for j in range(d):
                poly = np.polynomial.polynomial.polyadd(poly, -v_inv[i, j] * _poly_mul(HERMITE[i + 1], HERMITE[j + 1]))
        poly = poly * (eta / d)
        sigma2 += _poly_expect(_poly_mul(poly, poly), mm)
    return {
        "law": "hermite_d2_library_rule",
        "d": d,
        "K": K,
        "train_seed": rule.train_seed,
        "cuts_in_x": list(rule.cuts),
        "labels_between_cuts": list(rule.labels_between),
        "p": p.tolist(),
        "cell_means": (m_cells / p[:, None]).tolist(),
        "V": v.tolist(),
        "I_Z": i_z.tolist(),
        "det_ratio": ratio,
        "eta_D": eta,
        "sigma2": sigma2,
        "sigma": math.sqrt(sigma2),
    }


def run_popref() -> dict:
    out: dict = {"provenance": provenance("popref", {}), "laws": {}}
    for name, law in LAWS.items():
        ref = population_interval_law(law)
        if ref.get("sigma") is not None:
            r1 = _gauss_legendre_route(law, 14.0, 400, 24)
            r2 = _gauss_legendre_route(law, 16.0, 700, 16)
            ref["quadrature_routes"] = {
                "route_1": r1,
                "route_2": r2,
                "max_disagreement_with_closed_form": max(
                    abs(r1["eta_D"] - ref["eta_D"]),
                    abs(r2["eta_D"] - ref["eta_D"]),
                    abs(r1["sigma2"] - ref["sigma2"]),
                    abs(r2["sigma2"] - ref["sigma2"]),
                ),
            }
        out["laws"][name] = ref
    mix1 = population_mixture_law(MIXTURE, 30.0, 600, 24)
    mix2 = population_mixture_law(MIXTURE, 34.0, 900, 16)
    mix1["quadrature_routes"] = {
        "route_2": {k: mix2[k] for k in ("eta_D", "sigma2", "mass")},
        "max_disagreement": max(abs(mix1["eta_D"] - mix2["eta_D"]), abs(mix1["sigma2"] - mix2["sigma2"])),
    }
    out["laws"][MIXTURE.name] = mix1
    rule = fit_library_rule()
    ref = population_fitted_rule(rule)
    # cross-check with quadrature using the compiled rule's own labels at the nodes
    law_like = IntervalLaw("proxy", 2, rule.cuts)
    nodes_route = _gauss_legendre_route_labels(rule, 14.0, 400, 24)
    ref["quadrature_route_with_rule_labels"] = nodes_route
    ref["max_disagreement"] = max(abs(nodes_route["eta_D"] - ref["eta_D"]), abs(nodes_route["sigma2"] - ref["sigma2"]))
    del law_like
    out["laws"]["hermite_d2_library_rule"] = ref
    return out


def _gauss_legendre_route_labels(rule: FittedRule, half_range: float, panels_per_piece: int, order: int) -> dict:
    """Quadrature route that asks the compiled library rule for the label at every node."""
    x, w = _gauss_legendre_nodes(rule.cuts, half_range, panels_per_piece, order)
    z = np.asarray(rule.quantizer.predict_scores(hermite_scores(x, 2)))
    return _route_from_nodes(x, w, z, 2, rule.n_bins)


# ----------------------------------------------------------------------------
# coverage
# ----------------------------------------------------------------------------


def _moments(t: np.ndarray) -> dict:
    t = np.asarray(t, dtype=float)
    mu = float(np.mean(t))
    sd = float(np.std(t))
    skew = float(np.mean(((t - mu) / sd) ** 3)) if sd > 0 else float("nan")
    return {"mean": mu, "sd": sd, "skew": skew, "q95_abs": float(np.quantile(np.abs(t), 0.95))}


def _draw_hermite(rng: np.random.Generator, n: int, d: int, labeler) -> tuple[np.ndarray, np.ndarray]:
    x = rng.standard_normal(n)
    return hermite_scores(x, d), labeler(x)


def _draw_arc(rng: np.random.Generator, n: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """The atomless sigma^2 = 0 law of the fixture: uniform on the arc, cells by quarter turns."""
    z = rng.integers(0, 4, size=n)
    theta = math.pi + alpha * (2.0 * rng.random(n) - 1.0)
    s0 = np.column_stack([25.0 / 3.0 + (20.0 / 3.0) * np.cos(theta), (20.0 / 3.0) * np.sin(theta)])
    c, s_ = np.cos(0.5 * math.pi * z), np.sin(0.5 * math.pi * z)
    s = np.column_stack([c * s0[:, 0] - s_ * s0[:, 1], s_ * s0[:, 0] + c * s0[:, 1]])
    return s, z


def _coverage_rows(draw, d: int, n_bins: int, eta: float, sigma: float | None, sizes, reps, seed_tag: int) -> list[dict]:
    rows = []
    for n in sizes:
        rng = np.random.default_rng(np.random.SeedSequence([SEED_BASE, seed_tag, n]))
        eta_hat = np.zeros(reps)
        sigma_hat = np.zeros(reps)
        n_empty = np.zeros(reps, dtype=int)
        n_singular = 0
        for r in range(reps):
            s, z = draw(rng, n)
            fit = plugin_retention(s, z, n_bins)
            eta_hat[r] = fit.eta
            sigma_hat[r] = math.sqrt(fit.sigma2)
            n_empty[r] = fit.n_empty
            if fit.det_ratio == 0.0:
                n_singular += 1
        half = Z_95 * sigma_hat / math.sqrt(n)
        covered = np.abs(eta_hat - eta) <= half
        row = {
            "n": n,
            "reps": reps,
            "coverage": float(np.mean(covered)),
            "coverage_se": float(np.sqrt(np.mean(covered) * (1 - np.mean(covered)) / reps)),
            "eta_hat_mean": float(np.mean(eta_hat)),
            "eta_hat_sd": float(np.std(eta_hat)),
            "n_times_bias": float(n * (np.mean(eta_hat) - eta)),
            "mean_half_width": float(np.mean(half)),
            "replicates_with_empty_cells": int(np.sum(n_empty > 0)),
            "replicates_singular": n_singular,
            "eta_hat_ge_eta_fraction": float(np.mean(eta_hat >= eta)),
            "interval_below_zero_or_above_one": int(np.sum((eta_hat - half < 0) | (eta_hat + half > 1))),
        }
        if sigma:
            t = (eta_hat - eta) / (sigma_hat / math.sqrt(n))
            row.update(
                {
                    "sigma_over_sqrt_n": sigma / math.sqrt(n),
                    "mean_sigma_hat_over_sqrt_n": float(np.mean(sigma_hat) / math.sqrt(n)),
                    "sigma_hat_relative_rmse": float(np.sqrt(np.mean((sigma_hat / sigma - 1.0) ** 2))),
                    "studentized": _moments(t),
                    "oracle_studentized": _moments((eta_hat - eta) / (sigma / math.sqrt(n))),
                }
            )
        else:
            # singular / degenerate endpoint: record the scale of eta_hat and of the half-width
            row.update(
                {
                    "sqrt_n_eta_hat_minus_eta_mean": float(math.sqrt(n) * (np.mean(eta_hat) - eta)),
                    "n_pow_1_over_d_eta_hat_minus_eta_mean": float(n ** (1.0 / d) * (np.mean(eta_hat) - eta)),
                    "n_eta_hat_sd": float(n * np.std(eta_hat)),
                    "half_width_over_eta_hat_median": float(np.median(half / np.maximum(eta_hat, 1e-300))),
                }
            )
        rows.append(row)
    return rows


def run_coverage(popref: dict, sizes, reps: int) -> dict:
    out: dict = {
        "provenance": provenance("coverage", {"sizes": list(sizes), "reps": reps, "seed_base": SEED_BASE}),
        "laws": {},
    }
    tag = 0
    for name, law in LAWS.items():
        ref = popref["laws"][name]
        tag += 1
        rows = _coverage_rows(
            lambda rng, n, law=law: _draw_hermite(rng, n, law.d, law.labels),
            law.d,
            law.n_bins,
            float(ref["eta_D"]),
            ref.get("sigma"),
            sizes,
            reps,
            tag,
        )
        out["laws"][name] = {"population": {"eta_D": ref["eta_D"], "sigma": ref.get("sigma"), "rank_I_Z": ref["rank_I_Z"]}, "rows": rows}
    ref = popref["laws"][MIXTURE.name]
    tag += 1
    rows = _coverage_rows(
        lambda rng, n: (lambda x: (MIXTURE.scores(x), MIXTURE.labels(x)))(MIXTURE.draw(rng, n)),
        2,
        MIXTURE.n_bins,
        float(ref["eta_D"]),
        float(ref["sigma"]),
        sizes,
        reps,
        tag,
    )
    out["laws"][MIXTURE.name] = {"population": {"eta_D": ref["eta_D"], "sigma": ref["sigma"]}, "rows": rows}
    # library-fitted rule (rebuilt deterministically; the popref cuts are the frozen rule)
    ref = popref["laws"]["hermite_d2_library_rule"]
    rule_like = IntervalLaw("rule", 2, tuple(ref["cuts_in_x"]))
    between = np.asarray(ref["labels_between_cuts"])
    tag += 1
    rows = _coverage_rows(
        lambda rng, n: _draw_hermite(rng, n, 2, lambda x: between[rule_like.labels(x)]),
        2,
        int(ref["K"]),
        float(ref["eta_D"]),
        float(ref["sigma"]),
        sizes,
        reps,
        tag,
    )
    out["laws"]["hermite_d2_library_rule"] = {"population": {"eta_D": ref["eta_D"], "sigma": ref["sigma"]}, "rows": rows}
    # the atomless sigma^2 = 0 arc law: eta_D = 9/25 exactly, sigma = 0
    alpha = _solve_arc_half_width(0.8)
    tag += 1
    rows = _coverage_rows(lambda rng, n: _draw_arc(rng, n, alpha), 2, 4, 0.36, None, sizes, reps, tag)
    out["laws"]["arc_sigma_zero"] = {"population": {"eta_D": 0.36, "sigma": 0.0, "alpha": alpha}, "rows": rows}
    return out


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _dump(name: str, payload: dict) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["selftest", "fixtures", "popref", "coverage", "all"])
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 300, 1000, 3000])
    parser.add_argument("--no-write-fixture", action="store_true")
    args = parser.parse_args(argv)

    if args.mode in ("selftest", "all"):
        report = run_selftest()
        _dump("selftest", report)
        print("selftest:", json.dumps({k: v for k, v in report.items() if k != "provenance"}, indent=1))
        if not report["pass"]:
            return 1
    if args.mode in ("fixtures", "all"):
        report = run_fixtures(write=not args.no_write_fixture)
        _dump("fixtures", report)
        print("fixtures:", json.dumps({k: v for k, v in report.items() if k != "provenance"}, indent=1))
        if not report["pass"]:
            return 1
    popref = None
    if args.mode in ("popref", "all"):
        popref = run_popref()
        _dump("popref", popref)
        print("popref:", json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("eta_D", "sigma", "rank_I_Z", "K", "p")} for k, v in popref["laws"].items()}, indent=1))
    if args.mode in ("coverage", "all"):
        if popref is None:
            popref = json.loads((ARTIFACT_DIR / "popref.json").read_text())
        report = run_coverage(popref, args.sizes, args.reps)
        _dump("coverage", report)
        for name, block in report["laws"].items():
            print(name, "eta_D =", block["population"]["eta_D"])
            for row in block["rows"]:
                print(
                    f"  n={row['n']:5d} cov={row['coverage']:.3f} sd={row['eta_hat_sd']:.5f}"
                    + (f" sig/sqrt n={row['sigma_over_sqrt_n']:.5f} skew={row['studentized']['skew']:+.2f}" if "studentized" in row else f" n^(1/d)(eta_hat-eta)={row['n_pow_1_over_d_eta_hat_minus_eta_mean']:.4f} n*sd={row['n_eta_hat_sd']:.3f}")
                    + f" n*bias={row['n_times_bias']:+.3f}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
