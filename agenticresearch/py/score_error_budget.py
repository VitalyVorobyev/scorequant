"""Score-error budget for a frozen rule (SCORE-ERROR-BUDGET, closure step 3, OP17/OP18).

Instrument for KNOWN_RESULTS/10-oracle.md section O8. A true score ``s`` and a
proxy ``s_hat = s + e`` live on the same atoms; a frozen label map ``Z`` is
given (in the reporting budget it is whatever ``q(s_hat)`` produced; in the
rule-transfer budget it is compared with ``q(s)``). Everything the library
reports is the *proxy* retention

    eta_tilde_D = (det I_tilde_Z / det V_tilde)^(1/d),
    I_tilde_Z = sum_b p_b c_tilde_b c_tilde_b^T,  V_tilde = E[s_hat s_hat^T],

and the quantity of interest is the true retention with the same labels,
``eta_D = (det I_Z / det V)^(1/d)`` (O4, PROXY-TRUE-RETAINED-FI).

Modes
-----
selftest   exact ``fractions.Fraction`` checks of the finite inequalities behind
           B1 (directional reporting budget), B2 (log-det reporting budget: the
           Cauchy-Schwarz core, the rank-one Loewner bound, and the float
           log-det brackets built from them), B3 (pointwise mislabel distance
           and the Loewner label-perturbation sandwich) on the protocol's
           default falsification grid (d = 1, 2, 3; K = 2..4; N <= 8; unequal
           weights; duplicate atoms; singleton cells; near-singular
           information; atoms on cell boundaries) with random rational proxies,
           plus the exact sharpness attainer of the first-order constant.
fixtures   the three exact fixtures: CE-SCORE-ERROR-RHO-MIN-NECESSARY-001
           (fixed epsilon, unbounded reporting ratio as rho_min -> 0),
           CE-SCORE-ERROR-BOUNDARY-ATOM-001 (an atom on a cell boundary: the
           mislabel mass does not vanish with epsilon), and
           CE-AUC-INVARIANT-PROXY-RETENTION-001 (a monotone distortion keeps
           every corresponding rank cut and the true retention, and moves the
           reported one).
door3      the classifier example: the door3 logistic-regression rungs rebuilt
           through the O6 audit instrument's closed forms; epsilon, epsilon_Z,
           epsilon_R, the reporting gap against the B2 brackets, the mislabel
           mass against the B3 margin bound, all by quadrature.
synthetic2d a d = 2 bounded mixture law with two proxies at matched epsilon:
           a mis-specified template (natural) and an error aligned with the
           least-retained direction (adversarial); the reporting gap of each
           against the B2 brackets.

Nothing here is a proof. Artifacts under WORK/artifacts/SCORE-ERROR-BUDGET/.
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

RESEARCH = Path(__file__).resolve().parents[1]
ROOT = RESEARCH.parent
ARTIFACTS = RESEARCH / "WORK" / "artifacts" / "SCORE-ERROR-BUDGET"
FIXTURES = RESEARCH / "COUNTEREXAMPLES"
DATE = "2026-09-08"

Vec = list[Fraction]
Mat = list[list[Fraction]]


# --------------------------------------------------------------------------
# Bookkeeping
# --------------------------------------------------------------------------


def provenance(mode: str, parameters: dict[str, object]) -> dict[str, object]:
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


def write_record(name: str, record: dict[str, object]) -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / f"{name}.json"
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def fr(value: Fraction | int) -> str:
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def fr_vec(v: Vec) -> list[str]:
    return [fr(x) for x in v]


def fr_mat(m: Mat) -> list[list[str]]:
    return [fr_vec(row) for row in m]


class Lcg:
    """Deterministic 64-bit LCG so every run is reproducible without numpy."""

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next(self) -> int:
        self.state = (6364136223846793005 * self.state + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return self.state >> 33

    def randint(self, lo: int, hi: int) -> int:
        return lo + self.next() % (hi - lo + 1)

    def rational(self, span: int, denominator: int) -> Fraction:
        return Fraction(self.randint(-span * denominator, span * denominator), denominator)


# --------------------------------------------------------------------------
# Exact linear algebra (d <= 4)
# --------------------------------------------------------------------------


def zeros(d: int) -> Mat:
    return [[Fraction(0)] * d for _ in range(d)]


def identity(d: int) -> Mat:
    return [[Fraction(int(i == j)) for j in range(d)] for i in range(d)]


def madd(a: Mat, b: Mat) -> Mat:
    return [[x + y for x, y in zip(ra, rb, strict=True)] for ra, rb in zip(a, b, strict=True)]


def msub(a: Mat, b: Mat) -> Mat:
    return [[x - y for x, y in zip(ra, rb, strict=True)] for ra, rb in zip(a, b, strict=True)]


def mscale(a: Mat, k: Fraction) -> Mat:
    return [[k * x for x in row] for row in a]


def outer(u: Vec, v: Vec) -> Mat:
    return [[x * y for y in v] for x in u]


def dot(u: Vec, v: Vec) -> Fraction:
    return sum((x * y for x, y in zip(u, v, strict=True)), Fraction(0))


def matvec(m: Mat, v: Vec) -> Vec:
    return [dot(row, v) for row in m]


def matmul(a: Mat, b: Mat) -> Mat:
    d = len(a)
    return [[sum((a[i][k] * b[k][j] for k in range(d)), Fraction(0)) for j in range(d)] for i in range(d)]


def quad_form(v: Vec, m: Mat) -> Fraction:
    return dot(v, matvec(m, v))


def trace(m: Mat) -> Fraction:
    return sum((m[i][i] for i in range(len(m))), Fraction(0))


def det(m: Mat) -> Fraction:
    d = len(m)
    if d == 1:
        return m[0][0]
    total = Fraction(0)
    for j in range(d):
        minor = [row[:j] + row[j + 1 :] for row in m[1:]]
        total += (-1) ** j * m[0][j] * det(minor)
    return total


def inv(m: Mat) -> Mat:
    d = len(m)
    aug = [list(row) + [Fraction(int(i == j)) for j in range(d)] for i, row in enumerate(m)]
    for col in range(d):
        pivot = next((r for r in range(col, d) if aug[r][col] != 0), None)
        if pivot is None:
            raise ZeroDivisionError("singular matrix")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        p = aug[col][col]
        aug[col] = [x / p for x in aug[col]]
        for r in range(d):
            if r != col and aug[r][col] != 0:
                f = aug[r][col]
                aug[r] = [x - f * y for x, y in zip(aug[r], aug[col], strict=True)]
    return [row[d:] for row in aug]


def is_psd(m: Mat) -> bool:
    """A symmetric matrix is positive semidefinite iff every principal minor is >= 0."""
    d = len(m)
    for size in range(1, d + 1):
        for idx in itertools.combinations(range(d), size):
            sub = [[m[i][j] for j in idx] for i in idx]
            if det(sub) < 0:
                return False
    return True


def pencil_eigs(a: Mat, b: Mat) -> list[float]:
    """Float eigenvalues of the pencil (A, B), i.e. of B^{-1/2} A B^{-1/2} (screening only)."""
    import numpy as np
    from scipy.linalg import eigh

    fa = np.array([[float(x) for x in r] for r in a])
    fb = np.array([[float(x) for x in r] for r in b])
    return sorted(float(x) for x in eigh(fa, fb, eigvals_only=True))


def logdet(m: Mat) -> float:
    return math.log(float(det(m)))


# --------------------------------------------------------------------------
# Cell moments
# --------------------------------------------------------------------------


def moments(atoms: list[Vec], weights: Vec, labels: list[int], n_bins: int) -> dict[str, object]:
    """p_b, m_b, c_b (0/0 := 0), V = E[s s^T], I_Z = sum_b m_b m_b^T / p_b."""
    d = len(atoms[0])
    p = [Fraction(0)] * n_bins
    m = [[Fraction(0)] * d for _ in range(n_bins)]
    v = zeros(d)
    for s, w, z in zip(atoms, weights, labels, strict=True):
        p[z] += w
        m[z] = [a + w * b for a, b in zip(m[z], s, strict=True)]
        v = madd(v, mscale(outer(s, s), w))
    c = [[x / p[b] for x in m[b]] if p[b] > 0 else [Fraction(0)] * d for b in range(n_bins)]
    i_z = zeros(d)
    for b in range(n_bins):
        if p[b] > 0:
            i_z = madd(i_z, mscale(outer(m[b], m[b]), 1 / p[b]))
    return {"p": p, "m": m, "c": c, "V": v, "I_Z": i_z}


def shifted(atoms: list[Vec], errors: list[Vec]) -> list[Vec]:
    return [[a + b for a, b in zip(s, e, strict=True)] for s, e in zip(atoms, errors, strict=True)]


# --------------------------------------------------------------------------
# B1/B2: the reporting budget for one label map
# --------------------------------------------------------------------------


def logdet_brackets(eps: float, d: int) -> tuple[float | None, float]:
    """Explicit brackets for log det(A + Delta) - log det A from the O8.2 derivation.

    Upper: tr(A^{-1} Delta) <= 2 sqrt(d) eps + eps^2 (concavity of log det).
    Lower: the larger of 2 d log(1 - eps) (rank-one Loewner bound at t = eps)
    and tr(A^{-1}Delta) - ||A^{-1/2} Delta A^{-1/2}||_F^2 / (2 (1 - eps)^2)
    with tr >= eps^2 - 2 sqrt(d) eps and ||.||_F <= 2 sqrt(d) eps + eps^2;
    both need eps < 1, else None.
    """
    up = 2 * math.sqrt(d) * eps + eps * eps
    if eps >= 1:
        return None, up
    frob = 2 * math.sqrt(d) * eps + eps * eps
    low_a = 2 * d * math.log1p(-eps)
    low_b = eps * eps - 2 * math.sqrt(d) * eps - frob * frob / (2 * (1 - eps) ** 2)
    return max(low_a, low_b), up


def reporting_budget(
    atoms: list[Vec],
    errors: list[Vec],
    weights: Vec,
    labels: list[int],
    n_bins: int,
    directions: list[Vec] | None = None,
) -> dict[str, object]:
    """Exact quantities and checks of B1/B2 for the label map ``labels``.

    The base pair is the truth (I_Z, V); the proxy pair is (I_tilde_Z, V_tilde).
    Returns the exact scalars and a ``checks`` dict of booleans (all must hold).
    """
    d = len(atoms[0])
    true = moments(atoms, weights, labels, n_bins)
    prox = moments(shifted(atoms, errors), weights, labels, n_bins)
    err = moments(errors, weights, labels, n_bins)
    e_full = err["V"]  # E[e e^T]
    e_z = err["I_Z"]  # E[e_Z e_Z^T] = sum_b p_b e_b e_b^T
    v, i_z = true["V"], true["I_Z"]
    vt, it = prox["V"], prox["I_Z"]
    v_inv = inv(v)
    eps2 = trace(matmul(v_inv, e_full))
    eps_z2 = trace(matmul(v_inv, e_z))
    checks: dict[str, bool] = {}
    checks["jensen_E_Z_below_E_full"] = is_psd(msub(e_full, e_z))
    checks["eps_Z_le_eps"] = eps_z2 <= eps2
    if directions is None:
        directions = [[Fraction(int(i == j)) for j in range(d)] for i in range(d)]
    # B1: directional Cauchy-Schwarz, numerator and denominator
    ok = True
    for a in directions:
        x = quad_form(a, msub(it, i_z)) - quad_form(a, e_z)
        ok &= x * x <= 4 * quad_form(a, i_z) * quad_form(a, e_z)
        y = quad_form(a, msub(vt, v)) - quad_form(a, e_full)
        ok &= y * y <= 4 * quad_form(a, v) * quad_form(a, e_full)
    checks["B1_directional"] = ok
    # denominator trace core: (tr(V^{-1}(V_tilde - V)) - eps^2)^2 <= 4 d eps^2
    t_den = trace(matmul(v_inv, msub(vt, v)))
    checks["B2_denominator_cauchy_schwarz"] = (t_den - eps2) ** 2 <= 4 * d * eps2
    # denominator Loewner: V_tilde >= (1 - t) V - ((1 - t)/t) E_full
    checks["B2_denominator_loewner"] = all(
        is_psd(madd(msub(vt, mscale(v, 1 - t)), mscale(e_full, (1 - t) / t)))
        for t in (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4))
    )
    out: dict[str, object] = {
        "d": d,
        "eps2": eps2,
        "eps_Z2": eps_z2,
        "V": v,
        "V_tilde": vt,
        "I_Z": i_z,
        "I_tilde_Z": it,
        "E_Z": e_z,
        "E_full": e_full,
        "det_V": det(v),
        "det_V_tilde": det(vt),
        "det_I_Z": det(i_z),
        "det_I_tilde_Z": det(it),
        "checks": checks,
    }
    if det(i_z) != 0:
        g = inv(i_z)
        eps_r2 = trace(matmul(g, e_z))
        t_num = trace(matmul(g, msub(it, i_z)))
        checks["B2_numerator_cauchy_schwarz"] = (t_num - eps_r2) ** 2 <= 4 * d * eps_r2
        checks["B2_numerator_loewner"] = all(
            is_psd(madd(msub(it, mscale(i_z, 1 - t)), mscale(e_z, (1 - t) / t)))
            for t in (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4))
        )
        # rho_min bound: eps_R^2 <= eps_Z^2 / rho_min, rho_min = min eig of V^{-1/2} I_Z V^{-1/2}
        rho = pencil_eigs(i_z, v)  # eigenvalues of V^{-1/2} I_Z V^{-1/2}
        rho_min = rho[0]
        checks["B2_eps_R_le_eps_Z_over_sqrt_rho_min"] = float(eps_r2) <= float(eps_z2) / rho_min + 1e-12
        out["rho"] = rho
        out["eps_R2"] = eps_r2
        # float log-det brackets: numerator, denominator, and the combined eta bracket
        eps_r = math.sqrt(float(eps_r2))
        eps = math.sqrt(float(eps2))
        low_n, up_n = logdet_brackets(eps_r, d)
        low_d, up_d = logdet_brackets(eps, d)
        gap_n = logdet(it) - logdet(i_z) if det(it) > 0 else -math.inf
        gap_d = logdet(vt) - logdet(v) if det(vt) > 0 else -math.inf
        tol = 1e-9
        checks["B2_numerator_logdet_upper"] = gap_n <= up_n + tol
        checks["B2_numerator_logdet_lower"] = low_n is None or gap_n >= low_n - tol
        checks["B2_denominator_logdet_upper"] = gap_d <= up_d + tol
        checks["B2_denominator_logdet_lower"] = low_d is None or gap_d >= low_d - tol
        log_ratio = (gap_n - gap_d) / d
        out["log_eta_ratio"] = log_ratio
        out["first_order_budget"] = 2 * (eps_r + eps) / math.sqrt(d)
        # exact first-order term T1 = (2/d)[E e_Z^T I_Z^{-1} c_Z - E e^T V^{-1} s] and the
        # sharper bound |T1| <= (2/d) sqrt(d - tr R) [eps_Z sqrt((1 - rho_min)/rho_min) + sqrt(eps^2 - eps_Z^2)]
        between = sum(
            (true["p"][b] * dot(err["c"][b], matvec(g, true["c"][b])) for b in range(n_bins) if true["p"][b] > 0),
            Fraction(0),
        )
        total = sum(
            (w * dot(e, matvec(v_inv, s)) for s, e, w in zip(atoms, errors, weights, strict=True)),
            Fraction(0),
        )
        t1 = Fraction(2, d) * (between - total)
        tr_r = trace(matmul(v_inv, i_z))
        loss = max(0.0, float(d - tr_r))
        within2 = max(0.0, float(eps2 - eps_z2))
        sharper = (2 / d) * math.sqrt(loss) * (
            math.sqrt(float(eps_z2) * max(0.0, (1 - rho_min) / rho_min)) + math.sqrt(within2)
        )
        checks["B2_first_order_term_sharper_bound"] = abs(float(t1)) <= sharper + 1e-12
        checks["B2_first_order_term_crude_bound"] = abs(float(t1)) <= out["first_order_budget"] + 1e-12
        u_n = 2 * math.sqrt(d) * eps_r + eps_r**2
        u_d = 2 * math.sqrt(d) * eps + eps**2
        rem = log_ratio - float(t1)
        rem_hi = (float(eps_r2) - float(eps2) + (u_d**2 / (2 * (1 - eps) ** 2) if eps < 1 else math.inf)) / d
        rem_lo = (float(eps_r2) - float(eps2) - (u_n**2 / (2 * (1 - eps_r) ** 2) if eps_r < 1 else math.inf)) / d
        in_domain = eps < 1 and eps_r < 1 and det(it) > 0 and det(vt) > 0
        checks["B2_remainder_bracket"] = (not in_domain) or (rem_lo - 1e-9 <= rem <= rem_hi + 1e-9)
        out["T1"] = t1
        out["tr_R"] = tr_r
        out["sharper_first_order_budget"] = sharper
        out["remainder"] = rem
        out["remainder_bracket"] = (rem_lo, rem_hi)
        out["bracket_upper"] = (up_n - low_d) / d if low_d is not None else None
        out["bracket_lower"] = (low_n - up_d) / d if low_n is not None else None
        checks["B2_eta_bracket"] = (
            out["bracket_upper"] is None or log_ratio <= out["bracket_upper"] + tol
        ) and (out["bracket_lower"] is None or log_ratio >= out["bracket_lower"] - tol)
    return out


def reporting_budget_both_ways(
    atoms: list[Vec], errors: list[Vec], weights: Vec, labels: list[int], n_bins: int, directions=None
) -> tuple[dict[str, object], dict[str, object]]:
    """Truth as base, then proxy as base (s = s_hat + (-e)): the deployable direction."""
    forward = reporting_budget(atoms, errors, weights, labels, n_bins, directions)
    neg = [[-x for x in e] for e in errors]
    prox_atoms = shifted(atoms, errors)
    if det(moments(prox_atoms, weights, labels, n_bins)["V"]) == 0:
        return forward, {"checks": {}, "skipped": "singular proxy second moment"}
    backward = reporting_budget(prox_atoms, neg, weights, labels, n_bins, directions)
    return forward, backward


def linear_reduction(atoms: list[Vec], errors: list[Vec], weights: Vec) -> dict[str, object]:
    """Least-squares matrix A* = E[s s_hat^T] E[s_hat s_hat^T]^{-1} and the reduced error A* s_hat - s.

    The reported retention is invariant under s_hat -> A s_hat (D-REPARAM-INVARIANCE), so every
    reporting inequality may be applied to A* s_hat only when A* is nonsingular; eps_aff^2 = tr(V^{-1} E[e_A e_A^T])
    equals d minus the sum of the squared uncentred canonical correlations between s and s_hat.
    """
    d = len(atoms[0])
    prox = shifted(atoms, errors)
    vt = moments(prox, weights, [0] * len(atoms), 1)["V"]
    v = moments(atoms, weights, [0] * len(atoms), 1)["V"]
    c = zeros(d)
    for s, sh, w in zip(atoms, prox, weights, strict=True):
        c = madd(c, mscale(outer(s, sh), w))
    a_star = matmul(c, inv(vt))
    reduced = [[x - y for x, y in zip(matvec(a_star, sh), s, strict=True)] for s, sh in zip(atoms, prox, strict=True)]
    e_a = moments(reduced, weights, [0] * len(atoms), 1)["V"]
    eps_aff2 = trace(matmul(inv(v), e_a))
    # canonical-correlation identity: eps_aff^2 = d - tr(V^{-1} C V_tilde^{-1} C^T)
    ct = [[c[j][i] for j in range(d)] for i in range(d)]
    cc = trace(matmul(matmul(inv(v), c), matmul(inv(vt), ct)))
    return {"A_star": a_star, "errors": reduced, "eps_aff2": eps_aff2, "identity": eps_aff2 == d - cc, "sum_r2": cc, "admissible": det(a_star) != 0}


# --------------------------------------------------------------------------
# B3: rule transfer — pointwise mislabel distance and the label-perturbation sandwich
# --------------------------------------------------------------------------


def nearest_center_labels(atoms: list[Vec], centers: list[Vec], metric: Mat) -> list[int]:
    """argmin_b (s - mu_b)^T G (s - mu_b), ties to the lowest index."""
    out = []
    for s in atoms:
        best, arg = None, 0
        for b, mu in enumerate(centers):
            diff = [x - y for x, y in zip(s, mu, strict=True)]
            val = quad_form(diff, metric)
            if best is None or val < best:
                best, arg = val, b
        out.append(arg)
    return out


def boundary_distance_squared(s: Vec, label: int, centers: list[Vec], metric: Mat) -> Fraction:
    """dist_G(s, boundary of its cell)^2 for a nearest-center rule, exactly rational."""
    mu_b = centers[label]
    diff_b = [x - y for x, y in zip(s, mu_b, strict=True)]
    own = quad_form(diff_b, metric)
    best: Fraction | None = None
    for b, mu in enumerate(centers):
        if b == label:
            continue
        diff = [x - y for x, y in zip(s, mu, strict=True)]
        gap = quad_form(diff, metric) - own  # >= 0 since label is the argmin
        sep = [x - y for x, y in zip(mu_b, mu, strict=True)]
        norm2 = quad_form(sep, metric)
        if norm2 == 0:
            continue
        val = gap * gap / (4 * norm2)
        best = val if best is None else min(best, val)
    return Fraction(0) if best is None else best


def label_perturbation(
    atoms: list[Vec], weights: Vec, labels: list[int], labels_hat: list[int], n_bins: int
) -> dict[str, object]:
    """The Loewner sandwich I_Zhat >= I_Z - Gamma, I_Z >= I_Zhat - Gamma_hat (O8.3)."""
    d = len(atoms[0])
    base = moments(atoms, weights, labels, n_bins)
    hat = moments(atoms, weights, labels_hat, n_bins)
    gamma = zeros(d)
    gamma_hat = zeros(d)
    pi = Fraction(0)
    second_moment_m = Fraction(0)
    for s, w, z, zh in zip(atoms, weights, labels, labels_hat, strict=True):
        if z == zh:
            continue
        pi += w
        second_moment_m += w * dot(s, s)
        c_at_hat = base["c"][zh]  # Z-cell means indexed by the hat label
        ch_at_base = hat["c"][z]  # hat-cell means indexed by the base label
        gamma = madd(gamma, mscale(madd(outer(s, s), outer(c_at_hat, c_at_hat)), 2 * w))
        gamma_hat = madd(gamma_hat, mscale(madd(outer(s, s), outer(ch_at_base, ch_at_base)), 2 * w))
    checks = {
        "sandwich_forward": is_psd(madd(msub(hat["I_Z"], base["I_Z"]), gamma)),
        "sandwich_backward": is_psd(madd(msub(base["I_Z"], hat["I_Z"]), gamma_hat)),
    }
    cmax2 = max(
        [dot(c, c) for c in base["c"]] + [dot(c, c) for c in hat["c"]] + [Fraction(0)]
    )
    trace_budget = 2 * second_moment_m + 2 * pi * cmax2
    checks["trace_corollary"] = abs(trace(base["I_Z"]) - trace(hat["I_Z"])) <= trace_budget
    return {
        "pi": pi,
        "I_Z": base["I_Z"],
        "I_Zhat": hat["I_Z"],
        "Gamma": gamma,
        "Gamma_hat": gamma_hat,
        "trace_budget": trace_budget,
        "checks": checks,
    }


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------


def random_case(rng: Lcg) -> dict[str, object]:
    d = rng.randint(1, 3)
    n = rng.randint(d + 1, 8)
    k = rng.randint(2, min(4, n))
    denominator = 4
    atoms: list[Vec] = []
    for _ in range(n):
        if atoms and rng.randint(0, 5) == 0:
            atoms.append(list(atoms[rng.randint(0, len(atoms) - 1)]))  # duplicate atom
        else:
            atoms.append([rng.rational(3, denominator) for _ in range(d)])
    raw = [Fraction(rng.randint(1, 9), 1) for _ in range(n)]
    total = sum(raw)
    weights = [w / total for w in raw]
    # labels: every cell nonempty, singleton cells common
    labels = list(range(k)) + [rng.randint(0, k - 1) for _ in range(n - k)]
    for i in range(n - 1, 0, -1):
        j = rng.randint(0, i)
        labels[i], labels[j] = labels[j], labels[i]
    scale = [Fraction(1, 8), Fraction(1, 2), Fraction(2)][rng.randint(0, 2)]
    errors = [[scale * rng.rational(2, denominator) for _ in range(d)] for _ in range(n)]
    if rng.randint(0, 3) == 0:  # a constant shift: E[e] != 0 on purpose
        shift = [rng.rational(1, denominator) for _ in range(d)]
        errors = [[x + y for x, y in zip(e, shift, strict=True)] for e in errors]
    directions = [[Fraction(int(i == j)) for j in range(d)] for i in range(d)]
    directions += [[rng.rational(2, denominator) for _ in range(d)] for _ in range(3)]
    directions = [a for a in directions if any(x != 0 for x in a)]
    return {"d": d, "atoms": atoms, "weights": weights, "labels": labels, "K": k, "errors": errors, "directions": directions}


def sharpness_attainer(d: int, kappa: Fraction) -> dict[str, object]:
    """e_i = kappa c_{Z_i}: Cauchy-Schwarz is an equality, det I_tilde_Z = (1 + kappa)^{2d} det I_Z."""
    rng = Lcg(7 + d)
    for _ in range(50):
        case = random_case(rng)
        if case["d"] != d:
            continue
        base = moments(case["atoms"], case["weights"], case["labels"], case["K"])
        if det(base["I_Z"]) == 0:
            continue
        errors = [[kappa * x for x in base["c"][z]] for z in case["labels"]]
        out = reporting_budget(case["atoms"], errors, case["weights"], case["labels"], case["K"])
        ratio = out["det_I_tilde_Z"] / out["det_I_Z"]
        return {
            "d": d,
            "kappa": kappa,
            "det_ratio_numerator": ratio,
            "det_ratio_expected": (1 + kappa) ** (2 * d),
            "identity": ratio == (1 + kappa) ** (2 * d),
            "eps_R2": out["eps_R2"],
            "eps_R2_expected": kappa * kappa * d,
            "eps_R_identity": out["eps_R2"] == kappa * kappa * d,
            "first_order_numerator_bound": 2 * math.sqrt(d) * math.sqrt(float(out["eps_R2"])),
            "actual_numerator_gap": 2 * d * math.log1p(float(kappa)),
        }
    raise RuntimeError("no nonsingular case found")


def stage_selftest(trials: int = 400, seed: int = 20260908) -> dict[str, object]:
    rng = Lcg(seed)
    counts = {"cases": 0, "nonsingular_I_Z": 0, "checks": 0}
    failures: list[dict[str, object]] = []
    while counts["cases"] < trials:
        case = random_case(rng)
        base = moments(case["atoms"], case["weights"], case["labels"], case["K"])
        if det(base["V"]) == 0:
            continue
        counts["cases"] += 1
        fwd, bwd = reporting_budget_both_ways(
            case["atoms"], case["errors"], case["weights"], case["labels"], case["K"], case["directions"]
        )
        if "eps_R2" in fwd:
            counts["nonsingular_I_Z"] += 1
        prox_v = moments(shifted(case["atoms"], case["errors"]), case["weights"], case["labels"], case["K"])["V"]
        if det(prox_v) != 0:
            red = linear_reduction(case["atoms"], case["errors"], case["weights"])
            counts["checks"] += 2
            if not red["identity"]:
                failures.append({"direction": "affine", "check": "canonical_correlation_identity", "case": serialize_case(case)})
            if red["eps_aff2"] > fwd["eps2"]:
                failures.append({"direction": "affine", "check": "eps_aff_le_eps", "case": serialize_case(case)})
            if red["admissible"]:
                aff = reporting_budget(case["atoms"], red["errors"], case["weights"], case["labels"], case["K"], case["directions"])
                for name, ok in aff["checks"].items():
                    counts["checks"] += 1
                    if not ok:
                        failures.append({"direction": "affine_reduced", "check": name, "case": serialize_case(case)})
                if "eps_R2" in aff and "eps_R2" in fwd:
                    counts["checks"] += 1
                    # the reported retention is the same number for s_hat and A* s_hat
                    same = aff["det_I_tilde_Z"] * fwd["det_V_tilde"] == fwd["det_I_tilde_Z"] * aff["det_V_tilde"]
                    if not same:
                        failures.append({"direction": "affine_reduced", "check": "reported_retention_invariant", "case": serialize_case(case)})
        for tag, rec in (("forward", fwd), ("backward", bwd)):
            for name, ok in rec["checks"].items():
                counts["checks"] += 1
                if not ok:
                    failures.append({"direction": tag, "check": name, "case": serialize_case(case)})
        # B3 (ii): a second labelling that differs on a random subset
        labels_hat = list(case["labels"])
        for i in range(len(labels_hat)):
            if rng.randint(0, 2) == 0:
                labels_hat[i] = rng.randint(0, case["K"] - 1)
        pert = label_perturbation(case["atoms"], case["weights"], case["labels"], labels_hat, case["K"])
        for name, ok in pert["checks"].items():
            counts["checks"] += 1
            if not ok:
                failures.append({"direction": "labels", "check": name, "case": serialize_case(case), "labels_hat": labels_hat})
        # B3 (i): nearest-center rule in the V^{-1} metric; mislabelled atoms sit within ||e||_V of the boundary
        metric = inv(base["V"])
        centers = [base["c"][b] for b in range(case["K"])]
        if len({tuple(c) for c in centers}) == case["K"]:
            z = nearest_center_labels(case["atoms"], centers, metric)
            zh = nearest_center_labels(shifted(case["atoms"], case["errors"]), centers, metric)
            for s, e, a, b in zip(case["atoms"], case["errors"], z, zh, strict=True):
                counts["checks"] += 1
                if a != b and boundary_distance_squared(s, a, centers, metric) > quad_form(e, metric):
                    failures.append({"direction": "mislabel", "check": "B3_pointwise_distance", "case": serialize_case(case)})
    # B4: the mixture-score map is Lipschitz in the posterior, coordinatewise and exactly
    b4 = 0
    for _ in range(300):
        m = rng.randint(2, 4)
        theta0 = simplex_point(rng, m)
        prior = simplex_point(rng, m)
        eta = simplex_point(rng, m)
        eta2 = simplex_point(rng, m)
        d_min = min(theta0[b] / prior[b] for b in range(m))
        phi1 = mixture_score_map(eta, theta0, prior)
        phi2 = mixture_score_map(eta2, theta0, prior)
        for a in range(m):
            counts["checks"] += 1
            b4 += 1
            rhs = (
                abs(eta2[a] - eta[a]) / prior[a]
                + sum(theta0[b] / prior[b] * abs(eta2[b] - eta[b]) for b in range(m)) / theta0[a]
            ) / d_min
            if abs(phi2[a] - phi1[a]) > rhs:
                failures.append({"direction": "B4", "check": "mixture_score_lipschitz", "theta0": fr_vec(theta0), "prior": fr_vec(prior), "eta": fr_vec(eta), "eta2": fr_vec(eta2)})
            # range: 0 <= Phi_a <= 1/theta0_a
            if not (0 <= phi1[a] <= 1 / theta0[a]):
                failures.append({"direction": "B4", "check": "mixture_score_range", "theta0": fr_vec(theta0), "prior": fr_vec(prior), "eta": fr_vec(eta)})
    counts["B4_pairs"] = b4
    attainers = [sharpness_attainer(d, Fraction(1, 5)) for d in (1, 2, 3)]
    attainers += [sharpness_attainer(2, Fraction(-1, 5))]
    return {
        "provenance": provenance("selftest", {"trials": trials, "seed": seed}),
        "counts": counts,
        "failures": failures,
        "sharpness_attainers": [
            {k: (fr(v) if isinstance(v, Fraction) else v) for k, v in a.items()} for a in attainers
        ],
    }


def simplex_point(rng: Lcg, m: int) -> Vec:
    raw = [Fraction(rng.randint(1, 12)) for _ in range(m)]
    total = sum(raw)
    return [x / total for x in raw]


def mixture_score_map(eta: Vec, theta0: Vec, prior: Vec) -> Vec:
    """Phi_a(eta) = (eta_a / pi_a) / sum_b theta0_b eta_b / pi_b (CLASSIFIER-MIXTURE-SCORE-FORMULA)."""
    r = [e / p for e, p in zip(eta, prior, strict=True)]
    denom = sum((t * x for t, x in zip(theta0, r, strict=True)), Fraction(0))
    return [x / denom for x in r]


def serialize_case(case: dict[str, object]) -> dict[str, object]:
    return {
        "d": case["d"],
        "K": case["K"],
        "atoms": [fr_vec(s) for s in case["atoms"]],
        "errors": [fr_vec(e) for e in case["errors"]],
        "weights": fr_vec(case["weights"]),
        "labels": case["labels"],
    }


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def rho_min_family(delta: Fraction, kappa: Fraction) -> dict[str, object]:
    """d = 2, E[s] = 0, three cells; the rule retains little in the second coordinate.

    Atoms (1, 1), (1, -1) in cell 0 (weight 1/4 each); (-1, 1) and (-1, -1) in
    cell 1 with weights w1, w2; (-1, delta) alone in cell 2 with weight 1/10;
    w1 + w2 = 2/5 and w1 - w2 = -delta/10 make E[s] = 0. The proxy adds
    (0, kappa) on the singleton cell only.
    """
    w3 = Fraction(1, 10)
    w1 = (Fraction(2, 5) - delta * w3) / 2
    w2 = Fraction(2, 5) - w1
    atoms = [
        [Fraction(1), Fraction(1)],
        [Fraction(1), Fraction(-1)],
        [Fraction(-1), Fraction(1)],
        [Fraction(-1), Fraction(-1)],
        [Fraction(-1), delta],
    ]
    weights = [Fraction(1, 4), Fraction(1, 4), w1, w2, w3]
    labels = [0, 0, 1, 1, 2]
    errors = [[Fraction(0), Fraction(0)]] * 4 + [[Fraction(0), kappa]]
    out = reporting_budget(atoms, errors, weights, labels, 3)
    return {"atoms": atoms, "weights": weights, "labels": labels, "errors": errors, "budget": out}


def fixture_rho_min() -> dict[str, object]:
    kappa = Fraction(1, 2)
    members = []
    for delta in (Fraction(1, 2), Fraction(1, 10), Fraction(1, 100)):
        fam = rho_min_family(delta, kappa)
        b = fam["budget"]
        ratio_sq = (b["det_I_tilde_Z"] / b["det_V_tilde"]) / (b["det_I_Z"] / b["det_V"])
        members.append(
            {
                "delta": fr(delta),
                "weights": fr_vec(fam["weights"]),
                "atoms": [fr_vec(s) for s in fam["atoms"]],
                "eps2": fr(b["eps2"]),
                "eps_Z2": fr(b["eps_Z2"]),
                "eps_R2": fr(b["eps_R2"]),
                "rho": b["rho"],
                "det_ratio_true": fr(b["det_I_Z"] / b["det_V"]),
                "det_ratio_proxy": fr(b["det_I_tilde_Z"] / b["det_V_tilde"]),
                "eta_D_true": math.sqrt(float(b["det_I_Z"] / b["det_V"])),
                "eta_D_proxy": math.sqrt(float(b["det_I_tilde_Z"] / b["det_V_tilde"])),
                "eta_ratio_proxy_over_true": math.sqrt(float(ratio_sq)),
                "eta_ratio_squared_exact": fr(ratio_sq),
                "log_eta_ratio": b["log_eta_ratio"],
                "first_order_budget": b["first_order_budget"],
                "bracket_upper": b["bracket_upper"],
                "I_Z": fr_mat(b["I_Z"]),
                "I_tilde_Z": fr_mat(b["I_tilde_Z"]),
                "V": fr_mat(b["V"]),
                "V_tilde": fr_mat(b["V_tilde"]),
                "checks_all_true": all(b["checks"].values()),
            }
        )
    primary = rho_min_family(Fraction(1, 2), kappa)
    return {
        "id": "CE-SCORE-ERROR-RHO-MIN-NECESSARY-001",
        "criterion": "D",
        "level": "information_accounting",
        "claim_falsified": "The relative/log reporting gap between the reported (proxy) geometric-mean retention and the true retention of the same frozen label map admits a bound that depends only on an upper bound on the Fisher-whitened score error epsilon, uniformly over laws and rules. Refuted: in the family below epsilon^2 = kappa^2 / (10 V_22) = kappa^2/(9+delta^2) stays within [0.026, 0.028] while the proxy-to-true retention ratio grows without bound as the least retained eigenvalue rho_min -> 0 (delta -> 0). The retention-weighted error epsilon_R^2 = E[e_Z^T I_Z^{-1} e_Z] <= epsilon_Z^2 / rho_min of SCORE-ERROR-RETENTION-BUDGET is the right scale.",
        "scores": [fr_vec(s) for s in primary["atoms"]],
        "proxy_scores": [fr_vec(s) for s in shifted(primary["atoms"], primary["errors"])],
        "weights": fr_vec(primary["weights"]),
        "K": 3,
        "labels_before": primary["labels"],
        "labels_after_or_optimum": None,
        "poi_indices": [0, 1],
        "nuisance_indices": [],
        "objective_before": f"true det ratio {fr(primary['budget']['det_I_Z'] / primary['budget']['det_V'])} (eta_D = {math.sqrt(float(primary['budget']['det_I_Z'] / primary['budget']['det_V'])):.6f})",
        "objective_after": f"proxy det ratio {fr(primary['budget']['det_I_tilde_Z'] / primary['budget']['det_V_tilde'])} (eta_tilde_D = {math.sqrt(float(primary['budget']['det_I_tilde_Z'] / primary['budget']['det_V_tilde'])):.6f})",
        "exact_quantities": {"kappa": fr(kappa), "family_by_delta": members},
        "verification": {
            "method": "exact_formula",
            "notes": "All moments in fractions.Fraction; E[s] = 0 exactly in every member; eta_D and the ratio are square roots of exact rationals (d = 2). The B1/B2 inequalities of SCORE-ERROR-RETENTION-BUDGET hold on every member (checks_all_true); only their epsilon-alone reading fails.",
        },
        "source": "SCORE-ERROR-BUDGET packet, py/score_error_budget.py fixtures",
        "audit": "AUDITS/AUDIT-SCORE-ERROR-BUDGET-001.md",
        "date": DATE,
    }


def fixture_boundary_atom() -> dict[str, object]:
    atoms = [[Fraction(-1)], [Fraction(0)], [Fraction(2)]]
    weights = [Fraction(1, 3), Fraction(1, 2), Fraction(1, 6)]
    # threshold rule Z = 1{s > 0}: the atom at 0 is on the boundary of cell 0
    labels = [0, 0, 1]
    results = []
    for kappa in (Fraction(1, 1000), Fraction(1, 10)):
        errors = [[Fraction(0)], [kappa], [Fraction(0)]]
        labels_hat = [int(s[0] + e[0] > 0) for s, e in zip(atoms, errors, strict=True)]
        pert = label_perturbation(atoms, weights, labels, labels_hat, 2)
        base = moments(atoms, weights, labels, 2)
        hat = moments(atoms, weights, labels_hat, 2)
        v = base["V"][0][0]
        eps2 = sum((w * e[0] * e[0] for w, e in zip(weights, errors, strict=True)), Fraction(0)) / v
        results.append(
            {
                "kappa": fr(kappa),
                "eps2": fr(eps2),
                "labels_hat": labels_hat,
                "mislabel_mass": fr(pert["pi"]),
                "eta_true_labels": fr(base["I_Z"][0][0] / v),
                "eta_hat_labels": fr(hat["I_Z"][0][0] / v),
                "sandwich_holds": all(pert["checks"].values()),
                "Gamma": fr_mat(pert["Gamma"]),
            }
        )
    return {
        "id": "CE-SCORE-ERROR-BOUNDARY-ATOM-001",
        "criterion": "D",
        "level": "information_accounting",
        "claim_falsified": "The mislabel mass P(q(s_hat) != q(s)) of a frozen rule tends to zero with the score error epsilon without a margin condition on the law at the cell boundaries. Refuted: an atom of mass 1/2 sits on the boundary of the threshold rule Z = 1{s > 0}; every proxy that lifts it by kappa > 0 relabels it, so pi = 1/2 for every epsilon > 0 and the true retention of the transferred labels jumps from 4/5 to 1/2. The margin function M(t) = P(dist_V(s, boundary) <= t) is >= 1/2 for all t > 0, so the SCORE-ERROR-RULE-TRANSFER bound pi <= inf_t {M(t) + epsilon^2/t^2} is vacuous, as it must be.",
        "scores": [fr_vec(s) for s in atoms],
        "weights": fr_vec(weights),
        "K": 2,
        "labels_before": labels,
        "labels_after_or_optimum": results[0]["labels_hat"],
        "poi_indices": [0],
        "nuisance_indices": [],
        "objective_before": "eta = 4/5 under the rule applied to the truth",
        "objective_after": "eta = 1/2 under the rule applied to the proxy (any kappa > 0)",
        "exact_quantities": {"by_kappa": results, "V": fr(moments(atoms, weights, labels, 2)["V"][0][0])},
        "verification": {
            "method": "exact_formula",
            "notes": "fractions.Fraction throughout; E[s] = 0; the Loewner sandwich of SCORE-ERROR-RULE-TRANSFER holds with Gamma of order one, which is the point: the budget is a function of the mislabel mass, and nothing makes the mass small here.",
        },
        "source": "SCORE-ERROR-BUDGET packet, py/score_error_budget.py fixtures",
        "audit": "AUDITS/AUDIT-SCORE-ERROR-BUDGET-001.md",
        "date": DATE,
    }


def fixture_auc_invariant() -> dict[str, object]:
    atoms = [[Fraction(-2)], [Fraction(-1)], [Fraction(1)], [Fraction(2)]]
    weights = [Fraction(1, 4)] * 4
    labels = [0, 0, 1, 1]  # threshold at 0; the same labels for every threshold in (-1, 1)
    base = moments(atoms, weights, labels, 2)
    v = base["V"][0][0]
    eta_true = base["I_Z"][0][0] / v
    distortions = {
        "compress_within_cells": [[Fraction(-11, 10)], [Fraction(-9, 10)], [Fraction(9, 10)], [Fraction(11, 10)]],
        "stretch_one_atom": [[Fraction(-2)], [Fraction(-1)], [Fraction(1)], [Fraction(100)]],
    }
    rows = []
    for name, prox in distortions.items():
        monotone = all(a[0] < b[0] for a, b in zip(prox[:-1], prox[1:], strict=True))
        assert monotone
        # every threshold rule on the proxy that separates the same atoms gives the same labels
        labels_prox = [int(s[0] > 0) for s in prox]
        pm = moments(prox, weights, labels_prox, 2)
        eta_proxy = pm["I_Z"][0][0] / pm["V"][0][0]
        errors = [[p[0] - s[0]] for p, s in zip(prox, atoms, strict=True)]
        b = reporting_budget(atoms, errors, weights, labels, 2)
        rows.append(
            {
                "distortion": name,
                "proxy_scores": [fr_vec(s) for s in prox],
                "labels_unchanged": labels_prox == labels,
                "eta_true": fr(eta_true),
                "eta_proxy": fr(eta_proxy),
                "eps2": fr(b["eps2"]),
                "eps_R2": fr(b["eps_R2"]),
                "log_eta_ratio": b["log_eta_ratio"],
                "first_order_budget": b["first_order_budget"],
                "bracket_upper": b["bracket_upper"],
                "bracket_lower": b["bracket_lower"],
                "checks_all_true": all(b["checks"].values()),
            }
        )
    return {
        "id": "CE-AUC-INVARIANT-PROXY-RETENTION-001",
        "criterion": "D",
        "level": "information_accounting",
        "claim_falsified": "A proxy score with the same ranking as the true score (hence the same ROC curve and AUC for every threshold) reports the true retention. Refuted: two strictly monotone distortions of a four-atom scalar law leave the labels of every corresponding rank cut and the true retention 9/10 unchanged while the reported (proxy) retention is 100/101 for one and 5105/10006 for the other; ranking quality says nothing about the reported number, the Fisher-whitened score error does (SCORE-ERROR-RETENTION-BUDGET), and it is not observable from a ROC curve.",
        "scores": [fr_vec(s) for s in atoms],
        "weights": fr_vec(weights),
        "K": 2,
        "labels_before": labels,
        "labels_after_or_optimum": None,
        "poi_indices": [0],
        "nuisance_indices": [],
        "objective_before": "true retention 9/10 (V = 5/2, I_Z = 9/4)",
        "objective_after": "reported retention 100/101 (compress) and 5105/10006 (stretch) with the same labels",
        "exact_quantities": {"distortions": rows},
        "verification": {
            "method": "exact_formula",
            "notes": "fractions.Fraction; the distortions are strictly increasing, so the ROC curve of any threshold classifier built on the proxy equals that of the truth; the B1/B2 brackets of SCORE-ERROR-RETENTION-BUDGET hold on both (checks_all_true) - the stretch has eps_R^2 = 4802/9 > 1, outside the lower bracket's domain, and the reported number is indeed far from the truth.",
        },
        "source": "SCORE-ERROR-BUDGET packet, py/score_error_budget.py fixtures",
        "audit": "AUDITS/AUDIT-SCORE-ERROR-BUDGET-001.md",
        "date": DATE,
    }


def stage_fixtures() -> dict[str, object]:
    written = []
    for fixture in (fixture_rho_min(), fixture_boundary_atom(), fixture_auc_invariant()):
        path = FIXTURES / f"{fixture['id']}.json"
        path.write_text(json.dumps(fixture, indent=2) + "\n")
        print(f"wrote {path.relative_to(ROOT)}")
        written.append(fixture["id"])
    return {"provenance": provenance("fixtures", {}), "written": written}


# --------------------------------------------------------------------------
# door3: the classifier example
# --------------------------------------------------------------------------


def _audit_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "audit_o6", RESEARCH / "py" / "audit_score_oracle_retention_uncertainty.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _quad_pieces(fn, breakpoints: list[float], limit: float = 15.0) -> float:
    from scipy.integrate import quad

    edges = [-limit, *sorted(b for b in breakpoints if -limit < b < limit), limit]
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        if hi - lo < 1e-14:
            continue
        value, _ = quad(fn, lo, hi, epsabs=1e-14, epsrel=1e-12, limit=400)
        total += value
    return total


def _true_score_roots(cuts: list[float]) -> list[float]:
    """x with s(x) = t for the door3 exact score: log(phi_s/phi_b) is quadratic in x."""
    from examples import door3_classifier as door3

    f0, f1 = door3.REFERENCE_FRACTIONS
    mu_s, sd_s = door3.SIGNAL_MU, door3.SIGNAL_SIGMA
    mu_b, sd_b = door3.BACKGROUND_MU, door3.BACKGROUND_SIGMA
    # log r(x) = log(sd_b/sd_s) - (x-mu_s)^2/(2 sd_s^2) + (x-mu_b)^2/(2 sd_b^2) = A x^2 + B x + C
    a = -1 / (2 * sd_s**2) + 1 / (2 * sd_b**2)
    b = mu_s / sd_s**2 - mu_b / sd_b**2
    c0 = math.log(sd_b / sd_s) - mu_s**2 / (2 * sd_s**2) + mu_b**2 / (2 * sd_b**2)
    roots: list[float] = []
    for t in cuts:
        # s = (r - 1)/(f0 r + f1) <=> r = (1 + f1 t)/(1 - f0 t)
        if 1 - f0 * t <= 0:
            continue
        r = (1 + f1 * t) / (1 - f0 * t)
        if r <= 0:
            continue
        ell = math.log(r)
        disc = b * b - 4 * a * (c0 - ell)
        if disc >= 0:
            roots.extend([(-b - math.sqrt(disc)) / (2 * a), (-b + math.sqrt(disc)) / (2 * a)])
    return sorted(roots)


def door3_budget(n_per_class: int) -> dict[str, object]:
    import numpy as np

    audit = _audit_module()
    rule = audit.door3_rule(n_per_class)
    geometry = audit.cell_geometry(rule)
    logit = rule["logit"]
    cuts = np.asarray(geometry["cuts_s_hat"])
    order = np.asarray(geometry["cell_order_by_center"])
    n_bins = int(rule["result"].n_bins)
    d = 1

    def label_of_value(v: float) -> int:
        return int(order[int(np.searchsorted(cuts, v))])

    def s_true(x: float) -> float:
        return float(audit.closed_form_exact_score(np.array([x]))[0])

    def s_prox(x: float) -> float:
        return float(audit.closed_form_surrogate(np.array([x]), logit)[0])

    def f(x: float) -> float:
        return float(audit.mixture_density(np.array([x]))[0])

    breakpoints = list(geometry["roots_x"]) + _true_score_roots(list(cuts))
    # moments under the proxy labels Z_hat = q(s_hat(x)) and under the true labels Z = q(s(x))
    def cell_integral(fn, which: str, cell: int) -> float:
        def g(x: float) -> float:
            lab = label_of_value(s_prox(x)) if which == "hat" else label_of_value(s_true(x))
            return fn(x) * f(x) if lab == cell else 0.0

        return _quad_pieces(g, breakpoints)

    v = _quad_pieces(lambda x: s_true(x) ** 2 * f(x), breakpoints)
    vt = _quad_pieces(lambda x: s_prox(x) ** 2 * f(x), breakpoints)
    e_full = _quad_pieces(lambda x: (s_prox(x) - s_true(x)) ** 2 * f(x), breakpoints)
    mean_s = _quad_pieces(lambda x: s_true(x) * f(x), breakpoints)
    mean_e = _quad_pieces(lambda x: (s_prox(x) - s_true(x)) * f(x), breakpoints)
    p_hat = [cell_integral(lambda x: 1.0, "hat", b) for b in range(n_bins)]
    m_hat = [cell_integral(s_true, "hat", b) for b in range(n_bins)]
    mt_hat = [cell_integral(s_prox, "hat", b) for b in range(n_bins)]
    p_true = [cell_integral(lambda x: 1.0, "true", b) for b in range(n_bins)]
    m_true = [cell_integral(s_true, "true", b) for b in range(n_bins)]
    i_z = sum(m * m / p for m, p in zip(m_hat, p_hat, strict=True) if p > 0)
    it = sum(m * m / p for m, p in zip(mt_hat, p_hat, strict=True) if p > 0)
    i_true_labels = sum(m * m / p for m, p in zip(m_true, p_true, strict=True) if p > 0)
    e_b = [(mt - m) / p if p > 0 else 0.0 for mt, m, p in zip(mt_hat, m_hat, p_hat, strict=True)]
    e_z = sum(p * e * e for p, e in zip(p_hat, e_b, strict=True))
    eps2 = e_full / v
    eps_z2 = e_z / v
    eps_r2 = e_z / i_z
    eta = i_z / v
    eta_t = it / vt
    log_ratio = math.log(eta_t) - math.log(eta)
    low_n, up_n = logdet_brackets(math.sqrt(eps_r2), d)
    low_d, up_d = logdet_brackets(math.sqrt(eps2), d)
    # reverse direction: proxy as base
    eps2_rev = e_full / vt
    eps_r2_rev = e_z / it
    low_n_r, up_n_r = logdet_brackets(math.sqrt(eps_r2_rev), d)
    low_d_r, up_d_r = logdet_brackets(math.sqrt(eps2_rev), d)
    # B3: mislabel mass and the margin function in the V metric
    pi = _quad_pieces(
        lambda x: f(x) if label_of_value(s_prox(x)) != label_of_value(s_true(x)) else 0.0, breakpoints
    )
    # affine reduction: a* = E[s s_hat] / E[s_hat^2]; reduced error a* s_hat - s
    cross = _quad_pieces(lambda x: s_true(x) * s_prox(x) * f(x), breakpoints)
    a_star = cross / vt
    e_aff = _quad_pieces(lambda x: (a_star * s_prox(x) - s_true(x)) ** 2 * f(x), breakpoints)
    eps_aff2 = e_aff / v
    e_aff_b = [(a_star * mt - m) / p if p > 0 else 0.0 for mt, m, p in zip(mt_hat, m_hat, p_hat, strict=True)]
    e_aff_z = sum(p * e * e for p, e in zip(p_hat, e_aff_b, strict=True))
    eps_aff_r2 = e_aff_z / i_z
    low_n_a, up_n_a = logdet_brackets(math.sqrt(eps_aff_r2), d)
    low_d_a, up_d_a = logdet_brackets(math.sqrt(eps_aff2), d)
    def first_order(e_b_list, e_full_val, e_z_val):
        between = sum(p * e * (m / p) for p, e, m in zip(p_hat, e_b_list, m_hat, strict=True) if p > 0) / i_z
        return 2 * (between - total_cross), 2 * math.sqrt(max(0.0, 1 - eta)) * (
            math.sqrt(e_z_val / v * max(0.0, (1 - eta) / eta)) + math.sqrt(max(0.0, (e_full_val - e_z_val) / v))
        )

    total_cross = _quad_pieces(lambda x: (s_prox(x) - s_true(x)) * s_true(x) * f(x), breakpoints) / v
    t1_raw, sharper_raw = first_order(e_b, e_full, e_z)
    total_cross = _quad_pieces(lambda x: (a_star * s_prox(x) - s_true(x)) * s_true(x) * f(x), breakpoints) / v
    t1_aff, sharper_aff = first_order(e_aff_b, e_aff, e_aff_z)
    affine = {
        "T1": t1_aff,
        "sharper_first_order_budget": sharper_aff,
        "a_star": a_star,
        "uncentred_correlation_s_shat": cross / math.sqrt(v * vt),
        "eps_aff": math.sqrt(eps_aff2),
        "eps_aff_Z": math.sqrt(e_aff_z / v),
        "eps_aff_R": math.sqrt(eps_aff_r2),
        "first_order_budget": 2 * (math.sqrt(eps_aff_r2) + math.sqrt(eps_aff2)) / math.sqrt(d),
        "bracket_upper": (up_n_a - low_d_a) if low_d_a is not None else None,
        "bracket_lower": (low_n_a - up_d_a) if low_n_a is not None else None,
    }

    def margin_mass(t: float) -> float:
        def g(x: float) -> float:
            s = s_true(x)
            dist = min(abs(s - c) for c in cuts) / math.sqrt(v)
            return f(x) if dist <= t else 0.0

        extra = []
        for c in cuts:
            extra += _true_score_roots([c - t * math.sqrt(v), c + t * math.sqrt(v)])
        return _quad_pieces(g, breakpoints + extra)

    grid = [10 ** (k / 8) for k in range(-32, 9)]
    margin_curve = [(t, margin_mass(t)) for t in grid]
    b3_bound = min(mm + eps2 / (t * t) for t, mm in margin_curve)
    b3_argmin = min(margin_curve, key=lambda tm: tm[1] + eps2 / (tm[0] ** 2))[0]
    # label-perturbation sandwich in d = 1: |I_Z - I_Zhat| <= Gamma-type budget
    c_true_by_label = [m / p if p > 0 else 0.0 for m, p in zip(m_true, p_true, strict=True)]
    c_hat_by_label = [m / p if p > 0 else 0.0 for m, p in zip(m_hat, p_hat, strict=True)]
    gamma = 2 * _quad_pieces(
        lambda x: (s_true(x) ** 2 + c_true_by_label[label_of_value(s_prox(x))] ** 2) * f(x)
        if label_of_value(s_prox(x)) != label_of_value(s_true(x))
        else 0.0,
        breakpoints,
    )
    gamma_hat = 2 * _quad_pieces(
        lambda x: (s_true(x) ** 2 + c_hat_by_label[label_of_value(s_true(x))] ** 2) * f(x)
        if label_of_value(s_prox(x)) != label_of_value(s_true(x))
        else 0.0,
        breakpoints,
    )
    return {
        "n_per_class": n_per_class,
        "logit": list(logit),
        "cuts_s_hat": cuts.tolist(),
        "ladder_reproduction": rule["ladder_reproduction"],
        "E_s": mean_s,
        "E_e": mean_e,
        "v": v,
        "v_tilde": vt,
        "p_hat_labels": p_hat,
        "p_true_labels": p_true,
        "eta_true_of_hat_labels": eta,
        "eta_proxy_reported": eta_t,
        "eta_true_of_true_labels": i_true_labels / v,
        "eps2": eps2,
        "eps": math.sqrt(eps2),
        "eps_Z2": eps_z2,
        "eps_R2": eps_r2,
        "eps_R": math.sqrt(eps_r2),
        "rho_min": eta,
        "log_eta_ratio": log_ratio,
        "T1": t1_raw,
        "sharper_first_order_budget": sharper_raw,
        "information_loss_1_minus_eta": 1 - eta,
        "first_order_budget": 2 * (math.sqrt(eps_r2) + math.sqrt(eps2)) / math.sqrt(d),
        "bracket_forward": {
            "upper": (up_n - low_d) if low_d is not None else None,
            "lower": (low_n - up_d) if low_n is not None else None,
        },
        "bracket_backward_for_minus_log_ratio": {
            "upper": (up_n_r - low_d_r) if low_d_r is not None else None,
            "lower": (low_n_r - up_d_r) if low_n_r is not None else None,
            "eps_tilde": math.sqrt(eps2_rev),
            "eps_R_tilde": math.sqrt(eps_r2_rev),
            "rho_tilde_min_reported": eta_t,
        },
        "affine_reduced": affine,
        "mislabel_mass": pi,
        "b3_margin_bound": b3_bound,
        "b3_argmin_t": b3_argmin,
        "margin_curve": margin_curve,
        "sandwich": {
            "I_Z_true_labels": i_true_labels,
            "I_Z_hat_labels": i_z,
            "Gamma": gamma,
            "Gamma_hat": gamma_hat,
            "forward_holds": i_z >= i_true_labels - gamma - 1e-12,
            "backward_holds": i_true_labels >= i_z - gamma_hat - 1e-12,
            "trace_gap": i_z - i_true_labels,
        },
    }


def stage_door3() -> dict[str, object]:
    sys.path.insert(0, str(ROOT))
    rungs = [door3_budget(n) for n in (15, 60, 300)]
    return {"provenance": provenance("door3", {"rungs": [15, 60, 300]}), "rungs": rungs}


# --------------------------------------------------------------------------
# synthetic2d: a bounded mixture law with a natural and an adversarial proxy
# --------------------------------------------------------------------------


def stage_synthetic2d() -> dict[str, object]:
    import numpy as np
    from scipy.integrate import quad

    comps = [(-1.5, 0.8), (1.0, 0.6), (0.0, 2.5)]
    lam = [0.35, 0.25, 0.40]
    cuts = [-2.0, -0.5, 1.0]

    def pdf(x, mu, sd):
        return math.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))

    def phis(x, c=comps):
        return [pdf(x, mu, sd) for mu, sd in c]

    def f(x):
        return sum(l * p for l, p in zip(lam, phis(x), strict=True))

    def score(x, c=comps):
        ph = phis(x, c)
        fx = sum(l * p for l, p in zip(lam, ph, strict=True))
        return np.array([(ph[0] - ph[2]) / fx, (ph[1] - ph[2]) / fx])

    def label(x):
        return int(np.searchsorted(cuts, x))

    def integrate(fn):
        edges = [-15.0, *cuts, 15.0]
        total = None
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            for i in range(4):
                val, _ = quad(lambda x: fn(x)[i], lo, hi, epsabs=1e-13, epsrel=1e-11, limit=300)
                total = np.zeros(4) if total is None else total
                total[i] += val
        return total

    def mat_moments(sfun):
        # returns V (2x2), cell moments p_b, m_b for 4 cells
        v = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                v[i, j] = quad(lambda x: sfun(x)[i] * sfun(x)[j] * f(x), -15, 15, points=cuts, limit=300, epsabs=1e-13)[0]
        p = np.zeros(4)
        m = np.zeros((4, 2))
        edges = [-15.0, *cuts, 15.0]
        for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            p[b] = quad(lambda x: f(x), lo, hi, epsabs=1e-13)[0]
            for i in range(2):
                m[b, i] = quad(lambda x: sfun(x)[i] * f(x), lo, hi, epsabs=1e-13)[0]
        return v, p, m

    v, p, m = mat_moments(score)
    i_z = sum(np.outer(m[b], m[b]) / p[b] for b in range(4))
    v_inv = np.linalg.inv(v)
    w_inv_sqrt = np.linalg.inv(np.linalg.cholesky(v))  # V^{-1/2} up to rotation
    r = w_inv_sqrt @ i_z @ w_inv_sqrt.T
    rho, vecs = np.linalg.eigh(r)
    u_min_whitened = vecs[:, 0]
    u_min = w_inv_sqrt.T @ u_min_whitened  # direction in score space aligned with the weakest retained whitened axis
    u_min = u_min / math.sqrt(float(u_min @ v_inv @ u_min))  # unit in the V^{-1} metric... scaled below
    eta = math.sqrt(np.linalg.det(i_z) / np.linalg.det(v))

    def evaluate(proxy_fn, name: str, extra: dict[str, object]) -> dict[str, object]:
        vt, _, mt = mat_moments(proxy_fn)
        it = sum(np.outer(mt[b], mt[b]) / p[b] for b in range(4))
        e_full = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                e_full[i, j] = quad(
                    lambda x: (proxy_fn(x) - score(x))[i] * (proxy_fn(x) - score(x))[j] * f(x),
                    -15, 15, points=cuts, limit=300, epsabs=1e-13,
                )[0]
        e_b = (mt - m) / p[:, None]
        e_z = sum(p[b] * np.outer(e_b[b], e_b[b]) for b in range(4))
        eps2 = float(np.trace(v_inv @ e_full))
        eps_z2 = float(np.trace(v_inv @ e_z))
        eps_r2 = float(np.trace(np.linalg.inv(i_z) @ e_z))
        eta_t = math.sqrt(np.linalg.det(it) / np.linalg.det(vt))
        log_ratio = math.log(eta_t) - math.log(eta)
        low_n, up_n = logdet_brackets(math.sqrt(eps_r2), 2)
        low_d, up_d = logdet_brackets(math.sqrt(eps2), 2)
        rt = w_inv_sqrt @ it @ w_inv_sqrt.T

        def first_order(proxy, e_b_mat, e_full_mat, e_z_mat):
            between = sum(p[b] * float(e_b_mat[b] @ np.linalg.inv(i_z) @ (m[b] / p[b])) for b in range(4))
            tot = 0.0
            for i in range(2):
                for j in range(2):
                    tot += v_inv[i, j] * quad(lambda x: (proxy(x) - score(x))[i] * score(x)[j] * f(x), -15, 15, points=cuts, limit=300, epsabs=1e-13)[0]
            t1 = (between - tot)
            loss = max(0.0, 2 - float(np.trace(v_inv @ i_z)))
            ez2 = float(np.trace(v_inv @ e_z_mat))
            ef2 = float(np.trace(v_inv @ e_full_mat))
            sharper = math.sqrt(loss) * (math.sqrt(ez2 * max(0.0, (1 - rho[0]) / rho[0])) + math.sqrt(max(0.0, ef2 - ez2)))
            return t1, sharper

        t1_raw, sharper_raw = first_order(proxy_fn, e_b, e_full, e_z)
        # affine reduction: A* = E[s s_hat^T] V_tilde^{-1}
        cross = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                cross[i, j] = quad(lambda x: score(x)[i] * proxy_fn(x)[j] * f(x), -15, 15, points=cuts, limit=300, epsabs=1e-13)[0]
        a_star = cross @ np.linalg.inv(vt)
        e_aff = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                e_aff[i, j] = quad(
                    lambda x: (a_star @ proxy_fn(x) - score(x))[i] * (a_star @ proxy_fn(x) - score(x))[j] * f(x),
                    -15, 15, points=cuts, limit=300, epsabs=1e-13,
                )[0]
        e_aff_b = (mt @ a_star.T - m) / p[:, None]
        e_aff_z = sum(p[b] * np.outer(e_aff_b[b], e_aff_b[b]) for b in range(4))
        eps_aff2 = float(np.trace(v_inv @ e_aff))
        eps_aff_r2 = float(np.trace(np.linalg.inv(i_z) @ e_aff_z))
        low_n_a, up_n_a = logdet_brackets(math.sqrt(eps_aff_r2), 2)
        low_d_a, up_d_a = logdet_brackets(math.sqrt(eps_aff2), 2)
        t1_aff, sharper_aff = first_order(lambda x: a_star @ proxy_fn(x), e_aff_b, e_aff, e_aff_z)
        affine = {
            "T1": t1_aff,
            "sharper_first_order_budget": sharper_aff,
            "eps_aff": math.sqrt(eps_aff2),
            "eps_aff_R": math.sqrt(eps_aff_r2),
            "canonical_correlations_squared": np.linalg.eigvalsh(w_inv_sqrt @ cross @ np.linalg.inv(vt) @ cross.T @ w_inv_sqrt.T).tolist(),
            "first_order_budget": 2 * (math.sqrt(eps_aff_r2) + math.sqrt(eps_aff2)) / math.sqrt(2),
            "bracket_upper": (up_n_a - low_d_a) / 2 if low_d_a is not None else None,
            "bracket_lower": (low_n_a - up_d_a) / 2 if low_n_a is not None else None,
        }
        return {
            "T1": t1_raw,
            "sharper_first_order_budget": sharper_raw,
            "affine_reduced": affine,
            "proxy": name,
            **extra,
            "eps": math.sqrt(eps2),
            "eps_Z": math.sqrt(eps_z2),
            "eps_R": math.sqrt(eps_r2),
            "eta_D_true": eta,
            "eta_D_reported": eta_t,
            "log_eta_ratio": log_ratio,
            "first_order_budget": 2 * (math.sqrt(eps_r2) + math.sqrt(eps2)) / math.sqrt(2),
            "bracket_upper": (up_n - low_d) / 2 if low_d is not None else None,
            "bracket_lower": (low_n - up_d) / 2 if low_n is not None else None,
            "retention_eigs_true": rho.tolist(),
            "retention_eigs_reported_in_true_whitening": np.linalg.eigvalsh(rt).tolist(),
            "within_bracket": (low_d is None or log_ratio <= (up_n - low_d) / 2 + 1e-9)
            and (low_n is None or log_ratio >= (low_n - up_d) / 2 - 1e-9),
        }

    rows = []
    # natural proxy: mis-specified template (component means shifted by delta)
    for delta in (0.05, 0.1, 0.2, 0.4):
        shifted_comps = [(mu + delta, sd) for mu, sd in comps]
        rows.append(evaluate(lambda x, c=shifted_comps: score(x, c), "template_shift", {"delta": delta}))
    # adversarial proxy at matched eps: e(x) = kappa u_min h(cell), h = +1 on cells 0, 2 and -1 on 1, 3
    for row in list(rows):
        target_eps = row["eps"]

        def adv(x, kappa=target_eps):
            h = 1.0 if label(x) in (0, 2) else -1.0
            return score(x) + kappa * h * u_min

        rows.append(evaluate(adv, "weak_direction_cell_correlated", {"matched_to_delta": row["delta"]}))
    # same magnitude along the strongest retained direction, for contrast
    u_max = w_inv_sqrt.T @ vecs[:, 1]
    u_max = u_max / math.sqrt(float(u_max @ v_inv @ u_max))
    for row in rows[:4]:
        target_eps = row["eps"]

        def adv_max(x, kappa=target_eps):
            h = 1.0 if label(x) in (0, 2) else -1.0
            return score(x) + kappa * h * u_max

        rows.append(evaluate(adv_max, "strong_direction_cell_correlated", {"matched_to_delta": row["delta"]}))
    return {
        "provenance": provenance("synthetic2d", {"components": comps, "fractions": lam, "cuts": cuts}),
        "law": {"eta_D": eta, "retention_eigs": rho.tolist(), "p": p.tolist()},
        "rows": rows,
    }


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["selftest", "fixtures", "door3", "synthetic2d"])
    parser.add_argument("--trials", type=int, default=400)
    args = parser.parse_args()
    if args.mode == "selftest":
        record = stage_selftest(args.trials)
        write_record("selftest", record)
        print(json.dumps(record["counts"]), "failures:", len(record["failures"]))
        for a in record["sharpness_attainers"]:
            print("attainer", a)
        if record["failures"]:
            print(json.dumps(record["failures"][:3], indent=1))
            sys.exit(1)
    elif args.mode == "fixtures":
        record = stage_fixtures()
        write_record("fixtures", record)
    elif args.mode == "door3":
        record = stage_door3()
        write_record("door3", record)
        for r in record["rungs"]:
            print(
                f"rung {r['n_per_class']}: eta_true {r['eta_true_of_hat_labels']:.4f} reported {r['eta_proxy_reported']:.4f} "
                f"eps {r['eps']:.4f} eps_R {r['eps_R']:.4f} log-ratio {r['log_eta_ratio']:.4f} "
                f"first-order {r['first_order_budget']:.4f} bracket {r['bracket_forward']} pi {r['mislabel_mass']:.4f} "
                f"B3 {r['b3_margin_bound']:.4f} | affine: eps {r['affine_reduced']['eps_aff']:.4f} eps_R {r['affine_reduced']['eps_aff_R']:.4f} "
                f"first-order {r['affine_reduced']['first_order_budget']:.4f} bracket [{r['affine_reduced']['bracket_lower']}, {r['affine_reduced']['bracket_upper']}]"
            )
    else:
        record = stage_synthetic2d()
        write_record("synthetic2d", record)
        for r in record["rows"]:
            print(
                f"{r['proxy']:32s} eps {r['eps']:.4f} eps_R {r['eps_R']:.4f} eta {r['eta_D_true']:.4f} -> {r['eta_D_reported']:.4f} "
                f"log-ratio {r['log_eta_ratio']:+.4f} first-order {r['first_order_budget']:.4f} in-bracket {r['within_bracket']} "
                f"| affine eps {r['affine_reduced']['eps_aff']:.4f} eps_R {r['affine_reduced']['eps_aff_R']:.4f} first-order {r['affine_reduced']['first_order_budget']:.4f}"
            )


if __name__ == "__main__":
    main()
