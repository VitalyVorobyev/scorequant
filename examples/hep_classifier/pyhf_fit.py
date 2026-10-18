"""The walkthrough's categories as a standard pyhf HistFactory fit.

The study (`experiment.py`) records each rule's signal and background yield
templates at the seven energy-scale points. This script builds the model a
physicist would write from them, for the profiled-D_s rule and the
equal-width-logit baseline:

* signal with a free normalisation ``mu``, the parameter of interest;
* background with a free normalisation ``nu``;
* the tau energy scale as a ``histosys`` on both samples, its ±1σ templates
  taken at the grid points one headline delta either side of nominal.

It fits the Asimov dataset at ``mu = 1`` with the energy scale fixed and
floating, and writes ``examples/outputs/hep-pyhf.json``. Run with::

    uv run python -m examples.hep_classifier.pyhf_fit

Agreement with the study. With the energy scale fixed both are the same
binned Poisson likelihood, so they agree to the precision of the Hessian.
With it floating, pyhf learns the energy-scale response from the ±1σ
templates alone: its default histosys interpolation has slope
``(hi - lo) / 2`` at nominal, a central difference, whereas the study fits a
line through all seven points to steady the response of categories that
hold a few hundred events. So pyhf matches the study's central-difference
number, and differs from the published least-squares number by the Monte
Carlo noise in the two shifted templates. pyhf's unit Gaussian constraint on
the histosys parameter, a scale width of one delta, is much looser than the
counts and tightens the result by about a part in a thousand.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyhf

OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
STUDY_PATH = OUTPUTS / "hep-classifier.json"
OUTPUT_PATH = OUTPUTS / "hep-pyhf.json"

#: The two rules the walkthrough's headline compares.
RULES = ("ds_rule", "classifier_logit_equal_width")
#: Energy-scale points used as the histosys ±1σ templates.
TES_DOWN, TES_NOMINAL, TES_UP = 0.95, 1.00, 1.05
#: Finite-difference step of the Hessian, in each parameter's own units. Large
#: enough that rounding in the ~1e6-event log likelihood stays far below the
#: curvature, small enough that the Poisson quartic terms do not show.
HESSIAN_STEP = 1e-2


def build_model(signal: np.ndarray, background: np.ndarray, points: list[float]) -> pyhf.pdf.Model:
    """Build the HistFactory model from yield templates ``[tes point, category]``."""

    def at(templates: np.ndarray, point: float) -> list[float]:
        return templates[int(np.argmin(np.abs(np.asarray(points) - point)))].tolist()

    def sample(name: str, norm: str, templates: np.ndarray) -> dict[str, object]:
        histosys = {"hi_data": at(templates, TES_UP), "lo_data": at(templates, TES_DOWN)}
        return {
            "name": name,
            "data": at(templates, TES_NOMINAL),
            "modifiers": [
                {"name": norm, "type": "normfactor", "data": None},
                {"name": "tes", "type": "histosys", "data": histosys},
            ],
        }

    spec = {
        "channels": [
            {
                "name": "categories",
                "samples": [sample("signal", "mu", signal), sample("background", "nu", background)],
            }
        ]
    }
    return pyhf.Model(spec, poi_name="mu")


def _hessian(
    function: Callable[[np.ndarray], float], point: np.ndarray, free: list[int]
) -> np.ndarray:
    """Central-difference Hessian of a scalar function in the free coordinates."""
    size = len(free)
    hessian = np.zeros((size, size))
    for a, i in enumerate(free):
        for b, j in enumerate(free[: a + 1]):
            total = 0.0
            for si, sj, sign in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
                shifted = point.copy()
                shifted[i] += si * HESSIAN_STEP
                shifted[j] += sj * HESSIAN_STEP
                total += sign * function(shifted)
            hessian[a, b] = hessian[b, a] = total / (4.0 * HESSIAN_STEP**2)
    return hessian


def sigma_mu(model: pyhf.pdf.Model, *, tes_floating: bool) -> float:
    """Fit the Asimov dataset at ``mu = 1`` and return the uncertainty on ``mu``.

    pyhf's scipy optimizer returns no uncertainties (``return_uncertainties``
    is honoured only by minuit), so they come from the same place minuit's
    HESSE takes them: the inverse Hessian of pyhf's own ``-2 log L`` at the
    best fit.
    """
    nominal = model.config.suggested_init()  # mu = 1, nu = 1, tes = 0
    asimov = model.expected_data(nominal)
    fixed = model.config.suggested_fixed()
    tes_index = model.config.par_slice("tes").start
    fixed[tes_index] = not tes_floating
    best = np.asarray(pyhf.infer.mle.fit(asimov, model, fixed_params=fixed), dtype=float)

    def twice_nll(pars: np.ndarray) -> float:
        return float(np.squeeze(pyhf.infer.mle.twice_nll(pars, asimov, model)))

    free = [index for index, is_fixed in enumerate(fixed) if not is_fixed]
    covariance = 2.0 * np.linalg.inv(_hessian(twice_nll, best, free))
    mu = free.index(model.config.poi_index)
    return float(np.sqrt(covariance[mu, mu]))


def run_demo(study_path: Path = STUDY_PATH) -> dict[str, object]:
    """Fit every rule in `RULES` and return the pyhf and study uncertainties side by side."""
    pyhf.set_backend("numpy", "scipy")
    study = json.loads(study_path.read_text(encoding="utf-8"))
    downstream = study["downstream"]["all_events"]
    rules: dict[str, object] = {}
    for key in RULES:
        row = downstream["rules"][key]
        model = build_model(
            np.asarray(row["signal_templates"]),
            np.asarray(row["background_templates"]),
            downstream["tes_points"],
        )
        rules[key] = {
            "label": row["label"],
            "pyhf_sigma_mu_tes_fixed": sigma_mu(model, tes_floating=False),
            "pyhf_sigma_mu": sigma_mu(model, tes_floating=True),
            "study_sigma_mu_tes_fixed": row["sigma_mu_tes_fixed"],
            "study_sigma_mu": row["sigma_mu"],
            "study_sigma_mu_central_difference": row["sigma_mu_central_difference"],
        }
    return {
        "pyhf_version": pyhf.__version__,
        "backend": "numpy",
        "optimizer": "scipy",
        "tes_histosys_points": [TES_DOWN, TES_UP],
        "rules": rules,
    }


def main() -> None:
    """Run the fits, print the table and write the JSON."""
    result = run_demo()
    print(f"{'rule':34} {'tes fixed':>20} {'tes floating':>20}")
    print(f"{'':34} {'pyhf':>9} {'study':>10} {'pyhf':>9} {'study':>10}")
    for row in result["rules"].values():  # type: ignore[union-attr]
        print(
            f"{row['label']:34} {row['pyhf_sigma_mu_tes_fixed']:9.3f} "
            f"{row['study_sigma_mu_tes_fixed']:10.3f} "
            f"{row['pyhf_sigma_mu']:9.3f} {row['study_sigma_mu']:10.3f}"
        )
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
