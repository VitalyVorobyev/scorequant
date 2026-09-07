---
entity: OPEN-DS-MARGINS-AT-OPTIMA
kind: claim
---
Along exact global finite profiled optima, the objective value converges to the unrestricted population supremum and cell masses converge to positive limits, so the mass margin comes for free — but the binned nuisance and cross blocks vanish, so the conditioning margin fails for every threshold. The optimizer sheds binned nuisance information by design, so the finite-to-population bridge (`OPEN-DS-FINITE-POP-BRIDGE`) governs margin-certified solutions, which are necessarily suboptimal, and never free global optima.

Results in this vein are scoped tightly: one parameter of interest, one nuisance parameter, equal weights, and a class of score laws whose nuisance component is conditionally centred.

A boundary example makes one of the hypotheses concrete: with too few cells, exact centring forces every feasible labelling's profiled value to exactly zero — a rank effect, not a data problem, and no sample size fixes it. The condition settled on is at least one cell more than the score dimension, which for a single parameter of interest is two more than the nuisance dimension. The library refuses such a configuration up front, citing `CE-DS-MARGINS-RANK-VACUITY-001`, rather than blaming the nuisance parameterization.
