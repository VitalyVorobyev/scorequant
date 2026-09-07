---
entity: CE-DS-MARGINS-RANK-VACUITY-001
kind: fixture
---
This is the boundary fixture behind `OPEN-DS-MARGINS-AT-OPTIMA`, and behind it the rank ceiling `FI-RANK-CEILING`: with too few cells, exact centring forces every feasible labelling's profiled value to exactly zero — a rank effect, not a data problem, that no sample size can fix. The library refuses such a configuration up front rather than blaming the nuisance parameterization, citing this fixture; the remedy is to raise the cell count above the score dimension, since on an exactly centred sample no sample size helps.
