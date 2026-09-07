---
entity: D-LOGDET-GAIN
kind: claim
---
Once the rank-two change to the information matrix (`D-RANK2-MOVE`) is known, the resulting change in the log determinant has a closed form with no matrix refactorization required. Because every candidate move can be scored this way, a search that accepts only exactly positive gains is strictly monotone on a finite set of labellings, so it cannot cycle and is guaranteed to terminate (`D-EXCHANGE-TERMINATES`).

The determinant identity behind this closed form is classical. What is transferred here is its application to the between-cell, centroid-coupled update rather than to the within-cluster scatter update used in classical determinant-clustering methods.
