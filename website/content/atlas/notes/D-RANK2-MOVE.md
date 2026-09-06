---
entity: D-RANK2-MOVE
kind: claim
---
Moving one weighted observation from one cell to another changes the binned information matrix by a rank-two update. Because the update has this closed form, every candidate relocation can be scored exactly and cheaply: the solver never has to rebuild and refactorize the information matrix to compare two partitions, and it never accepts a move on the strength of an approximation.

Exchange-method scatter updates of this general kind go back to Späth, and to Friedman and Rubin, and to Scott and Symons. What is new here is the between-cell form with centroid-coupled coefficients, rather than the within-scatter form those methods use — though it remains possible that Späth's exchange routines already contain this same centroid-coupled between-scatter update, which would make this a known result rather than an adaptation of one.
