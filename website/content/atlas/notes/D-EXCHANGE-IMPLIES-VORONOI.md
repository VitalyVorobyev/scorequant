---
entity: D-EXCHANGE-IMPLIES-VORONOI
kind: claim
---
Take a partition in which no single observation can be moved to another cell with a positive exact information gain. Under stated conditions — coincident score rows merged into one weighted atom, a positive-definite binned information matrix, exactly `K` nonempty cells, no move restriction beyond keeping cells nonempty, and a zero gain tolerance — that partition is a strict, self-consistent Voronoi partition in the Mahalanobis metric given by the inverse binned information. Any observation sitting in the wrong cell under that rule would have supplied a strictly positive gain, with an explicit lower bound on how large. Distinct cell centres are derived from stability rather than assumed.

This converts a list of labels into a geometric object: without it, a labelling of a sample says nothing about how a new observation should be labelled; with it, the terminal labelling is exactly reproduced by a nearest-centre rule that can be written down and applied later.

No direct precedent was found in a targeted search. The nearest prior work is Telgarsky and Vattani's analysis of Lloyd versus Hartigan fixed points, which studies the same kind of terminal state for sum-of-squared-errors and reaches the opposite conclusion there. The determinant-clustering literature between Friedman and Rubin's 1967 paper and Späth's exchange routines has not been fully swept, including non-English and pre-digital sources, and a Hartigan-style terminal-geometry result for another determinant criterion would count as directly relevant prior work.
