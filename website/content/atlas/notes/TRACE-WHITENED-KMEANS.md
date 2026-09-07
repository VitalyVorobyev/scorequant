---
entity: TRACE-WHITENED-KMEANS
kind: claim
---
Once the score coordinates are put into Fisher-whitened units, maximizing the retained trace is exactly weighted k-means — the same objective, the same optimum, the same algorithm. This explains why running k-means on whitened scores is a principled answer, and it pins down precisely which question that answer is answering.

This is a corollary of the information-loss decomposition (`FI-LOSS-DECOMPOSITION`) and is explicitly not presented as a theorem in its own right: that normalized trace after Fisher whitening is equivalent to weighted k-means was already established.
