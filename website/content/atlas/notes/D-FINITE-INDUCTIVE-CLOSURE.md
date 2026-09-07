---
entity: D-FINITE-INDUCTIVE-CLOSURE
kind: claim
---
Because a stable terminal state is a Voronoi partition, it compiles: the nearest-centre rule built from the terminal centres and metric reproduces every training label. Duplicate rows inherit the label of the merged atom they belong to.

This is the bridge from asking which labels a fixed sample of observations should get to asking which rule should label an observation not yet seen. It is bookkeeping rather than a second fit: no new optimization runs, and the resulting rule is not an approximation of the partition, it is the partition.

This follows from `D-EXCHANGE-IMPLIES-VORONOI` by a routine argument. It also carries a weakening the theorem statement itself does not: a real solver stops at a positive gain tolerance, not at zero, so the deployable guarantee is self-consistency at that tolerance — the rule reproduces every training label except on rows whose relocation is worth no more than the tolerance the partition was certified at.
