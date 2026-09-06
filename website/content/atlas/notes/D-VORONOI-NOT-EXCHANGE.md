---
entity: D-VORONOI-NOT-EXCHANGE
kind: claim
---
Exchange stability implies a nearest-centre structure, but the converse is false: a partition can be a fixed point of its own nearest-centre rule and still admit a single relocation with a strictly positive exact gain. The witness is a four-row, one-dimensional, two-cell example (`CE-D-VORONOI-CONVERSE-001`).

The consequence is that a partition looking geometric is not evidence that it is stable, and stability — not the appearance of geometry — is what licenses compiling a partition into a reusable rule. This is why `compile_quantizer()` refuses a partition whose stability certificate is `False`, citing the fixture, and reports the remaining gain so the size of the gap is visible.

The nearest precedent is the analogous statement for sum-of-squared-errors, where Lloyd fixed points are known not to be Hartigan-stable.
