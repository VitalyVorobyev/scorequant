---
entity: CE-D-VORONOI-CONVERSE-001
kind: fixture
---
This is the boundary fixture for `D-VORONOI-NOT-EXCHANGE`: a four-row, one-dimensional, two-cell example of a partition that is a fixed point of its own nearest-centre rule while still admitting a single relocation with a strictly positive exact information gain. `compile_quantizer()` refuses to compile a partition whose stability certificate is `False`, citing this fixture, and reports the remaining gain so the size of the violation is visible.
