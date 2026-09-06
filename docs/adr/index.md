# Architecture Decision Records

Only decisions that materially constrain future implementation are recorded here.

1. [ADR 0001 — Scores are the core data contract](0001-score-contract.md)
2. [ADR 0002 — Start with a small Python/NumPy implementation](0002-python-numpy-first.md) — superseded
3. [ADR 0003 — Keep optional concerns outside the core](0003-small-core.md)
4. [ADR 0004 — Use a JAX-native core with an array-oriented boundary](0004-jax-first.md) — superseded by ADR 0018
5. [ADR 0005 — Make variables, components, and scores explicit API layers](0005-explicit-representations.md) — superseded
6. [ADR 0006 — Publish generated documentation through GitHub Pages](0006-documentation-site.md) — partially superseded by ADR 0019
7. [ADR 0007 — Evolve the API around generic statistical contracts](0007-generic-api-evolution.md)
8. [ADR 0008 — Expose only the mathematical classifier-posterior bridge](0008-classifier-posterior-bridge.md) — superseded in part
9. [ADR 0009 — Separate finite partitions from quantizers](0009-partition-quantizer-separation.md)
10. [ADR 0010 — Separate sources from score providers](0010-source-provider-separation.md)
11. [ADR 0011 — Keep solver semantics criterion-specific](0011-criterion-specific-semantics.md) — partially superseded by ADR 0014
12. [ADR 0012 — Keep classifier training outside the core](0012-classifier-callback-boundary.md) — partially superseded by ADR 0017
13. [ADR 0013 — Complete the pre-1.0 API with capability-specific boundaries](0013-complete-pre-1-api-boundaries.md) — partially superseded by ADR 0014
14. [ADR 0014 — One exchange engine with criterion-specific objectives and explicit certificates](0014-unified-exchange-and-certificates.md)
15. [ADR 0015 — Efficient-score upper bound and solver initialization](0015-efficient-score-bound-and-initialization.md)
16. [ADR 0016 — Verify partition geometry at the tolerance it was optimized at](0016-tolerance-consistent-geometry-verification.md) — refines ADR 0014
17. [ADR 0017 — Density ratios are a first-class statistical representation](0017-density-ratio-representation.md) — partially supersedes ADR 0012
18. [ADR 0018 — Use explicit JAX and NumPy execution behind one mathematical core](0018-explicit-multi-backend-execution.md) — supersedes ADR 0004
19. [ADR 0019 — Add a React learning portal beside the engineering reference](0019-react-learning-portal.md) — partially supersedes ADR 0006
20. [ADR 0020 — Keep the plain-English development blog in the portal](0020-portal-development-blog.md) — extends ADR 0019
21. [ADR 0021 — Name the score coordinates](0021-named-score-schema.md) — extends ADR 0001
22. [ADR 0022 — Make `ScoreProvider` a protocol, not a closed union](0022-score-provider-protocol.md) — extends ADR 0010
23. [ADR 0023 — Separate the deployable rule from the fit](0023-versioned-quantizer-artifact.md) — extends ADR 0009
24. [ADR 0024 — Exception hierarchy and two-stage fit pipeline](0024-error-hierarchy-and-fit-pipeline.md) — extends ADR 0023
25. [ADR 0025 — Serve the portal at the site root and MkDocs at `/reference/`](0025-portal-at-the-site-root.md) — completes ADR 0019; superseded by ADR 0027
26. [ADR 0026 — One workflow builds and publishes the assembled site](0026-one-workflow-publishes-the-site.md) — completes ADR 0019's deployment authorization, amends ADR 0025
27. [ADR 0027 — A landing page at the root, the documentation at `/docs/`, the portal at `/portal/`](0027-landing-page-at-the-root.md) — supersedes ADR 0025
28. [ADR 0028 — Focus research on user decisions and documentation on executable lessons](0028-focused-research-and-teaching.md)
29. [ADR 0029 — Lessons replace the free-form browser Lab](0029-lessons-replace-the-free-form-lab.md) — extends ADR 0028; supersedes ADR 0019's Lab console and marimo embed
30. [ADR 0030 — A bounded formal-verification track for the finite D chain](0030-formal-verification-track.md) — extends ADR 0028
31. [ADR 0031 — The portal is four surfaces, and its pages are articles](0031-portal-reduced-to-four-surfaces.md) — supersedes in part ADR 0019, 0020 and 0029
32. [ADR 0032 — Two directories for portal figures, one committed and one derived](0032-portal-figure-assets.md) — extends ADR 0019
33. [ADR 0033 — The research section is an atlas generated from the registry](0033-research-atlas.md) — extends ADR 0031 and ADR 0030
