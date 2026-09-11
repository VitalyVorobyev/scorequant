# Independent audit: D compilation at positive tolerance

11 September 2026. Fresh-context auditor; no derivation context inherited. Scope:
`D-COMPILE-TOLERANCE-GUARANTEE`, implementation and boundary assertions; the
established exact-zero theorem was inspected, not re-derived.

## 1. Target statement

The recorded claim says positive-tolerance stopping yields a nearest-centroid
rule reproducing training labels except where relocation gain is at most that
tolerance. Audit its quantifiers and the boundary between an exact theorem and
floating-point diagnostics.

## 2. Criterion and problem level

Full D, finite positive-weight assignment followed by an empirical inductive
rule. A relocation changes one label from the original terminal assignment,
recomputing both affected cell moments. Prediction changes all labels using the
frozen terminal centers and metric. These are different operations.

## 3. Status before attempt

`project_proved`, internal, with no positive-tolerance independent audit or Lean
proof. Exact-zero closure has separate formal evidence; it does not certify
positive tolerance or Python/JAX/NumPy execution.

## 4. Dependencies (rechecked)

D5 requires nonempty cells, positive atom weights, nonsingular retained
information, unrestricted nonempty-preserving moves and zero-tolerance
stability. Its distinct-centroid conclusion uses exact stability and does not
transfer to positive tolerance. D2/D3 price a relocation only if the source
retains positive weight; their denominators explicitly exclude a singleton.
D6 duplicate inheritance preserves moments only when all copies inherit one
atom label. A lower bound on gain cannot itself certify an upper bound on gain:
the latter follows from checking every admissible move against the tolerance.

## 5. Nearest literature

The bounded triangulation is recorded in
`LITERATURE/audits/D-COMPILE-TOLERANCE-GUARANTEE-2026-09-11.md` using the already
inspected sources of the independent D5 audit. No fresh novelty or saturation
claim is made. None licenses the singleton, batch or population extensions.

## 6. Counterexample search

Exact Fraction enumeration of all admissible relocations in two minimized
one-dimensional fixtures found two boundary failures. The singleton fixture
has N=3, K=2, scores (0,1,2), labels (0,1,0), unit weights. Two distinct atoms
are necessary in the nonsingleton cell and one singleton, so N=3 is minimal
for this mechanism. Finite tables are not centered, consistently with the
library contract. The second fixture has scores (-3,-1,1,3), labels (0,1,0,1).
It is centered and retains two nonempty cells after batch prediction. This is
a targeted boundary search, not the default exhaustive d=1,2,3/N<=10 campaign;
that larger search and numerical near-singularity sweep were not run.

## 7. Algebraic reduction

Singleton fixture: both centroids are 1, retained information is 3. The two
admissible moves each give information 9/2, ratio 3/2. At epsilon=1/2 the
assignment is stable, but lowest-index argmin sends the singleton to cell 0.
That move is inadmissible; it has no gain under the nonempty-cell move contract.

Batch fixture: retained information is 4. The maximum admissible one-row
ratio is 3, while simultaneous nearest-centroid prediction gives information
16, ratio 4. Thus log(3)<6/5<log(4): every individual disagreement is tolerated,
but the batch objective change exceeds the same tolerance.

## 8. Result

**Required correction; established core retained.** Positive-tolerance
stability bounds the exact gains of all *admissible individual* moves by
epsilon. It therefore bounds admissible prediction disagreements, each priced
against the original partition. It does not establish distinct centroids,
admissibility of every prediction disagreement, or a batch objective bound.
Compilation is conditional on the separate admissibility guard. Do not
replace the actual gain upper bound with D5's lower bound read backwards.

## 9. Adversarial audit

- Strictness/ties: positive tolerance admits coincident centers; deterministic
  lowest-index argmin is essential and does not imply exact reproduction.
- Singleton/empty cells: singleton disagreement is inadmissible. The existing
  `_voronoi_disagreement_gain` returns infinity and `optimize_d_partition`
  refuses it before result construction. Preserve this guard. GeometryReport
  alone omits singleton moves and cannot establish this precondition.
- Duplicate scores: consistent merged inheritance preserves moments; split
  labels are outside the theorem (`CE-D-UNMERGED-DUPLICATES-001`). New test
  combines consistent split weight with compilation.
- Singular information: the library projects the full-score nullspace first;
  positive definiteness is required in the retained coordinates. The test
  exercises a rank-one embedding into dimension two. No full-dimensional
  pseudodeterminant theorem follows and near-threshold robustness is untested.
- Nuisance singularity: in-bin profiled D is outside this claim and compilation
  is refused; no profiled guarantee is inferred.
- Atomic laws: fixtures are finite atomic laws; no atomless assumption used.
- Hidden compactness: none required for a finite move scan.
- First-order-to-finite jump: actual rank-two gain, not distance or lower bound,
  supplies the upper bound. Batch counterexample prevents a second jump.
- Empirical-to-population jump: no population risk conclusion.
- Score-estimation error: supplied-score surrogate only unless scores are exact.
- New-event extension: the rule is defined, but no held-out loss guarantee.
- Zero weights: no optimization measure and no positive-weight label guarantee;
  prediction still works, including a row outside the retained training span.
- Floating point: the implementation evaluates exact-formula gains numerically;
  its tolerance is a numerical diagnostic, not an interval-arithmetic proof.

## 10. Algorithmic consequence

No solver change needed. Preserve the existing upstream singleton refusal and
actual-gain scan. Update wording of `compile_quantizer`, the disagreement helper,
claim and D6 to make the quantifiers and numerical status explicit.

## 11. Deployability consequence

An accepted stable D result supplies a deterministic score-space rule. Merely
stopping at tolerance does not guarantee that result construction succeeds.
Duplicate consistency and nonempty-preserving prediction disagreements matter.
Neither exact-label reproduction nor nonempty output cells on new data follows.

## 12. Information-loss consequence

No bound on the simultaneous change, global optimality gap, held-out retention,
worst retention eigenvalue or population loss follows from this certificate.
The batch fixture disproves an absolute batch-change bound by epsilon; it does
not purport to be a negative-gain Lloyd example.

## 13. Updated status

Keep `project_proved` for the corrected conditional statement; owner integration
must record the audit and boundary assumptions. No publication, release,
programme closure or formal-proof promotion is authorized by this report.

## 14. Registry patch

Set `audit` to `AUDITS/AUDIT-D-COMPILE-TOLERANCE-001.md`; add both fixture IDs
from item 15 to `boundary_counterexamples`. State the conditional admissible
individual-relocation guarantee in item 8. Explicit assumptions: epsilon>0;
finite positive-weight effective atoms; nonempty cells; positive-definite
retained-coordinate information; all admissible nonempty-preserving moves
have gain <=epsilon; duplicates merged or consistently inherited; all actual
prediction disagreements admissible (otherwise compilation is refused).
Add a warning distinguishing numerical gain evaluation, simultaneous prediction,
zero-weight rows, singular projected directions and population guarantees.
Keep exact-zero closure separate. Parent owns registry integration and reindex.

## 15. Regression artifacts and checks

- `COUNTEREXAMPLES/CE-D-COMPILE-SINGLETON-TIE-001.json`.
- `COUNTEREXAMPLES/CE-D-COMPILE-BATCH-TOLERANCE-001.json`.
- `tests/test_research_claims.py::test_compile_tolerance_boundary_exact_ratios`
  recomputes exact moments, nearest labels and every admissible ratio.
- `tests/test_compile_boundaries.py` pins singleton refusal, accepted batch
  behavior, duplicate inheritance, zero-weight prediction and rank projection.

Targeted command (passed, 5 tests):
`JAX_ENABLE_X64=1 MPLBACKEND=Agg uv run pytest tests/test_compile_boundaries.py tests/test_research_claims.py -k 'compile_tolerance_boundary or compile_duplicate or individual_compile or tolerated_singleton'`.
Full repository handoff, registry/Atlas regeneration and formal trust checks
are parent integration responsibilities, not claimed as run here. No Lean file
was changed. The following targeted Ruff check also passed:
`uv run ruff check tests/test_compile_boundaries.py tests/test_research_claims.py`.

## 16. Closing status line

Step 5a compilation audit: corrected conditional admissible-individual-gain
contract; singleton and simultaneous boundary fixtures added; exact-zero and
positive-tolerance/formal scopes remain separate. Programme stays open pending
owner disposition; no release or publication sign-off inferred.
