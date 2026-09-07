# Example data

`flowcyt_fixture.npz` contains a small deterministic subset of the public
[FlowCyt benchmark](https://github.com/VIPER-GENEVA/FlowCyt-Classification-Benchmark).
It is used only by the example and tests; it is not included in the ScoreQuant
Python package.

The FlowCyt data, and therefore this derived fixture, are licensed separately
from ScoreQuant under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Copyright belongs to Lorenzo Bini, Fatemeh Nassajian Mojarrad, Margarita
Liarou, Thomas Matthes, and Stéphane Marchand-Maillet. See
`flowcyt_fixture.json` for provenance and sampling details.

`flowcyt_walkthrough.npz` is the fixture-scale table the FlowCyt walkthrough
page runs on: the mixture scores of the reference cells the rule is fitted on,
of the reference cells that calibrate the categories, and of a subsample of the
held-out patients' cells, with patient ids, expert labels and the frozen
classifier's ratios. It is derived from the fixture through
`examples.cell_population` by `website/scripts/generate_showcase.py --force`,
and its sidecar `flowcyt_walkthrough.json` records the run and what the
walkthrough's own chain produces on it. As a derivative of the benchmark it
carries the same CC BY-NC-SA 4.0 terms and attribution; it is not covered by
ScoreQuant's MIT licence.

The complete research workflow does not commit its 600,000-cell bounded sample.
It recreates `flowcyt-results/flowcyt_sample_20000.npz` with deterministic HTTP
range reads from all 180 upstream FCS files. The complete command, sample digest,
protocol, and resulting metrics are documented in the
[FlowCyt use case](../../docs/usecases/flowcyt/index.md).

## `hep_higgsml_fixture.npz`

1,000 row-aligned events from the FAIR Universe HiggsML Uncertainty Challenge public
dataset, at seven committed tau-energy-scale (`tes`) values. It is used only by
`examples/hep_classifier/`; it is not included in the ScoreQuant Python package.

The bytes were fetched from `FAIR-Universe/HEP-Challenge`, a code repository with no
licence file of its own; the dataset is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) under its Zenodo archival
record, DOI [`10.5281/zenodo.15131565`](https://doi.org/10.5281/zenodo.15131565). See
`hep_higgsml_fixture.json` for the complete provenance record -- the upstream commit,
the fetched-sample SHA-256, the `dopostprocess=False` modelling choice, and the
measured composition and weight facts -- and the
[HEP classifier study](../../docs/usecases/hep/index.md) for the full example.

`examples/hep_classifier/fixture.py` is the recorded build procedure; it needs the
network and packages the project does not depend on, so it is never run by the
example, the tests, or CI.
