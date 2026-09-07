"""The single source of truth for the ``/get-started`` portal page.

This is a genuine, standalone Python program: run it top to bottom with
``uv run python website/scripts/get_started_program.py`` and it prints, in
order, every number the ``/get-started`` page shows. Nothing on that page is
retyped from a notebook or hand-copied from a terminal -- the page's prose
quotes this file's own stdout.

``website/scripts/generate_snippets.py`` is the other reader of this file. It
splits the source on the ``# %% cell: <id>`` markers below, executes each cell
in one shared namespace (so later cells see earlier cells' variables, exactly
as running this file straight through would), and captures each cell's stdout
into ``website/src/generated/snippet-outputs.json``. The portal's ``Snippet``
component then renders one cell's code alongside its captured output.

Four cells, in the order the page walks a reader through them: the score table
and the execution it is fitted under, the reusable quantizer, its prediction on
scores it has not seen, and -- for the reader whose rows are the final object --
the finite partition of the same table. The deeper material this page used to
carry (the compile bridge and its refusal, profiled ``D_s``, the scalar dynamic
programme, global certification) lives in the reference documentation, whose
own fences are executed by ``tests/test_docs_snippets.py``.

Every cell that prints a number does so through an explicit format spec
(``f"{x:.4f}"``, never a bare ``repr`` of a float or array) and runs on the
NumPy backend at float64, so the output is bit-reproducible across runs and
across machines.
"""

# %% cell: setup
import numpy as np

import scorequant as sq

rng = np.random.default_rng(21)
execution = sq.ExecutionConfig(backend="numpy", precision="float64", device="cpu")

# N(mu, I_2) at mu0 = 0 has score s(x) = x: these raw observations already
# are the score vectors every later cell works with.
scores = rng.normal(size=(1_200, 2))
weights = np.ones(scores.shape[0])

# %% cell: fit
quantizer = sq.fit_quantizer(
    sq.ScoreSample(scores, weights),
    n_bins=5,
    criterion=sq.DOptimality(),
    config=sq.SoftVoronoiConfig(seed=21, initializer_restarts=4, max_steps=120, record_every=20),
    execution=execution,
)

report = quantizer.train_report
print(f"D-efficiency    {report.geometric_mean_retention:.4f}")
print(f"effective rank  {report.effective_rank}")

# %% cell: predict
new_scores = rng.normal(loc=0.2, size=(300, 2))
labels = quantizer.predict_scores(new_scores)
counts = ", ".join(f"{count}" for count in np.bincount(labels, minlength=5))

print(f"bin counts  [{counts}]")

# %% cell: partition
partition = sq.optimize_partition(
    scores,
    weights=weights,
    n_bins=5,
    criterion=sq.DOptimality(),
    config=sq.DExchangeConfig(seed=21),
    execution=execution,
)

print(f"D-efficiency  {partition.train_report.geometric_mean_retention:.4f}")
