# Historical format-1 quantizer

`rule.sqz` was written by the isolated v0.2.0 checkout at the commit recorded in
`expected.json`, using that checkout's locked uv environment. The JSON records the exact input
scores, historical predictions, NumPy version and artifact SHA-256. The current reader never
regenerates this artifact in tests.

To reproduce intentionally, extract `git archive v0.2.0` into a temporary directory, run
`uv sync --project <checkout> --locked`, then from that directory run
`uv run --project <checkout> python <absolute-path-to-generate.py> <output-directory> <tag-commit>`.
The script prints the historical import path for inspection. ZIP metadata can vary, so a new
archive needs an intentional fixture/hash review even when its numerical contents agree.
