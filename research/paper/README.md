# Paper

A short paper on the D theorem and its limits: `main.tex` and `references.bib`.
Build with `latexmk -pdf main.tex` in this directory; the figure is read from `examples/outputs`.
`python research/paper/check_numbers.py`, run from the repository root, recomputes every fixture number it quotes.
The proofs it cites are in `research/formal`.
