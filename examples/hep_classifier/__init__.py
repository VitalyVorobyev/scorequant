"""HEP classifier showcase: profiled D_s with a tau-energy-scale nuisance.

Door 3 (classifier to density ratios to scores) through profiled \\(D_s\\) on
the FAIR Universe HiggsML public dataset, following the module shape of
`examples.cell_population`: `data.py` loads and validates the committed
fixture, `scores.py` is the classifier-to-score bridge, `experiment.py` runs
the study, and `figures.py` renders the committed figures. See
`docs/usecases/hep/index.md` for the narrative.

``examples/hep_classifier/fixture.py`` is the recorded fixture-build
procedure; it is never run by this package, an example script, or a test.
"""

from .data import HepData, load_fixture, load_provenance
from .experiment import EventTable, Study, cross_fitted_table, run_study
from .scores import SCHEMA

__all__ = [
    "SCHEMA",
    "EventTable",
    "HepData",
    "Study",
    "cross_fitted_table",
    "load_fixture",
    "load_provenance",
    "run_study",
]
