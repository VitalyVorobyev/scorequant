"""Manual fixture creation: run with the isolated v0.2.0 environment, never in tests."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import scorequant as sq

output = Path(sys.argv[1])
assert sq.__version__ == "0.2.0"
scores = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
inputs = np.array([[-4.0], [-1.5], [0.0], [0.75], [4.0]])
result = sq.fit_quantizer(
    sq.ScoreSample(scores, schema=sq.ScoreSchema(("location",))),
    n_bins=2,
    config=sq.ScalarDPConfig(),
    execution=sq.ExecutionConfig(backend="numpy", precision="float64"),
)
artifact = result.quantizer.save(output / "rule.sqz")
record = {
    "writer_tag": "v0.2.0",
    "writer_commit": sys.argv[2],
    "writer_version": sq.__version__,
    "numpy_version": np.__version__,
    "format_version": 1,
    "training_scores": scores.tolist(),
    "prediction_scores": inputs.tolist(),
    "expected_labels": result.predict_scores(inputs).tolist(),
    "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
}
(output / "expected.json").write_text(json.dumps(record, indent=2) + "\n")
print(sq.__file__)
