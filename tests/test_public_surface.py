"""Every supported name and member must have an explicit compatibility classification."""

import dataclasses
import inspect
from pathlib import Path

import scorequant as sq


def test_public_stability_inventory_is_complete() -> None:
    guide = (Path(__file__).parents[1] / "docs/api.md").read_text()
    rows = {
        line.split("|")[1].strip().strip("`"): line
        for line in guide.splitlines()
        if line.startswith("| `") and ("| Stable |" in line or "| Diagnostic |" in line)
    }
    assert set(rows) == set(sq.__all__)
    for name in sq.__all__:
        obj = getattr(sq, name)
        assert inspect.getdoc(obj), name
        members = []
        if dataclasses.is_dataclass(obj):
            members.extend(field.name for field in dataclasses.fields(obj))
        if inspect.isclass(obj):
            members.extend(
                key
                for key, value in vars(obj).items()
                if not key.startswith("_")
                and (callable(value) or isinstance(value, (property, classmethod, staticmethod)))
            )
        for member in members:
            assert f"`{member}`" in rows[name], (name, member)
