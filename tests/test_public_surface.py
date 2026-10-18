"""Every supported name and member must have an explicit compatibility classification."""

import dataclasses
import importlib
import inspect
from pathlib import Path

import scorequant as sq

#: Public submodules. Their ``__all__`` names are inventoried as ``module.Name``.
SUBMODULES = ("ratios",)


def _supported_names() -> dict[str, object]:
    names: dict[str, object] = {name: getattr(sq, name) for name in sq.__all__}
    for module_name in SUBMODULES:
        module = importlib.import_module(f"scorequant.{module_name}")
        names.update({f"{module_name}.{name}": getattr(module, name) for name in module.__all__})
    return names


def test_public_stability_inventory_is_complete() -> None:
    guide = (Path(__file__).parents[1] / "docs/api.md").read_text()
    rows = {
        line.split("|")[1].strip().strip("`"): line
        for line in guide.splitlines()
        if line.startswith("| `")
        and any(f"| {level} |" in line for level in ("Stable", "Diagnostic", "Unstable"))
    }
    supported = _supported_names()
    assert set(rows) == set(supported)
    for name, obj in supported.items():
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


def test_every_supported_object_has_exactly_one_route() -> None:
    # No aliases: an object reachable from the top level is not exported again
    # by a submodule, and no two submodules export the same object.
    routes: dict[int, list[str]] = {}
    for name, obj in _supported_names().items():
        routes.setdefault(id(obj), []).append(name)
    assert [names for names in routes.values() if len(names) > 1] == []
