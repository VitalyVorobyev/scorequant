from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "src" / "scorequant"

# Installed as `sitecustomize` so it runs before the interpreter imports anything
# the package pulls in. A meta-path finder that raises is the only reliable way to
# simulate a runtime where JAX and Optax are genuinely absent.
_BLOCKER_SOURCE = """\
import importlib.abc
import sys


class BlockExecutionLibraries(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {"jax", "optax"}:
            raise ModuleNotFoundError(fullname)
        return None


sys.meta_path.insert(0, BlockExecutionLibraries())
"""


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def _package_sources() -> list[Path]:
    # rglob, not glob: the shared numerical kernels live in the `solvers/` package,
    # which is exactly where a backend import would be most tempting to add.
    return sorted(path for path in PACKAGE.rglob("*.py") if "__pycache__" not in path.parts)


def _run_without_execution_libraries(tmp_path: Path, code: str) -> subprocess.CompletedProcess[str]:
    (tmp_path / "sitecustomize.py").write_text(_BLOCKER_SOURCE)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(ROOT / "src")))
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_execution_libraries_stay_behind_adapter() -> None:
    offenders: list[str] = []
    for path in _package_sources():
        if path.name == "_execution.py":
            continue
        imports = _imports(path)
        if any(name == "jax" or name.startswith("jax.") or name == "optax" for name in imports):
            offenders.append(str(path.relative_to(PACKAGE)))
    assert offenders == []


def test_domain_contracts_do_not_import_solver_or_api_layers() -> None:
    domain_modules = ("config.py", "criteria.py", "reports.py", "sources.py")
    forbidden = {"api", "partition", "quantizers", "certify", "visualization"}
    offenders: dict[str, list[str]] = {}
    for name in domain_modules:
        reverse = sorted(_imports(PACKAGE / name) & forbidden)
        if reverse:
            offenders[name] = reverse
    assert offenders == {}


def test_frontend_concerns_never_enter_python_core() -> None:
    forbidden = {"pyodide", "marimo", "react", "docusaurus"}
    offenders: dict[str, list[str]] = {}
    for path in _package_sources():
        matched = sorted(_imports(path) & forbidden)
        if matched:
            offenders[str(path.relative_to(PACKAGE))] = matched
    assert offenders == {}


def test_execution_library_blocker_actually_blocks(tmp_path: Path) -> None:
    # Guards the test below. A blocker that silently fails to install would make
    # the JAX-free import claim pass vacuously on a machine that has JAX.
    completed = _run_without_execution_libraries(tmp_path, "import jax")
    assert completed.returncode != 0, "the blocker did not prevent `import jax`"
    assert "ModuleNotFoundError" in completed.stderr


def test_package_import_does_not_require_jax_or_optax(tmp_path: Path) -> None:
    completed = _run_without_execution_libraries(
        tmp_path,
        "import scorequant; print(scorequant.ExecutionConfig(backend='numpy'))",
    )
    assert completed.returncode == 0, completed.stderr
    assert "backend='numpy'" in completed.stdout


def test_artifact_never_imports_fit_layers() -> None:
    # artifact.py is the deployable, backend-free layer; pulling in any fitting
    # module would let loading or predicting drag the fit machinery along.
    forbidden = {"result", "api", "partition", "quantizers", "solvers", "certify", "visualization"}
    assert sorted(_imports(PACKAGE / "artifact.py") & forbidden) == []


def test_solvers_never_import_orchestration() -> None:
    # The solvers package holds private numerical kernels; none of them may
    # import the orchestration layer that calls them.
    forbidden = {
        f"scorequant.{module}"
        for module in (
            "api",
            "result",
            "artifact",
            "partition",
            "quantizers",
            "information",
            "certify",
        )
    }
    offenders = {
        path.name: sorted(_imports(path) & forbidden) for path in (PACKAGE / "solvers").glob("*.py")
    }
    assert {name: modules for name, modules in offenders.items() if modules} == {}


def test_predict_and_errors_modules_stay_leaf() -> None:
    # _predict.py is the leaf both artifact.py and result.py depend on, and
    # _errors.py is stdlib-only so every module can import it without risking
    # a cycle.
    assert _imports(PACKAGE / "_predict.py") <= {"__future__", "_chunking", "_execution"}
    assert _imports(PACKAGE / "_errors.py") <= {"__future__"}


def test_api_constructs_results_once() -> None:
    # Results are built once, with every field passed to the constructor,
    # rather than assembled and then patched after the fact; the façade no
    # longer reaches into a solver module's global state either.
    assert "object.__setattr__" not in (PACKAGE / "api.py").read_text()
    assert not (PACKAGE / "quantizers.py").exists()


def _refusal_error_call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def test_every_refusal_cites_a_registered_counterexample() -> None:
    # AGENTS.md: "code that refuses a capability names the counterexample
    # forcing the refusal. Keep both in sync with the registry." Every
    # RefusalError call site must cite a counterexample id that actually
    # exists in the registry, and there must be more than zero of them.
    counterexamples = ROOT / "agenticresearch" / "COUNTEREXAMPLES"
    sites: list[tuple[Path, int, str]] = []
    for path in _package_sources():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _refusal_error_call_name(node.func) != "RefusalError":
                continue
            counterexample_id: str | None = None
            if (
                len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                counterexample_id = node.args[1].value
            else:
                for keyword in node.keywords:
                    if (
                        keyword.arg == "counterexample"
                        and isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, str)
                    ):
                        counterexample_id = keyword.value.value
            assert counterexample_id is not None, (
                f"{path}:{node.lineno} RefusalError call has no literal counterexample id"
            )
            assert (counterexamples / f"{counterexample_id}.json").exists(), (
                f"{path}:{node.lineno} cites unregistered counterexample {counterexample_id!r}"
            )
            sites.append((path, node.lineno, counterexample_id))
    assert len(sites) >= 6, f"expected at least six RefusalError sites, found {len(sites)}: {sites}"
