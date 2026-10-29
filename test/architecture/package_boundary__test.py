"""Architectural tests that keep pm25ml and publish from importing each other."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PM25ML_ROOT = ROOT / "src" / "pm25ml"
PUBLISH_ROOT = ROOT / "src" / "publish"
THIS_FILE = Path(__file__).resolve()


def _python_files(root: Path) -> list[Path]:
    return [path for path in sorted(root.rglob("*.py")) if path.resolve() != THIS_FILE]


def _imported_names(tree: ast.AST) -> list[tuple[int, str]]:
    names: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append((node.lineno, node.module))
        elif isinstance(node, ast.Call):
            argument = _static_import_argument(node)
            if argument is not None:
                names.append((node.lineno, argument))
    return names


def _static_import_argument(node: ast.Call) -> str | None:
    func = node.func
    is_import_module = (
        isinstance(func, ast.Attribute) and func.attr == "import_module" and bool(node.args)
    ) or (
        isinstance(func, ast.Name)
        and func.id in {"__import__", "import_module"}
        and bool(node.args)
    )
    if not is_import_module:
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def _forbidden(imported: str, package: str) -> bool:
    return imported == package or imported.startswith(f"{package}.")


def _violations(root: Path, package: str) -> list[str]:
    found: list[str] = []
    for path in _python_files(root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for lineno, name in _imported_names(tree):
            if _forbidden(name, package):
                relative = path.relative_to(ROOT)
                found.append(f"{relative}:{lineno} imports {name}")
    return found


def test__package_boundary__pm25ml_modules__do_not_import_publish() -> None:
    assert _violations(PM25ML_ROOT, "publish") == []


def test__package_boundary__publish_modules__do_not_import_pm25ml() -> None:
    assert _violations(PUBLISH_ROOT, "pm25ml") == []
