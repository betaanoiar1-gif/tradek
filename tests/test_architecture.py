"""Enforceable architecture boundaries (import-linter + AST fallback)."""
import ast
import pathlib
import subprocess
import sys

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "trading_school_ai"

FORBIDDEN = {
    "ai": ["risk", "execution"],
    "research": ["risk", "execution"],
    "risk": ["ai"],
    "backtest": ["execution", "ai"],
    "learning": ["execution"],
    "experiment": ["execution"],
    "features": ["execution", "ai", "risk"],
}


def _imports(pkg: str) -> set[str]:
    found = set()
    for path in (SRC / pkg).rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module)
            elif isinstance(node, ast.ImportFrom) and node.level:
                found.add("." * node.level + (node.module or ""))
            elif isinstance(node, ast.Import):
                found.update(a.name for a in node.names)
        # relative imports like `from ..risk.engine import X`
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level and node.module:
                found.add(node.module)
    return found


@pytest.mark.parametrize("pkg,banned", [(k, v) for k, vs in FORBIDDEN.items() for v in [vs]])
def test_forbidden_imports(pkg, banned):
    mods = _imports(pkg)
    for bad in banned:
        assert not any(m == bad or m.startswith(f"{bad}.") for m in mods), \
            f"{pkg} must not import {bad}: {sorted(mods)}"


def test_backtest_does_not_call_live_apis():
    src = "\n".join(p.read_text() for p in (SRC / "backtest").rglob("*.py"))
    for token in ("httpx", "requests", "urllib.request", "okx.com"):
        assert token not in src


def test_import_linter_contracts():
    try:
        import importlinter  # noqa: F401
    except ImportError:
        pytest.skip("import-linter not installed (pip install 'trading-school-ai[dev]')")
    proc = subprocess.run([sys.executable, "-m", "importlinter.cli", "lint"],
                          cwd=SRC.parents[1], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_no_placeholder_implementations_in_core():
    """Core modules must not contain placeholder markers."""
    core = ["data", "mtf", "features", "strategies", "backtest", "metrics", "fitness",
            "walkforward", "experiment", "learning", "montecarlo", "stress", "memory",
            "risk", "execution", "reporting"]
    offenders = []
    for pkg in core:
        for path in (SRC / pkg).rglob("*.py"):
            text = path.read_text()
            for marker in ("TODO", "TBD", "NotImplemented", "FIXME"):
                if marker in text:
                    offenders.append(f"{path.name}:{marker}")
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    body = [n for n in node.body if not isinstance(n, ast.Expr)]
                    if len(body) == 1 and isinstance(body[0], ast.Pass):
                        offenders.append(f"{path.name}:{node.name} is a bare pass")
    assert not offenders, offenders
