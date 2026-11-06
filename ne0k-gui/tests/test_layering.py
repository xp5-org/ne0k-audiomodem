"""The engine and audio layers must stay importable (and testable) without Qt."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "modemgui"


def test_no_qt_outside_ui():
    bad = []
    for f in ROOT.rglob("*.py"):
        if "ui" in f.relative_to(ROOT).parts[:1]:
            continue
        for n in ast.walk(ast.parse(f.read_text())):
            mods = ([a.name for a in n.names] if isinstance(n, ast.Import)
                    else [n.module or ""] if isinstance(n, ast.ImportFrom) else [])
            if any(m.split(".")[0] in ("PySide6", "PyQt5", "PyQt6", "shiboken6", "pyqtgraph")
                   for m in mods):
                bad.append(f"{f.relative_to(ROOT)}: {mods}")
    assert not bad, bad
