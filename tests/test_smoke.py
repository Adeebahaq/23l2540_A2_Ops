import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKIP = {"venv", ".git", "__pycache__", "node_modules"}


def python_files():
    for p in ROOT.rglob("*.py"):
        if not SKIP.intersection(p.parts):
            yield p


def test_python_files_parse():
    # Syntax check only: no model files or heavy imports needed in CI
    for path in python_files():
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_project_has_entrypoint():
    assert any((ROOT / name).exists() for name in ("app.py", "main.py"))