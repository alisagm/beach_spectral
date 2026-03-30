"""
Audit declared vs. actual dependencies.
Requires only stdlib — no extra installs.
Run from project root: python audit_deps.py
"""

import ast
import sys
import tomllib
from pathlib import Path

# ── known import-name → package-name mismatches ──────────────────────────────
IMPORT_TO_PACKAGE = {
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "yaml": "PyYAML",
    "dateutil": "python-dateutil",
    "pyproj": "pyproj",
    "osgeo": "GDAL",
}

def collect_imports(root: Path) -> set[str]:
    """Walk all .py files and collect top-level import names."""
    imports = set()
    for py_file in root.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except SyntaxError:
            print(f"  [skip] syntax error in {py_file}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])
    return imports

def load_declared(pyproject_path: Path) -> set[str]:
    """Load dependency names from pyproject.toml."""
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    raw = data.get("project", {}).get("dependencies", [])
    # strip version specifiers: "numpy>=1.21.0" → "numpy"
    return {r.split(">")[0].split("<")[0].split("=")[0].split("!")[0].strip()
            for r in raw}

def load_installed() -> set[str]:
    """Get all installed package names from the active environment."""
    import importlib.metadata
    return {d.metadata["Name"].lower() for d in importlib.metadata.distributions()}

# ── main ──────────────────────────────────────────────────────────────────────
root = Path("spectral_classifier")
if not root.exists():
    sys.exit("Run from project root (spectral_classifier/ not found)")

print("Scanning imports...\n")
raw_imports = collect_imports(root)

# normalise: apply known mismatches, lowercase
imports = {IMPORT_TO_PACKAGE.get(name, name).lower() for name in raw_imports}

declared = {p.lower() for p in load_declared(Path("pyproject.toml"))}
installed = load_installed()

# add this inside main, before the IGNORE set
local_modules = {p.stem for p in root.rglob("*.py")}
local_modules |= {p.parent.name for p in root.rglob("__init__.py")}

# stdlib and local modules to ignore
IGNORE = {
    "os", "sys", "re", "json", "math", "copy", "time", "typing", "pathlib",
    "collections", "itertools", "functools", "warnings", "logging", "abc",
    "dataclasses", "enum", "io", "struct", "textwrap", "traceback",
    "concurrent", "multiprocessing", "subprocess", "shutil", "tempfile",
    "unittest", "contextlib", "inspect", "importlib", "gc",
    "spectral_classifier",
}

IGNORE |= local_modules

imports -= IGNORE

print("=" * 60)
print("1. IMPORTED in code but NOT declared in pyproject.toml")
print("   (most actionable — these are missing deps)\n")
undeclared = imports - declared
for name in sorted(undeclared):
    status = "✓ installed" if name in installed else "✗ NOT installed (!)"
    print(f"   {name:<30} {status}")

print()
print("=" * 60)
print("2. DECLARED in pyproject.toml but NOT imported in code")
print("   (may be transitive deps — don't remove without checking)\n")
unused_declared = declared - imports
for name in sorted(unused_declared):
    print(f"   {name}")