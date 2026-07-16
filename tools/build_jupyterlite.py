#!/usr/bin/env python3
"""
Build a static JupyterLite site hosting the Example notebooks.

The notebooks at the repo root are the source of truth and are never modified.
This script copies them into a build tree and patches the copies for Pyodide:

  * text.usetex is disabled -- there is no LaTeX binary in the browser, so
    leaving it on makes every plot raise. mathtext renders the same equations.
  * a piplite cell is injected to install the ext-distill-sim wheel, so
    `import thermo_models` resolves without depending on the working directory.
  * Example2's pickle path is rewritten to load from package data rather than
    a path relative to the repo root.

Usage:  python tools/build_jupyterlite.py [--output _site]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / "_jupyterlite"
CONTENT = BUILD / "content"

WHEEL_NAME = "ext-distill-sim"

# Installed into the kernel before the notebook's own imports run.
BOOTSTRAP_SOURCE = """\
# --- JupyterLite bootstrap (injected by tools/build_jupyterlite.py) ---
# Installs the distillation library into the in-browser Python environment.
# Takes a few seconds on first run while Pyodide fetches numpy/scipy/matplotlib.
%pip install -q ipywidgets ext-distill-sim
"""

# Pyodide has no LaTeX and no Helvetica; mathtext + the bundled DejaVu render
# the same equations with no external dependency.
USETEX_PATTERN = re.compile(
    r'plt\.rcParams\.update\(\s*\{[^}]*"text\.usetex"\s*:\s*True[^}]*\}\s*\)'
)
USETEX_REPLACEMENT = (
    'plt.rcParams.update({"text.usetex": False, "font.family": "sans-serif", '
    '"mathtext.fontset": "dejavusans"})'
)

# Example2 opens this relative to the repo root, which does not exist in the
# browser filesystem. The pickle ships as package data inside the wheel.
PICKLE_PATTERN = re.compile(
    r'open\(\s*["\']src/utils/pickles/([^"\']+)["\']\s*,\s*["\']rb["\']\s*\)'
)
PICKLE_REPLACEMENT = (
    r'importlib.resources.files("utils").joinpath("pickles/\1").open("rb")'
)


def patch_source(source: str) -> tuple[str, list[str]]:
    """Apply the Pyodide fixes to one cell's source. Returns (source, notes)."""
    notes = []

    if USETEX_PATTERN.search(source):
        source = USETEX_PATTERN.sub(USETEX_REPLACEMENT, source)
        notes.append("usetex disabled")

    if PICKLE_PATTERN.search(source):
        source = PICKLE_PATTERN.sub(PICKLE_REPLACEMENT, source)
        if "import importlib.resources" not in source:
            source = "import importlib.resources\n" + source
        notes.append("pickle path -> package data")

    return source, notes


def patch_notebook(path: Path, dest: Path) -> list[str]:
    nb = json.loads(path.read_text())
    notes: list[str] = []

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        patched, cell_notes = patch_source(source)
        if cell_notes:
            cell["source"] = patched.splitlines(keepends=True)
            notes.extend(cell_notes)
        # Stale outputs reference widget state that no longer exists; the
        # student re-runs anyway, and clearing keeps the site small.
        cell["outputs"] = []
        cell["execution_count"] = None

    bootstrap = {
        "cell_type": "code",
        "metadata": {},
        "source": BOOTSTRAP_SOURCE.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }
    nb["cells"].insert(0, bootstrap)
    notes.append("bootstrap cell injected")

    dest.write_text(json.dumps(nb, indent=1))
    return notes


def source_date_epoch() -> str:
    """
    Timestamp to bake into the wheel, taken from the last commit that touched the
    library itself.

    Wheels embed file mtimes, so an unpinned build is not reproducible: two runs
    over identical source produce different bytes and therefore a different
    sha256, under an unchanging filename. piplite records the wheel's hash in
    pypi/all.json and verifies it after download, so a browser holding a cached
    all.json from an earlier deploy fails with "Invalid checksum" against a
    freshly built but semantically identical wheel.

    Pinning SOURCE_DATE_EPOCH to the library's last commit makes the wheel a pure
    function of the source: commits that only touch prose or docs leave the wheel
    byte-identical, so nothing can go stale relative to it.
    """
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%ct", "--", "src", "pyproject.toml"],
            cwd=REPO, check=True, capture_output=True, text=True,
        ).stdout.strip()
        if out:
            return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    # No git (e.g. a source tarball): fall back to a fixed epoch so the build is
    # still deterministic rather than silently reintroducing timestamp drift.
    return "1700000000"


def build_wheel() -> Path:
    dist = REPO / "dist"
    for stale in dist.glob("*.whl"):
        stale.unlink()
    env = {**os.environ, "SOURCE_DATE_EPOCH": source_date_epoch()}
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist)],
        cwd=REPO, check=True, stdout=subprocess.DEVNULL, env=env,
    )
    wheels = list(dist.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one wheel in {dist}, found {wheels}")
    return wheels[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="_site", help="output directory")
    args = parser.parse_args()

    print("building wheel...")
    wheel = build_wheel()
    print(f"  {wheel.name}")

    if BUILD.exists():
        shutil.rmtree(BUILD)
    CONTENT.mkdir(parents=True)

    notebooks = sorted(REPO.glob("Example*.ipynb"))
    if not notebooks:
        raise SystemExit("no Example*.ipynb found at repo root")

    print("patching notebooks...")
    for nb in notebooks:
        notes = patch_notebook(nb, CONTENT / nb.name)
        print(f"  {nb.name}: {', '.join(notes)}")

    # Make the wheel available to piplite offline, so the notebooks do not
    # need to reach PyPI at run time.
    wheels_dir = BUILD / "wheels"
    wheels_dir.mkdir()
    shutil.copy(wheel, wheels_dir / wheel.name)

    (BUILD / "jupyter_lite_config.json").write_text(json.dumps({
        "PipliteAddon": {"piplite_urls": [f"wheels/{wheel.name}"]},
    }, indent=2))

    (BUILD / "jupyter-lite.json").write_text(json.dumps({
        "jupyter-lite-schema-version": 0,
        "jupyter-config-data": {
            "appName": "Distillation Demos",
        },
    }, indent=2))

    print(f"running jupyter lite build -> {args.output}")
    subprocess.run(
        ["jupyter", "lite", "build",
         "--contents", str(CONTENT),
         "--output-dir", str(REPO / args.output)],
        cwd=BUILD, check=True,
    )
    print(f"\ndone. serve with:  python -m http.server -d {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
