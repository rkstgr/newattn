"""Make `import newattn` work when running a script straight from a clone.

If the package isn't installed (e.g. you cloned the repo and didn't `pip install -e .`),
add the local `src/` to sys.path so the experiment scripts still run.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import newattn  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
