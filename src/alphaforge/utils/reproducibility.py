"""Reproducibility primitives: seeding, content hashing and run manifests.

Every experiment writes a :class:`RunManifest` next to its results. Together with the
config hash and the data hash, it pins down exactly what produced a given equity curve.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "lightgbm",
    "pydantic",
    "ccxt",
    "alphaforge",
)


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed Python, NumPy's legacy global RNG and return a fresh ``Generator``.

    Library code should prefer an explicit ``np.random.Generator`` passed down from here.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds third-party code relying on the global RNG
    os.environ["PYTHONHASHSEED"] = str(seed)
    return np.random.default_rng(seed)


def hash_object(obj: Any, length: int = 16) -> str:
    """Stable short SHA-256 of a JSON-serialisable object (keys sorted)."""
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def hash_dataframe(frame: pd.DataFrame | pd.Series, length: int = 16) -> str:
    """Content hash of a DataFrame/Series including its index and column labels."""
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    if isinstance(frame, pd.DataFrame):
        digest.update("|".join(map(str, frame.columns)).encode("utf-8"))
    return digest.hexdigest()[:length]


def git_commit(cwd: str | Path | None = None) -> str | None:
    """Current git commit hash, or ``None`` outside a repository / without commits."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def git_dirty(cwd: str | Path | None = None) -> bool | None:
    """Whether tracked files differ from ``HEAD`` (``None`` outside a repository).

    A commit hash alone over-claims reproducibility when the run used uncommitted code.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(out.stdout.strip())


def environment_info() -> dict[str, Any]:
    """Python, platform and key package versions."""
    versions: dict[str, str | None] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    if versions.get("alphaforge") is None:  # running from source (PYTHONPATH=src)
        from alphaforge import __version__

        versions["alphaforge"] = f"{__version__} (source)"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }


@dataclass
class RunManifest:
    """Provenance record stored with every experiment run."""

    run_id: str
    experiment: str
    seed: int
    config_hash: str
    data_hash: str | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None  # True: uncommitted changes to tracked files at run time
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    environment: dict[str, Any] = field(default_factory=environment_info)

    def to_json(self, path: str | Path) -> Path:
        """Write the manifest as pretty-printed JSON and return the path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")
        return target


def create_run_dir(root: str | Path, experiment: str) -> Path:
    """Create ``<root>/<UTC timestamp>_<experiment>`` and return it."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in experiment)
    run_dir = Path(root) / f"{stamp}_{safe}"
    suffix = 1
    while run_dir.exists():
        run_dir = Path(root) / f"{stamp}_{safe}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir
