"""Minimal model registry: versioned artifacts with the metadata needed to reproduce them."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

import race_engineer


def data_fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()).hexdigest()[:12]


def save_model(obj, name: str, root: Path, metadata: dict) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version = f"{stamp}-{metadata.get('data_fingerprint', 'nodata')}"
    path = Path(root) / name / version
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path / "model.joblib")
    meta = {"name": name, "version": version, "code_version": race_engineer.__version__, **metadata}
    (path / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    (Path(root) / name / "LATEST").write_text(version)
    return path


def load_model(name: str, root: Path, version: str | None = None):
    base = Path(root) / name
    version = version or (base / "LATEST").read_text().strip()
    return joblib.load(base / version / "model.joblib"), json.loads((base / version / "meta.json").read_text())
