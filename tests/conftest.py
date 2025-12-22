import os
import sys
from pathlib import Path

import pytest

from app.core.config import get_settings

# Ensure project root is on the import path for `import app.*`
ROOT = os.path.abspath(os.getcwd())
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def isolate_runtime_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Keep tests isolated from repo/runtime artifacts.

    This prevents test runs from overwriting production/demo files under
    data/ and models/ (e.g. when tests are executed inside shared containers).
    """
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models" / "baseline"
    sqlite_file = tmp_path / "graph_aml_test.db"

    (data_dir / "raw").mkdir(parents=True, exist_ok=True)
    (data_dir / "interim").mkdir(parents=True, exist_ok=True)
    (data_dir / "processed").mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.setenv("SQLITE_URL", f"sqlite+aiosqlite:///{sqlite_file}")
    monkeypatch.setenv("API_AUTH_TOKEN", "")

    try:
        get_settings.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass

    yield

    try:
        get_settings.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass
