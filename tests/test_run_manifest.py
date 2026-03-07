from __future__ import annotations

import json
from pathlib import Path

from src.utils.run_manifest import path_fingerprint, write_run_manifest


def test_path_fingerprint_counts_files(tmp_path: Path) -> None:
    d = tmp_path / "inputs"
    d.mkdir(parents=True, exist_ok=True)
    (d / "a.txt").write_text("a")
    (d / "b.txt").write_text("b")
    fp = path_fingerprint([d])
    assert fp["file_count"] == 2
    assert fp["sha256_meta"] is not None


def test_write_run_manifest_creates_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = write_run_manifest(
        run_dir,
        script_name="unit.test",
        run_id="abc123",
        config={"x": 1},
        seeds=[2, 4, 6],
        input_paths=[tmp_path],
        extra={"note": "ok"},
    )
    assert manifest.exists()
    payload = json.loads(manifest.read_text())
    assert payload["run_id"] == "abc123"
    assert payload["script"] == "unit.test"
    assert payload["config"]["x"] == 1
    assert payload["seeds"] == [2, 4, 6]
