from __future__ import annotations

from pathlib import Path

from src.utils.profiling import RunProfiler


def test_run_profiler_writes_csv_and_png(tmp_path: Path) -> None:
    p = RunProfiler("unit_profiler")
    with p.stage("phase_a"):
        x = 1 + 1
        assert x == 2
    with p.stage("phase_b"):
        y = 2 + 2
        assert y == 4

    csv_path, png_path = p.write_artifacts(tmp_path)
    assert csv_path.exists()
    assert png_path.exists()
