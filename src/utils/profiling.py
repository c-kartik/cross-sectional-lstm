from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import platform
import resource
import time

import matplotlib.pyplot as plt
import pandas as pd


def _rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is bytes on macOS, KiB on Linux.
    if platform.system() == "Darwin":
        return float(ru.ru_maxrss) / (1024.0 * 1024.0)
    return float(ru.ru_maxrss) / 1024.0


@dataclass
class StageRecord:
    stage: str
    elapsed_s: float
    rss_mb_start: float
    rss_mb_end: float


class RunProfiler:
    def __init__(self, run_name: str) -> None:
        self.run_name = run_name
        self.records: list[StageRecord] = []

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        rss0 = _rss_mb()
        yield
        t1 = time.perf_counter()
        rss1 = _rss_mb()
        self.records.append(
            StageRecord(
                stage=name,
                elapsed_s=float(t1 - t0),
                rss_mb_start=float(rss0),
                rss_mb_end=float(rss1),
            )
        )

    def to_frame(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(columns=["stage", "elapsed_s", "rss_mb_start", "rss_mb_end"])
        return pd.DataFrame([r.__dict__ for r in self.records])

    def write_artifacts(self, run_dir: Path) -> tuple[Path, Path]:
        run_dir = Path(run_dir)
        df = self.to_frame()
        csv_path = run_dir / "profiling_summary.csv"
        png_path = run_dir / "profiling_stages.png"
        df.to_csv(csv_path, index=False)

        if not df.empty:
            plt.figure(figsize=(10, 4))
            plt.bar(df["stage"], df["elapsed_s"])
            plt.xticks(rotation=35, ha="right")
            plt.ylabel("Seconds")
            plt.title(f"Profiling - {self.run_name}")
            plt.tight_layout()
            plt.savefig(png_path, dpi=150)
            plt.close()
        return csv_path, png_path
