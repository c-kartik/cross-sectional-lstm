from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def generate_walk_forward_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    report_path = run_dir / "report.html"

    summary = _read_csv(run_dir / "walk_forward_summary.csv")
    by_year = _read_csv(run_dir / "seed_sweep_by_year.csv")
    overall = _read_csv(run_dir / "seed_sweep_overall.csv")
    config_path = run_dir / "config.json"

    equity_imgs = sorted(run_dir.glob("equity_seed*_*.png"))

    drawdown_imgs: list[Path] = []
    equity_csvs = sorted(run_dir.glob("equity_seed*_*.csv"))
    for csv_path in equity_csvs:
        try:
            eq = pd.read_csv(csv_path, index_col=0, parse_dates=True).iloc[:, 0]
        except Exception:
            continue
        dd = eq / eq.cummax() - 1.0
        out_path = run_dir / csv_path.name.replace("equity_", "drawdown_").replace(".csv", ".png")
        plt.figure()
        dd.plot(title=f"Drawdown - {csv_path.stem.replace('equity_', '')}")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        drawdown_imgs.append(out_path)

    extra_imgs: list[Path] = []
    if summary is not None and not summary.empty:
        if {"year", "total_return", "qqq_total_return"} <= set(summary.columns):
            tmp = summary.copy()
            tmp["ret_diff_qqq"] = tmp["total_return"] - tmp["qqq_total_return"]
            by = tmp.groupby("year")["ret_diff_qqq"].mean()
            out_path = run_dir / "avg_ret_diff_vs_qqq_by_year.png"
            plt.figure()
            by.plot(kind="bar", title="Avg return diff vs QQQ (by year)")
            plt.ylabel("Return difference")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            extra_imgs.append(out_path)

        if "avg_weekly_turnover" in summary.columns:
            by = summary.groupby("year")["avg_weekly_turnover"].mean()
            out_path = run_dir / "avg_turnover_by_year.png"
            plt.figure()
            by.plot(kind="bar", title="Avg weekly turnover (by year)")
            plt.ylabel("Avg weekly turnover")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            extra_imgs.append(out_path)

    def table_html(df: pd.DataFrame | None) -> str:
        if df is None or df.empty:
            return "<p class=\"muted\">(no data)</p>"
        return df.to_html(index=False)

    config_html = ""
    if config_path.exists():
        config_html = f"<pre>{config_path.read_text()}</pre>"

    img_html = ""
    if equity_imgs:
        items = "\n".join(
            [f"<div class=\"img\"><img src=\"{p.name}\" alt=\"{p.name}\"></div>" for p in equity_imgs]
        )
        img_html = f"<div class=\"grid\">{items}</div>"

    drawdown_html = ""
    if drawdown_imgs:
        items = "\n".join(
            [f"<div class=\"img\"><img src=\"{p.name}\" alt=\"{p.name}\"></div>" for p in drawdown_imgs]
        )
        drawdown_html = f"<div class=\"grid\">{items}</div>"

    extra_html = ""
    if extra_imgs:
        items = "\n".join(
            [f"<div class=\"img\"><img src=\"{p.name}\" alt=\"{p.name}\"></div>" for p in extra_imgs]
        )
        extra_html = f"<div class=\"grid\">{items}</div>"

    html = f"""
    <html>
    <head>
      <meta charset="utf-8">
      <title>Walk-forward Report</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; }}
        h1, h2 {{ margin-bottom: 8px; }}
        .muted {{ color: #666; }}
        table {{ border-collapse: collapse; width: 100%; margin: 8px 0 16px; }}
        th, td {{ border: 1px solid #ddd; padding: 6px; font-size: 12px; text-align: right; }}
        th {{ background: #f3f3f3; text-align: left; }}
        pre {{ background: #f7f7f7; padding: 12px; overflow: auto; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }}
        .img img {{ width: 100%; height: auto; border: 1px solid #ddd; }}
      </style>
    </head>
    <body>
      <h1>Walk-forward Report</h1>
      <p class="muted">Run directory: {run_dir.name}</p>

      <h2>Overall Summary</h2>
      {table_html(overall)}

      <h2>Summary by Year</h2>
      {table_html(by_year)}

      <h2>Seed/Year Detail</h2>
      {table_html(summary)}

      <h2>Config</h2>
      {config_html or '<p class="muted">(config.json not found)</p>'}

      <h2>Equity Curves</h2>
      {img_html or '<p class="muted">(no equity charts found)</p>'}

      <h2>Drawdowns</h2>
      {drawdown_html or '<p class="muted">(no drawdown charts found)</p>'}

      <h2>Additional Charts</h2>
      {extra_html or '<p class="muted">(no additional charts generated)</p>'}
    </body>
    </html>
    """

    report_path.write_text(html)
    return report_path
