from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    return None


def generate_optimizer_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    report_path = run_dir / "report.html"

    eq = _read_csv(run_dir / "equity.csv")
    eq_base = _read_csv(run_dir / "equity_baseline.csv")
    eq_qqq = _read_csv(run_dir / "equity_qqq.csv")
    eq_spy = _read_csv(run_dir / "equity_spy.csv")
    config_path = run_dir / "config.json"
    stats_path = run_dir / "stats.json"
    stats_base_path = run_dir / "stats_baseline.json"
    turnover = _read_csv(run_dir / "turnover.csv")
    turnover_base = _read_csv(run_dir / "turnover_baseline.csv")
    profiling_img = run_dir / "profiling_stages.png"

    # Relative performance chart
    rel_img = ""
    if eq is not None and eq_base is not None:
        s_opt = eq.iloc[:, 0]
        s_base = eq_base.iloc[:, 0]
        common = s_opt.index.intersection(s_base.index)
        if len(common) > 0:
            rel = (s_opt.loc[common] / s_base.loc[common]).rename("opt_vs_base")
            out_path = run_dir / "equity_relative.png"
            plt.figure()
            rel.plot(title="Relative equity (optimizer / baseline)")
            plt.axhline(1.0, color="gray", linewidth=1)
            plt.xlabel("Date")
            plt.ylabel("Relative")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            rel_img = f"<div class=\"img\"><img src=\"{out_path.name}\" alt=\"{out_path.name}\"></div>"

    # Combined equity chart (optimizer, baseline, QQQ, SPY)
    combined_img = ""
    if eq is not None and eq_base is not None and eq_qqq is not None and eq_spy is not None:
        s_opt = eq.iloc[:, 0]
        s_base = eq_base.iloc[:, 0]
        s_qqq = eq_qqq.iloc[:, 0]
        s_spy = eq_spy.iloc[:, 0]
        common = s_opt.index.intersection(s_base.index).intersection(s_qqq.index).intersection(s_spy.index)
        if len(common) > 0:
            out_path = run_dir / "equity_comparison.png"
            plt.figure()
            pd.DataFrame(
                {
                    "optimizer": s_opt.loc[common],
                    "baseline": s_base.loc[common],
                    "qqq": s_qqq.loc[common],
                    "spy": s_spy.loc[common],
                }
            ).plot(title="Equity comparison")
            plt.xlabel("Date")
            plt.ylabel("Equity (start=1.0)")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            combined_img = f"<div class=\"img\"><img src=\"{out_path.name}\" alt=\"{out_path.name}\"></div>"

    # Drawdown charts
    drawdown_img = ""
    if eq is not None:
        s_opt = eq.iloc[:, 0]
        dd = s_opt / s_opt.cummax() - 1.0
        out_path = run_dir / "drawdown_optimizer.png"
        plt.figure()
        dd.plot(title="Drawdown - Optimizer")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        drawdown_img = f"<div class=\"img\"><img src=\"{out_path.name}\" alt=\"{out_path.name}\"></div>"

    drawdown_base_img = ""
    if eq_base is not None:
        s_base = eq_base.iloc[:, 0]
        dd = s_base / s_base.cummax() - 1.0
        out_path = run_dir / "drawdown_baseline.png"
        plt.figure()
        dd.plot(title="Drawdown - Baseline")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        drawdown_base_img = f"<div class=\"img\"><img src=\"{out_path.name}\" alt=\"{out_path.name}\"></div>"

    # Turnover charts
    turnover_img = ""
    if turnover is not None and not turnover.empty:
        s = turnover.iloc[:, 0]
        out_path = run_dir / "turnover_optimizer.png"
        plt.figure()
        s.plot(title="Daily Turnover - Optimizer")
        plt.xlabel("Date")
        plt.ylabel("Turnover")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        turnover_img = f"<div class=\"img\"><img src=\"{out_path.name}\" alt=\"{out_path.name}\"></div>"

    turnover_base_img = ""
    if turnover_base is not None and not turnover_base.empty:
        s = turnover_base.iloc[:, 0]
        out_path = run_dir / "turnover_baseline.png"
        plt.figure()
        s.plot(title="Daily Turnover - Baseline")
        plt.xlabel("Date")
        plt.ylabel("Turnover")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        turnover_base_img = f"<div class=\"img\"><img src=\"{out_path.name}\" alt=\"{out_path.name}\"></div>"

    config_html = ""
    if config_path.exists():
        config_html = f"<pre>{config_path.read_text()}</pre>"

    stats_html = ""
    if stats_path.exists():
        try:
            stats = pd.read_json(stats_path, typ="series").to_frame(name="optimizer").reset_index()
            stats.columns = ["metric", "optimizer"]
            stats_html = stats.to_html(index=False)
        except Exception:
            stats_html = f"<pre>{stats_path.read_text()}</pre>"

    stats_base_html = ""
    if stats_base_path.exists():
        try:
            stats_base = pd.read_json(stats_base_path, typ="series").to_frame(name="baseline").reset_index()
            stats_base.columns = ["metric", "baseline"]
            stats_base_html = stats_base.to_html(index=False)
        except Exception:
            stats_base_html = f"<pre>{stats_base_path.read_text()}</pre>"

    combined_stats_html = ""
    if stats_html and stats_base_html:
        try:
            opt = pd.read_json(stats_path, typ="series")
            base = pd.read_json(stats_base_path, typ="series")
            combo = pd.concat([opt.rename("optimizer"), base.rename("baseline")], axis=1).reset_index()
            combo.columns = ["metric", "optimizer", "baseline"]
            combined_stats_html = combo.to_html(index=False)
        except Exception:
            combined_stats_html = ""

    html = f"""
    <html>
    <head>
      <meta charset="utf-8">
      <title>Optimizer Report</title>
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
        .banner {{
          background: #fff3cd;
          border: 1px solid #ffe08a;
          color: #664d03;
          padding: 10px 12px;
          margin: 8px 0 16px;
          font-weight: 600;
        }}
      </style>
    </head>
    <body>
      <h1>Optimizer Report</h1>
      <p class="muted">Run directory: {run_dir.name}</p>
      <div class="banner">Experimental R&D output. Baseline Top-N remains the headline strategy.</div>

      <h2>Config</h2>
      {config_html or '<p class="muted">(config.json not found)</p>'}

      <h2>Equity Curves</h2>
      <div class="grid">
        {f'<div class="img"><img src="equity.png" alt="equity.png"></div>' if (run_dir / "equity.png").exists() else ''}
        {f'<div class="img"><img src="equity_baseline.png" alt="equity_baseline.png"></div>' if (run_dir / "equity_baseline.png").exists() else ''}
      </div>

      <h2>Equity Comparison (Optimizer vs Baseline vs QQQ vs SPY)</h2>
      <div class="grid">
        {combined_img or '<p class="muted">(comparison chart not available)</p>'}
      </div>

      <h2>Stats</h2>
      {combined_stats_html or stats_html or '<p class="muted">(no stats found)</p>'}

      <h2>Drawdowns</h2>
      <div class="grid">
        {drawdown_img or '<p class="muted">(optimizer drawdown not available)</p>'}
        {drawdown_base_img or '<p class="muted">(baseline drawdown not available)</p>'}
      </div>

      <h2>Turnover</h2>
      <div class="grid">
        {turnover_img or '<p class="muted">(optimizer turnover not available)</p>'}
        {turnover_base_img or '<p class="muted">(baseline turnover not available)</p>'}
      </div>

      <h2>Relative Performance</h2>
      <div class="grid">
        {rel_img or '<p class="muted">(relative chart not available)</p>'}
      </div>

      <h2>Profiling</h2>
      <div class="grid">
        {f'<div class="img"><img src="{profiling_img.name}" alt="{profiling_img.name}"></div>' if profiling_img.exists() else '<p class="muted">(profiling chart not available)</p>'}
      </div>
    </body>
    </html>
    """

    report_path.write_text(html)
    return report_path
