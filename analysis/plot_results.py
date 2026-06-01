"""Generate publication-style figures for the ICBES cAMP experiment pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = REPO_ROOT / "results" / "summary_metrics.csv"
DEFAULT_FIGURES = REPO_ROOT / "figures"


def configure_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def style_axes(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="0.88", linewidth=0.8)


def mark_no_data(ax: plt.Axes) -> None:
    ax.text(0.5, 0.5, "No matching runs", ha="center", va="center", transform=ax.transAxes)


def read_timecourse(row: pd.Series) -> pd.DataFrame:
    return pd.read_csv(Path(row["timeseries_csv"]))


def plot_continuous_vs_pulsed(summary: pd.DataFrame, figures_dir: Path) -> None:
    wanted = [
        "A_continuous_baseline",
        "B_single_pulse_width_0.25s",
        "C_pulse_train_period_0.5s",
    ]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for run_id in wanted:
        match = summary[summary["run_id"] == run_id]
        if match.empty:
            continue
        row = match.iloc[0]
        ts = read_timecourse(row)
        ax.plot(ts["t"], ts["avg_cAMP"], linewidth=2, label=run_id.replace("_", " "))
    style_axes(ax, "Time (s)", "Average cAMP (uM)")
    if ax.lines:
        ax.legend(frameon=False)
    else:
        mark_no_data(ax)
    fig.tight_layout()
    fig.savefig(figures_dir / "continuous_vs_pulsed_camp.png")
    plt.close(fig)


def plot_response(summary: pd.DataFrame, group: str, x_column: str, y_column: str, xlabel: str, filename: str, figures_dir: Path) -> None:
    if x_column not in summary.columns or y_column not in summary.columns:
        rows = pd.DataFrame()
    else:
        rows = summary[summary["group"] == group].sort_values(x_column)
        rows = rows.dropna(subset=[x_column, y_column])
        rows[x_column] = pd.to_numeric(rows[x_column], errors="coerce")
        rows[y_column] = pd.to_numeric(rows[y_column], errors="coerce")
        rows = rows.dropna(subset=[x_column, y_column]).sort_values(x_column)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    if rows.empty:
        mark_no_data(ax)
    else:
        ax.plot(rows[x_column], rows[y_column], marker="o", linewidth=2, color="#2a6f97")
    style_axes(ax, xlabel, y_column.replace("_", " "))
    fig.tight_layout()
    fig.savefig(figures_dir / filename)
    plt.close(fig)


def plot_downstream(summary: pd.DataFrame, figures_dir: Path) -> None:
    required = {"group", "run_id", "peak_avg_pSer845", "peak_avg_pSer831", "peak_PKA_frac"}
    if not required.issubset(summary.columns):
        rows = pd.DataFrame()
    else:
        rows = summary[
            summary["group"].isin([
                "A_continuous_baseline",
                "B_single_pulse_width",
                "C_pulse_train_period",
            ])
        ].copy()
        rows = rows.sort_values(["group", "run_id"])

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    if rows.empty:
        mark_no_data(ax)
    else:
        labels = rows["run_id"].str.replace("_", " ", regex=False)
        x = range(len(rows))
        width = 0.25
        ax.bar([i - width for i in x], rows["peak_avg_pSer845"], width=width, label="pSer845", color="#8f2d56")
        ax.bar(x, rows["peak_avg_pSer831"], width=width, label="pSer831", color="#386641")
        ax.bar([i + width for i in x], rows["peak_PKA_frac"], width=width, label="PKA frac", color="#2a6f97")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.legend(frameon=False, ncol=3)
    style_axes(ax, "", "Peak response")
    fig.tight_layout()
    fig.savefig(figures_dir / "downstream_phosphorylation.png")
    plt.close(fig)


def plot_spatial_timecourse(summary: pd.DataFrame, column: str, ylabel: str, filename: str, figures_dir: Path) -> None:
    wanted = [
        "A_continuous_baseline",
        "B_single_pulse_width_0.25s",
        "C_pulse_train_period_0.5s",
    ]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for run_id in wanted:
        match = summary[summary["run_id"] == run_id]
        if match.empty:
            continue
        ts = read_timecourse(match.iloc[0])
        if column not in ts.columns:
            continue
        ax.plot(ts["t"], ts[column], linewidth=2, label=run_id.replace("_", " "))
    style_axes(ax, "Time (s)", ylabel)
    if ax.lines:
        ax.legend(frameon=False)
    else:
        mark_no_data(ax)
    fig.tight_layout()
    fig.savefig(figures_dir / filename)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ICBES cAMP experiment results.")
    parser.add_argument("--summary_csv", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--figures_dir", default=str(DEFAULT_FIGURES))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_style()
    summary = pd.read_csv(args.summary_csv)
    figures_dir = Path(args.figures_dir).resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_continuous_vs_pulsed(summary, figures_dir)
    plot_response(summary, "B_single_pulse_width", "pulse_width", "peak_avg_cAMP",
                  "Pulse width (s)", "pulse_width_response.png", figures_dir)
    plot_response(summary, "C_pulse_train_period", "pulse_period", "peak_avg_cAMP",
                  "Pulse period (s)", "pulse_period_response.png", figures_dir)
    plot_response(summary, "D_D_cAMP_sweep", "D_cAMP", "cAMP_fold_change",
                  "D_cAMP (um^2/s)", "diffusion_sweep.png", figures_dir)
    plot_response(summary, "E_V_PDE_sweep", "V_PDE", "cAMP_fold_change",
                  "V_PDE (uM/s)", "pde_sweep.png", figures_dir)
    plot_downstream(summary, figures_dir)
    plot_spatial_timecourse(summary, "gradient_index", "Gradient index", "spatial_gradient_index.png", figures_dir)
    plot_spatial_timecourse(summary, "head_neck_ratio", "Head/neck cAMP ratio", "head_neck_ratio.png", figures_dir)

    print(f"Wrote figures to {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
