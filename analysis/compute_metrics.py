"""Compute summary metrics from SMART/FEniCS experiment time series."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENTS_DIR = REPO_ROOT / "results" / "experiments"
DEFAULT_SUMMARY = REPO_ROOT / "results" / "summary_metrics.csv"


def read_manifest(experiments_dir: Path) -> pd.DataFrame:
    manifest_path = experiments_dir / "manifest.csv"
    if not manifest_path.exists():
        return pd.DataFrame()
    manifest = pd.read_csv(manifest_path)
    manifest["run_id"] = manifest["run_id"].astype(str)
    return manifest


def compute_one(timeseries_path: Path, metadata: dict[str, object]) -> dict[str, object]:
    df = pd.read_csv(timeseries_path)
    required = [
        "t",
        "avg_cAMP",
        "PKA_frac",
        "avg_pSer845",
        "avg_pSer831",
        "CaMKII_p_frac",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{timeseries_path} is missing columns: {', '.join(missing)}")

    peak_idx = int(df["avg_cAMP"].idxmax())
    basal = float(df["avg_cAMP"].iloc[0])
    peak = float(df["avg_cAMP"].iloc[peak_idx])
    auc = float(np.trapezoid(df["avg_cAMP"].to_numpy(), df["t"].to_numpy()))

    metrics = {
        **metadata,
        "timeseries_csv": str(timeseries_path),
        "peak_avg_cAMP": peak,
        "final_avg_cAMP": float(df["avg_cAMP"].iloc[-1]),
        "time_to_peak": float(df["t"].iloc[peak_idx]),
        "cAMP_fold_change": peak / basal if basal > 1e-12 else np.nan,
        "peak_PKA_frac": float(df["PKA_frac"].max()),
        "peak_avg_pSer845": float(df["avg_pSer845"].max()),
        "peak_avg_pSer831": float(df["avg_pSer831"].max()),
        "final_CaMKII_p_frac": float(df["CaMKII_p_frac"].iloc[-1]),
        "avg_cAMP_auc": auc,
    }
    optional_columns = {
        "gradient_index": ("peak_gradient_index", "final_gradient_index"),
        "head_neck_ratio": ("peak_head_neck_ratio", "final_head_neck_ratio"),
        "max_cAMP": ("peak_max_cAMP", "final_max_cAMP"),
    }
    for column, (peak_name, final_name) in optional_columns.items():
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            metrics[peak_name] = float(values.max(skipna=True))
            metrics[final_name] = float(values.iloc[-1]) if not pd.isna(values.iloc[-1]) else np.nan
        else:
            metrics[peak_name] = np.nan
            metrics[final_name] = np.nan
    return metrics


def collect_metrics(experiments_dir: Path) -> pd.DataFrame:
    manifest = read_manifest(experiments_dir)
    manifest_by_run = {}
    if not manifest.empty:
        manifest_by_run = {
            str(row["run_id"]): row.dropna().to_dict()
            for _, row in manifest.iterrows()
        }

    rows: list[dict[str, object]] = []
    for timeseries_path in sorted(experiments_dir.glob("*/timeseries.csv")):
        run_id = timeseries_path.parent.name
        metadata = manifest_by_run.get(
            run_id,
            {
                "run_id": run_id,
                "group": "unlabeled",
                "status": "completed",
                "output_dir": str(timeseries_path.parent),
            },
        )
        if str(metadata.get("status", "completed")) == "failed":
            continue
        rows.append(compute_one(timeseries_path, metadata))

    if not rows:
        raise FileNotFoundError(f"No timeseries.csv files found under {experiments_dir}")
    return pd.DataFrame(rows).sort_values(["group", "run_id"], kind="stable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute ICBES summary metrics from experiment outputs.")
    parser.add_argument("--experiments_dir", default=str(DEFAULT_EXPERIMENTS_DIR))
    parser.add_argument("--output_csv", default=str(DEFAULT_SUMMARY))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiments_dir = Path(args.experiments_dir).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    summary = collect_metrics(experiments_dir)
    summary.to_csv(output_csv, index=False)
    print(f"Wrote {len(summary)} rows to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
