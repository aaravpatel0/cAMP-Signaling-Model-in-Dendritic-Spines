"""Lightweight validation checks for cAMP model timeseries outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_REQUIRED_COLUMNS = [
    "t",
    "avg_cAMP",
    "avg_Ca",
    "avg_ATP",
    "PKA_frac",
    "avg_pSer845",
    "avg_pSer831",
    "CaMKII_p_frac",
]

SPATIAL_REQUIRED_COLUMNS = [
    "min_cAMP",
    "max_cAMP",
    "gradient_index",
    "stimulus",
    "neck_avg_cAMP",
    "head_avg_cAMP",
    "head_neck_ratio",
]


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)
    print(f"FAIL: {message}")


def warn(message: str) -> None:
    print(f"WARN: {message}")


def check_required_columns(df: pd.DataFrame, failures: list[str]) -> None:
    missing = [column for column in BASE_REQUIRED_COLUMNS + SPATIAL_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        fail(f"missing required columns: {', '.join(missing)}", failures)


def check_nonnegative(df: pd.DataFrame, failures: list[str]) -> None:
    for column in ["avg_cAMP", "min_cAMP", "max_cAMP", "avg_Ca", "avg_ATP"]:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if (values < -1e-12).any():
            fail(f"{column} contains negative values", failures)


def check_stimulus_response(df: pd.DataFrame, failures: list[str]) -> None:
    if "stimulus" not in df.columns or "avg_cAMP" not in df.columns:
        return
    stimulus = pd.to_numeric(df["stimulus"], errors="coerce").fillna(0.0)
    camp = pd.to_numeric(df["avg_cAMP"], errors="coerce")
    active_indices = np.flatnonzero(stimulus.to_numpy() > 1e-12)
    if len(active_indices) == 0:
        warn("no active stimulus samples found; skipping cAMP response check")
        return
    start_idx = int(active_indices[0])
    baseline = float(camp.iloc[max(start_idx - 1, 0)])
    post_stim = camp.iloc[start_idx:]
    if float((post_stim - baseline).abs().max()) <= 1e-9:
        fail("avg_cAMP does not change after stimulus onset", failures)


def check_finite_spatial(df: pd.DataFrame, failures: list[str]) -> None:
    if "gradient_index" in df.columns:
        values = pd.to_numeric(df["gradient_index"], errors="coerce")
        if not np.isfinite(values).all():
            fail("gradient_index contains non-finite values", failures)

    head_neck_columns = {"neck_avg_cAMP", "head_avg_cAMP", "head_neck_ratio"}
    if head_neck_columns.issubset(df.columns):
        ratio = pd.to_numeric(df["head_neck_ratio"], errors="coerce")
        if not np.isfinite(ratio).all():
            fail("head_neck_ratio contains non-finite values", failures)


def validate(timeseries_path: Path) -> int:
    failures: list[str] = []
    if not timeseries_path.exists():
        fail(f"timeseries.csv not found: {timeseries_path}", failures)
        return 1

    df = pd.read_csv(timeseries_path)
    if df.empty:
        fail("timeseries.csv is empty", failures)
        return 1

    check_required_columns(df, failures)
    check_nonnegative(df, failures)
    check_stimulus_response(df, failures)
    check_finite_spatial(df, failures)

    if failures:
        print(f"Validation failed with {len(failures)} issue(s).")
        return 1
    print(f"Validation passed: {timeseries_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate cAMP model timeseries output.")
    parser.add_argument("timeseries_csv", help="Path to a run's timeseries.csv file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return validate(Path(args.timeseries_csv))


if __name__ == "__main__":
    raise SystemExit(main())
