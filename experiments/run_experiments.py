"""Batch launcher for SMART/FEniCS cAMP conference experiments.

Each experiment calls ``camp_realistic_model.py`` in a subprocess and writes the
run output to a unique directory under ``results/experiments``.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "experiments"


@dataclass(frozen=True)
class Experiment:
    run_id: str
    group: str
    mode: str
    pulse_width: float
    pulse_period: float
    pulse_count: int
    D_cAMP: float
    V_PDE: float
    stim_amp: float = 1.0
    stim_start: float = 0.25


def build_experiments(args: argparse.Namespace) -> list[Experiment]:
    base = {
        "pulse_width": args.pulse_width,
        "pulse_period": args.pulse_period,
        "pulse_count": args.pulse_count,
        "D_cAMP": args.D_cAMP,
        "V_PDE": args.V_PDE,
        "stim_amp": args.stim_amp,
        "stim_start": args.stim_start,
    }
    experiments = [
        Experiment("A_continuous_baseline", "A_continuous_baseline", "continuous", **base),
    ]

    for width in (0.1, 0.25, 0.5):
        params = dict(base)
        params["pulse_width"] = width
        experiments.append(
            Experiment(
                f"B_single_pulse_width_{width:g}s",
                "B_single_pulse_width",
                "single_pulse",
                **params,
            )
        )

    for period in (0.25, 0.5, 1.0):
        params = dict(base)
        params["pulse_period"] = period
        experiments.append(
            Experiment(
                f"C_pulse_train_period_{period:g}s",
                "C_pulse_train_period",
                "pulse_train",
                **params,
            )
        )

    for diffusion in (5.0, 10.0, 20.0, 30.0, 50.0):
        params = dict(base)
        params["D_cAMP"] = diffusion
        experiments.append(
            Experiment(
                f"D_D_cAMP_{diffusion:g}",
                "D_D_cAMP_sweep",
                "continuous",
                **params,
            )
        )

    for pde in (1.0, 2.0, 5.0, 10.0):
        params = dict(base)
        params["V_PDE"] = pde
        experiments.append(
            Experiment(
                f"E_V_PDE_{pde:g}",
                "E_V_PDE_sweep",
                "continuous",
                **params,
            )
        )

    return experiments


def write_manifest(manifest_path: Path, rows: list[dict[str, object]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "group",
        "status",
        "returncode",
        "output_dir",
        "mode",
        "t_end",
        "dt",
        "stim_amp",
        "stim_start",
        "pulse_width",
        "pulse_period",
        "pulse_count",
        "D_cAMP",
        "V_PDE",
        "save_every",
        "publication",
        "command",
    ]
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_one(
    experiment: Experiment,
    args: argparse.Namespace,
    model_script: Path,
    results_dir: Path,
) -> dict[str, object]:
    output_dir = results_dir / experiment.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"

    cmd = [
        args.python,
        str(model_script),
        "--mode",
        experiment.mode,
        "--t_end",
        str(args.t_end),
        "--dt",
        str(args.dt),
        "--stim_amp",
        str(experiment.stim_amp),
        "--stim_start",
        str(experiment.stim_start),
        "--pulse_width",
        str(experiment.pulse_width),
        "--pulse_period",
        str(experiment.pulse_period),
        "--pulse_count",
        str(experiment.pulse_count),
        "--D_cAMP",
        str(experiment.D_cAMP),
        "--V_PDE",
        str(experiment.V_PDE),
        "--save_every",
        str(args.save_every),
        "--output_dir",
        str(output_dir),
    ]
    if args.publication:
        cmd.append("--publication")

    if args.dry_run:
        status = "dry_run"
        returncode = 0
        log_path.write_text("Dry run only; command was not executed.\n" + " ".join(cmd) + "\n")
    else:
        with log_path.open("w") as log_file:
            log_file.write("Command: " + " ".join(cmd) + "\n\n")
            log_file.flush()
            completed = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        returncode = completed.returncode
        status = "completed" if returncode == 0 else "failed"

    return {
        **asdict(experiment),
        "status": status,
        "returncode": returncode,
        "output_dir": str(output_dir),
        "t_end": args.t_end,
        "dt": args.dt,
        "save_every": args.save_every,
        "publication": args.publication,
        "command": " ".join(cmd),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ICBES cAMP experiment batches.")
    parser.add_argument("--python", default=sys.executable, help="Python executable inside the SMART environment.")
    parser.add_argument("--model_script", default=str(REPO_ROOT / "camp_realistic_model.py"))
    parser.add_argument("--results_dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--t_end", type=float, default=0.5, help="Quick prototype default; use --publication for paper runs.")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--stim_amp", type=float, default=1.0)
    parser.add_argument("--stim_start", type=float, default=0.25)
    parser.add_argument("--pulse_width", type=float, default=0.25)
    parser.add_argument("--pulse_period", type=float, default=0.5)
    parser.add_argument("--pulse_count", type=int, default=5)
    parser.add_argument("--D_cAMP", type=float, default=30.0)
    parser.add_argument("--V_PDE", type=float, default=2.0)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--publication", action="store_true", help="Use paper-grade defaults unless t_end/dt are overridden.")
    parser.add_argument("--max_runs", type=int, default=None, help="Limit runs for smoke testing.")
    parser.add_argument("--only_group", choices=[
        "A_continuous_baseline",
        "B_single_pulse_width",
        "C_pulse_train_period",
        "D_D_cAMP_sweep",
        "E_V_PDE_sweep",
    ], default=None)
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.publication and "--t_end" not in sys.argv:
        args.t_end = 10.0
    if args.publication and "--dt" not in sys.argv:
        args.dt = 0.02

    model_script = Path(args.model_script).resolve()
    results_dir = Path(args.results_dir).resolve()
    experiments = build_experiments(args)
    if args.only_group:
        experiments = [exp for exp in experiments if exp.group == args.only_group]
    if args.max_runs is not None:
        experiments = experiments[: args.max_runs]

    rows: list[dict[str, object]] = []
    for index, experiment in enumerate(experiments, start=1):
        print(f"[{index}/{len(experiments)}] {experiment.run_id}")
        row = run_one(experiment, args, model_script, results_dir)
        rows.append(row)
        write_manifest(results_dir / "manifest.csv", rows)
        if row["status"] == "failed" and not args.continue_on_error:
            print(f"Run failed. See {Path(row['output_dir']) / 'run.log'}")
            return int(row["returncode"])

    write_manifest(results_dir / "manifest.csv", rows)
    print(f"Wrote manifest: {results_dir / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
