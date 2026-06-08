"""Repository-wide smoke tests for the cAMP dendritic spine model.

The fast conference pipeline is expected to run on normal Windows Python. The
full SMART/FEniCS model is tested only when dolfin is importable.
"""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "camp_realistic_model.py",
    "spine_mesh.xml",
    "run_conference_experiments.py",
    "experiments/run_experiments.py",
    "analysis/compute_metrics.py",
    "analysis/plot_results.py",
    "visualization/make_video_pyvista.py",
    "scripts/validate_outputs.py",
]

REQUIRED_IMPORTS = ["numpy", "pandas", "matplotlib"]
OPTIONAL_IMPORTS = ["pyvista", "imageio", "dolfin", "smart"]

EXPECTED_PLOT_FILES = [
    "continuous_vs_pulsed_camp.png",
    "pulse_width_response.png",
    "pulse_period_response.png",
    "diffusion_sweep.png",
    "pde_sweep.png",
    "downstream_phosphorylation.png",
    "spatial_gradient_index.png",
    "head_neck_ratio.png",
]


@dataclass
class CheckResult:
    status: str
    name: str
    detail: str = ""


class Reporter:
    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def pass_(self, name: str, detail: str = "") -> None:
        self._record("PASS", name, detail)

    def warn(self, name: str, detail: str = "") -> None:
        self._record("WARN", name, detail)

    def fail(self, name: str, detail: str = "") -> None:
        self._record("FAIL", name, detail)

    def _record(self, status: str, name: str, detail: str) -> None:
        self.results.append(CheckResult(status, name, detail))
        suffix = f" - {detail}" if detail else ""
        print(f"{status} {name}{suffix}")

    def overall_exit_code(self) -> int:
        return 1 if any(result.status == "FAIL" for result in self.results) else 0

    def print_summary(self) -> None:
        counts = {status: sum(r.status == status for r in self.results) for status in ("PASS", "WARN", "FAIL")}
        print("\nFinal summary")
        print(f"PASS: {counts['PASS']}  WARN: {counts['WARN']}  FAIL: {counts['FAIL']}")
        if counts["FAIL"]:
            print("FAIL overall")
        else:
            print("PASS overall")


def import_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def run_command(args: list[str], reporter: Reporter, name: str, timeout: int = 300) -> subprocess.CompletedProcess[str] | None:
    print(f"\nRunning: {' '.join(args)}")
    try:
        completed = subprocess.run(
            args,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        reporter.fail(name, f"timed out after {timeout}s")
        if exc.stdout:
            print(exc.stdout)
        return None

    if completed.stdout:
        print(completed.stdout.strip())
    if completed.returncode != 0:
        reporter.fail(name, f"exit code {completed.returncode}")
    return completed


def check_required_files(reporter: Reporter) -> None:
    missing = [path for path in REQUIRED_FILES if not (REPO_ROOT / path).exists()]
    if missing:
        reporter.fail("required files", "missing: " + ", ".join(missing))
    else:
        reporter.pass_("required files", f"{len(REQUIRED_FILES)} files found")


def check_imports(reporter: Reporter) -> None:
    missing_required = [name for name in REQUIRED_IMPORTS if not import_available(name)]
    if missing_required:
        reporter.fail("imports", "missing required: " + ", ".join(missing_required))
    else:
        reporter.pass_("imports", "required imports available: " + ", ".join(REQUIRED_IMPORTS))

    missing_optional = [name for name in OPTIONAL_IMPORTS if not import_available(name)]
    present_optional = [name for name in OPTIONAL_IMPORTS if name not in missing_optional]
    if present_optional:
        reporter.pass_("optional imports", "available: " + ", ".join(present_optional))
    if missing_optional:
        reporter.warn("optional imports", "missing: " + ", ".join(missing_optional))


def finite_numeric_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_columns: list[str] = []
    bad_columns: list[str] = []
    for column in df.columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().any():
            numeric_columns.append(column)
            if not numeric.dropna().map(math.isfinite).all():
                bad_columns.append(column)
    return numeric_columns, bad_columns


def validate_summary_metrics(path: Path, reporter: Reporter) -> bool:
    if not path.exists():
        reporter.fail("summary metrics", f"missing {path}")
        return False

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        reporter.fail("summary metrics", f"could not read CSV: {exc}")
        return False

    print("summary_metrics.csv columns: " + ", ".join(df.columns.astype(str)))
    if df.empty:
        reporter.fail("summary metrics", "CSV is empty")
        return False
    if len(df) < 3:
        reporter.fail("summary metrics", f"expected at least 3 rows, found {len(df)}")
        return False

    all_nan_columns = [column for column in df.columns if df[column].isna().all()]
    if all_nan_columns:
        reporter.fail("summary metrics", "all-NaN columns: " + ", ".join(all_nan_columns))
        return False

    numeric_columns, bad_numeric_columns = finite_numeric_columns(df)
    if not numeric_columns:
        reporter.fail("summary metrics", "no numeric columns found")
        return False
    if bad_numeric_columns:
        reporter.fail("summary metrics", "non-finite numeric values in: " + ", ".join(bad_numeric_columns))
        return False

    id_columns = {"run_id", "run_name", "name"}
    peak_columns = {"peak_cAMP", "peak_avg_cAMP", "peak_mean_cAMP", "peak_max_cAMP"}
    missing_concepts: list[str] = []
    if not id_columns.intersection(df.columns):
        missing_concepts.append("run identifier")
    if not peak_columns.intersection(df.columns):
        missing_concepts.append("peak cAMP metric")
    if "mode" not in df.columns:
        missing_concepts.append("mode")
    if missing_concepts:
        reporter.warn("summary metrics schema", "missing expected aggregate concept(s): " + ", ".join(missing_concepts))
    else:
        reporter.pass_("summary metrics schema", "run identifier, mode, and peak cAMP metric present")

    reporter.pass_("summary metrics", f"{len(df)} rows, {len(numeric_columns)} numeric columns")
    return True


def test_fast_pipeline(reporter: Reporter) -> None:
    completed = run_command([sys.executable, "run_conference_experiments.py"], reporter, "fast conference pipeline")
    if completed is None or completed.returncode != 0:
        return

    summary_path = RESULTS_DIR / "summary_metrics.csv"
    if validate_summary_metrics(summary_path, reporter):
        reporter.pass_("fast conference pipeline", "summary_metrics.csv generated and validated")

    pngs = sorted(FIGURES_DIR.glob("*.png"))
    if not FIGURES_DIR.exists():
        reporter.fail("fast figures", "figures/ directory was not created")
    elif len(pngs) < 3:
        reporter.fail("fast figures", f"expected at least 3 PNG files, found {len(pngs)}")
    else:
        reporter.pass_("fast figures", f"{len(pngs)} PNG files in figures/")


def plot_requirements_for_missing(summary: pd.DataFrame, filename: str) -> str:
    requirements = {
        "continuous_vs_pulsed_camp.png": ["run_id", "timecourse_csv or timeseries_csv"],
        "pulse_width_response.png": ["group", "pulse_width", "peak_cAMP or peak_avg_cAMP"],
        "pulse_period_response.png": ["group", "pulse_period", "peak_cAMP or peak_avg_cAMP"],
        "diffusion_sweep.png": ["group", "D_cAMP", "cAMP_fold_change or peak_cAMP"],
        "pde_sweep.png": ["group", "V_PDE", "cAMP_fold_change or peak_cAMP"],
        "downstream_phosphorylation.png": ["peak_avg_pSer845", "peak_avg_pSer831", "peak_PKA_frac"],
        "spatial_gradient_index.png": ["timecourse_csv or timeseries_csv", "gradient_index in timecourse"],
        "head_neck_ratio.png": ["timecourse_csv or timeseries_csv", "head_neck_ratio in timecourse"],
    }
    return "; needs " + ", ".join(requirements.get(filename, []))


def test_plotting(reporter: Reporter) -> None:
    completed = run_command([sys.executable, "analysis/plot_results.py"], reporter, "plotting")
    if completed is None or completed.returncode != 0:
        return

    summary_path = RESULTS_DIR / "summary_metrics.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    missing = [name for name in EXPECTED_PLOT_FILES if not (FIGURES_DIR / name).exists()]
    if not missing:
        reporter.pass_("plots", f"all {len(EXPECTED_PLOT_FILES)} expected plot files exist")
        return

    details = [name + plot_requirements_for_missing(summary, name) for name in missing]
    reporter.warn("plots", "missing or skipped: " + " | ".join(details))


def test_full_smart_fenics(reporter: Reporter) -> None:
    if not import_available("dolfin"):
        reporter.warn("full SMART/FEniCS", "Skipping full SMART/FEniCS model because dolfin is not installed in this environment.")
        return

    smoke_dir = RESULTS_DIR / "smoke_test"
    completed = run_command(
        [
            sys.executable,
            "camp_realistic_model.py",
            "--mode",
            "pulse_train",
            "--t_end",
            "0.2",
            "--dt",
            "0.1",
            "--save_every",
            "1",
            "--output_dir",
            str(smoke_dir),
        ],
        reporter,
        "full SMART/FEniCS smoke test",
        timeout=600,
    )
    if completed is None or completed.returncode != 0:
        return

    expected = ["timeseries.csv", "cAMP.xdmf", "Ca.xdmf", "ATP.xdmf"]
    missing = [name for name in expected if not (smoke_dir / name).exists()]
    if missing:
        reporter.fail("full SMART/FEniCS outputs", "missing: " + ", ".join(missing))
        return
    reporter.pass_("full SMART/FEniCS outputs", "smoke-test files generated")

    validate_cmd = [sys.executable, "scripts/validate_outputs.py", str(smoke_dir / "timeseries.csv")]
    validation = run_command(validate_cmd, reporter, "full SMART/FEniCS output validation")
    if validation is not None and validation.returncode == 0:
        reporter.pass_("full SMART/FEniCS output validation", "timeseries.csv passed validate_outputs.py")


def discover_and_validate_timeseries(reporter: Reporter) -> None:
    timeseries_files = sorted(RESULTS_DIR.rglob("timeseries.csv")) if RESULTS_DIR.exists() else []
    if not timeseries_files:
        reporter.warn("timeseries discovery", "no timeseries.csv files found under results/")
        return

    failures = 0
    for path in timeseries_files:
        completed = run_command(
            [sys.executable, "scripts/validate_outputs.py", str(path)],
            reporter,
            f"timeseries validation {path.relative_to(REPO_ROOT)}",
        )
        if completed is None or completed.returncode != 0:
            failures += 1
    if failures == 0:
        reporter.pass_("timeseries discovery", f"validated {len(timeseries_files)} timeseries.csv file(s)")


def main() -> int:
    reporter = Reporter()
    print(f"Testing repository: {REPO_ROOT}")

    check_required_files(reporter)
    check_imports(reporter)
    test_fast_pipeline(reporter)
    test_plotting(reporter)
    test_full_smart_fenics(reporter)
    discover_and_validate_timeseries(reporter)
    reporter.print_summary()
    return reporter.overall_exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
