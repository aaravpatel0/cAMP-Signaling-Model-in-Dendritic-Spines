"""
Conference proceeding experiment pipeline for the cAMP dendritic spine model.

This driver is intentionally fast: it uses a reduced 1D spine-axis
reaction-diffusion surrogate with the same experimental controls exposed by the
SMART/FEniCS model. Use it to generate reproducible sweeps, summary metrics,
plots, and fixed-limit inferno animations for paper drafting. The full
SMART/FEniCS model in camp_realistic_model.py accepts the same stimulation and
parameter CLI arguments for higher-fidelity reruns.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


MODES = ("continuous", "single_pulse", "pulse_train")


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    group: str
    mode: str = "continuous"
    t_end: float = 2.0
    dt: float = 0.05
    stim_amp: float = 1.0
    stim_start: float = 0.25
    pulse_width: float = 0.25
    pulse_period: float = 0.5
    pulse_count: int = 5
    D_cAMP: float = 30.0
    V_PDE: float = 2.0
    save_every: int = 1


def square_wave_stimulus(
    t: float,
    mode: str,
    stim_start: float,
    pulse_width: float,
    pulse_period: float,
    pulse_count: int,
    stim_amp: float,
) -> float:
    """Return stim_amp during pulse windows and zero otherwise."""
    if t < stim_start:
        return 0.0
    if mode == "continuous":
        return stim_amp
    if mode == "single_pulse":
        return stim_amp if stim_start <= t < stim_start + pulse_width else 0.0
    if pulse_period <= 0.0:
        return 0.0
    elapsed = t - stim_start
    pulse_idx = int(elapsed // pulse_period)
    in_train = pulse_idx < pulse_count
    in_width = (elapsed - pulse_idx * pulse_period) < pulse_width
    return stim_amp if in_train and in_width else 0.0


def simulate_reduced_spine(config: RunConfig, n_nodes: int = 80) -> dict[str, np.ndarray]:
    """Fast cAMP reaction-diffusion surrogate along the neck-to-head axis."""
    x = np.linspace(0.0, 1.0, n_nodes)
    dx = x[1] - x[0]
    c = np.full(n_nodes, 0.10, dtype=float)

    head_source = np.exp(-0.5 * ((x - 0.86) / 0.12) ** 2)
    head_source /= head_source.mean()
    neck_sink = np.exp(-0.5 * (x / 0.10) ** 2)

    t_values: list[float] = []
    mean_values: list[float] = []
    max_values: list[float] = []
    min_values: list[float] = []
    gradient_values: list[float] = []
    stim_values: list[float] = []
    fields: list[np.ndarray] = []
    field_times: list[float] = []

    n_steps = int(np.ceil(config.t_end / config.dt))
    # Effective diffusion is scaled to this reduced 1D geometry while preserving
    # the requested D_cAMP ordering across the sweep.
    diff_coeff = 0.0015 * config.D_cAMP
    km_pde = 0.35
    production_rate = 3.0
    pde_scale = 0.12
    neck_exchange = 0.10
    basal_camp = 0.02

    for step in range(n_steps + 1):
        t = min(step * config.dt, config.t_end)
        stim = square_wave_stimulus(
            t,
            config.mode,
            config.stim_start,
            config.pulse_width,
            config.pulse_period,
            config.pulse_count,
            config.stim_amp,
        )

        if step % max(config.save_every, 1) == 0 or step == n_steps:
            mean_c = float(c.mean())
            max_c = float(c.max())
            min_c = float(c.min())
            t_values.append(t)
            mean_values.append(mean_c)
            max_values.append(max_c)
            min_values.append(min_c)
            gradient_values.append((max_c - min_c) / max(mean_c, 1e-12))
            stim_values.append(stim)
            fields.append(c.copy())
            field_times.append(t)

        if step == n_steps:
            break

        remaining_dt = min(config.dt, config.t_end - t)
        # Explicit substepping keeps the prototype stable for the full D sweep.
        stable_dt = 0.35 * dx * dx / max(diff_coeff, 1e-12)
        n_substeps = max(1, int(np.ceil(remaining_dt / stable_dt)))
        sub_dt = remaining_dt / n_substeps

        for _ in range(n_substeps):
            lap = np.empty_like(c)
            lap[1:-1] = (c[:-2] - 2.0 * c[1:-1] + c[2:]) / (dx * dx)
            lap[0] = (c[1] - c[0]) / (dx * dx)
            lap[-1] = (c[-2] - c[-1]) / (dx * dx)

            pde_loss = pde_scale * config.V_PDE * c / (km_pde + c)
            production = production_rate * stim * head_source
            neck_loss = neck_exchange * neck_sink * (c - basal_camp)
            c = np.maximum(c + sub_dt * (diff_coeff * lap + production - pde_loss - neck_loss), 0.0)

    return {
        "x": x,
        "t": np.array(t_values),
        "mean_cAMP": np.array(mean_values),
        "max_cAMP": np.array(max_values),
        "min_cAMP": np.array(min_values),
        "gradient_index": np.array(gradient_values),
        "stimulus": np.array(stim_values),
        "fields": np.array(fields),
        "field_times": np.array(field_times),
    }


def build_experiment_set(args: argparse.Namespace) -> list[RunConfig]:
    t_end = args.t_end if args.t_end is not None else (8.0 if args.publication else 2.0)
    dt = args.dt if args.dt is not None else (0.01 if args.publication else 0.05)
    base = dict(
        t_end=t_end,
        dt=dt,
        stim_amp=args.stim_amp,
        stim_start=args.stim_start,
        pulse_width=args.pulse_width,
        pulse_period=args.pulse_period,
        pulse_count=args.pulse_count,
        D_cAMP=args.D_cAMP,
        V_PDE=args.V_PDE,
        save_every=args.save_every,
    )

    configs = [
        RunConfig("A_continuous_baseline", "A_continuous_baseline", mode="continuous", **base)
    ]
    for width in (0.1, 0.25, 0.5):
        configs.append(
            RunConfig(
                f"B_single_pulse_width_{width:g}s",
                "B_single_pulse_width",
                mode="single_pulse",
                pulse_width=width,
                **{k: v for k, v in base.items() if k != "pulse_width"},
            )
        )
    for period in (0.25, 0.5, 1.0):
        configs.append(
            RunConfig(
                f"C_pulse_train_period_{period:g}s",
                "C_pulse_train_period",
                mode="pulse_train",
                pulse_period=period,
                **{k: v for k, v in base.items() if k != "pulse_period"},
            )
        )
    for diffusion in (5.0, 10.0, 20.0, 30.0, 50.0):
        configs.append(
            RunConfig(
                f"D_diffusion_{diffusion:g}",
                "D_diffusion_sweep",
                D_cAMP=diffusion,
                **{k: v for k, v in base.items() if k != "D_cAMP"},
            )
        )
    for pde in (1.0, 2.0, 5.0, 10.0):
        configs.append(
            RunConfig(
                f"E_pde_{pde:g}",
                "E_pde_sweep",
                V_PDE=pde,
                **{k: v for k, v in base.items() if k != "V_PDE"},
            )
        )
    return configs


def write_timecourse(path: Path, data: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "mean_cAMP", "max_cAMP", "min_cAMP", "gradient_index", "stimulus"])
        for row in zip(
            data["t"],
            data["mean_cAMP"],
            data["max_cAMP"],
            data["min_cAMP"],
            data["gradient_index"],
            data["stimulus"],
        ):
            writer.writerow([f"{value:.8g}" for value in row])


def summarize_run(config: RunConfig, data: dict[str, np.ndarray], timecourse_csv: Path) -> dict[str, object]:
    peak_idx = int(np.argmax(data["mean_cAMP"]))
    return {
        **asdict(config),
        "time_to_peak": float(data["t"][peak_idx]),
        "peak_cAMP": float(data["mean_cAMP"][peak_idx]),
        "final_cAMP": float(data["mean_cAMP"][-1]),
        "peak_max_cAMP": float(data["max_cAMP"].max()),
        "final_max_cAMP": float(data["max_cAMP"][-1]),
        "final_min_cAMP": float(data["min_cAMP"][-1]),
        "peak_gradient_index": float(data["gradient_index"].max()),
        "final_gradient_index": float(data["gradient_index"][-1]),
        "timecourse_csv": str(timecourse_csv),
    }


def save_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def style_axes(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="0.88", linewidth=0.8)


def generate_plots(figures_dir: Path, runs: dict[str, dict[str, np.ndarray]], summary: list[dict[str, object]]) -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
    })

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for run_id in [
        "A_continuous_baseline",
        "B_single_pulse_width_0.25s",
        "C_pulse_train_period_0.5s",
    ]:
        data = runs[run_id]
        ax.plot(data["t"], data["mean_cAMP"], linewidth=2, label=run_id.replace("_", " "))
    style_axes(ax, "Time (s)", "Mean cAMP (uM)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "continuous_vs_pulsed_timecourse.png")
    plt.close(fig)

    period_rows = [r for r in summary if r["group"] == "C_pulse_train_period"]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot([r["pulse_period"] for r in period_rows], [r["peak_cAMP"] for r in period_rows],
            marker="o", color="#2a6f97", linewidth=2)
    style_axes(ax, "Pulse period (s)", "Peak mean cAMP (uM)")
    fig.tight_layout()
    fig.savefig(figures_dir / "pulse_period_response.png")
    plt.close(fig)

    diffusion_rows = [r for r in summary if r["group"] == "D_diffusion_sweep"]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot([r["D_cAMP"] for r in diffusion_rows], [r["peak_gradient_index"] for r in diffusion_rows],
            marker="o", color="#8f2d56", linewidth=2)
    style_axes(ax, "D_cAMP (um^2/s)", "Peak gradient index")
    fig.tight_layout()
    fig.savefig(figures_dir / "diffusion_sweep_gradient_index.png")
    plt.close(fig)

    pde_rows = [r for r in summary if r["group"] == "E_pde_sweep"]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot([r["V_PDE"] for r in pde_rows], [r["peak_gradient_index"] for r in pde_rows],
            marker="o", color="#386641", linewidth=2)
    style_axes(ax, "V_PDE (uM/s)", "Peak gradient index")
    fig.tight_layout()
    fig.savefig(figures_dir / "pde_sweep_gradient_index.png")
    plt.close(fig)


def save_animation(figures_dir: Path, filename_stem: str, data: dict[str, np.ndarray], clim: tuple[float, float]) -> None:
    fields = data["fields"]
    times = data["field_times"]
    x = data["x"]

    fig, ax = plt.subplots(figsize=(5.0, 1.5))
    image = ax.imshow(
        fields[0][None, :],
        aspect="auto",
        cmap="inferno",
        vmin=clim[0],
        vmax=clim[1],
        extent=[x.min(), x.max(), 0.0, 1.0],
    )
    ax.set_yticks([])
    ax.set_xlabel("Neck to spine head axis")
    title = ax.set_title(f"t = {times[0]:.2f} s")
    fig.colorbar(image, ax=ax, label="cAMP (uM)", fraction=0.08, pad=0.02)

    def update(frame_idx: int):
        image.set_data(fields[frame_idx][None, :])
        title.set_text(f"t = {times[frame_idx]:.2f} s")
        return image, title

    ani = animation.FuncAnimation(fig, update, frames=len(fields), interval=100, blit=False)
    gif_path = figures_dir / f"{filename_stem}.gif"
    try:
        ani.save(gif_path, writer=animation.PillowWriter(fps=8))
    except Exception:
        for idx in np.linspace(0, len(fields) - 1, min(8, len(fields)), dtype=int):
            update(int(idx))
            fig.savefig(figures_dir / f"{filename_stem}_frame_{idx:03d}.png")
    finally:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run conference-ready cAMP experiment sweeps.")
    parser.add_argument("--mode", choices=MODES, default="continuous",
                        help="Accepted for parity with the SMART model CLI; sweeps run all modes.")
    parser.add_argument("--t_end", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--stim_amp", type=float, default=1.0)
    parser.add_argument("--stim_start", type=float, default=0.25)
    parser.add_argument("--pulse_width", type=float, default=0.25)
    parser.add_argument("--pulse_period", type=float, default=0.5)
    parser.add_argument("--pulse_count", type=int, default=5)
    parser.add_argument("--D_cAMP", type=float, default=30.0)
    parser.add_argument("--V_PDE", type=float, default=2.0)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--publication", action="store_true",
                        help="Use longer t_end and smaller dt defaults.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    timecourse_dir = output_dir / "timecourses"
    figures_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    timecourse_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    runs: dict[str, dict[str, np.ndarray]] = {}
    summary_rows: list[dict[str, object]] = []
    for config in build_experiment_set(args):
        data = simulate_reduced_spine(config)
        runs[config.run_id] = data
        timecourse_csv = timecourse_dir / f"{config.run_id}.csv"
        write_timecourse(timecourse_csv, data)
        summary_rows.append(summarize_run(config, data, timecourse_csv))

    summary_path = output_dir / "summary_metrics.csv"
    save_summary(summary_path, summary_rows)
    generate_plots(figures_dir, runs, summary_rows)

    continuous = runs["A_continuous_baseline"]
    pulse_train = runs["C_pulse_train_period_0.5s"]
    global_min = min(float(continuous["fields"].min()), float(pulse_train["fields"].min()))
    global_max = max(float(continuous["fields"].max()), float(pulse_train["fields"].max()))
    save_animation(figures_dir, "continuous_stimulation", continuous, (global_min, global_max))
    save_animation(figures_dir, "pulse_train_stimulation", pulse_train, (global_min, global_max))

    print(f"Wrote {len(summary_rows)} runs to {summary_path}")
    print(f"Wrote figures to {figures_dir.resolve()}")


if __name__ == "__main__":
    main()
