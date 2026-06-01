"""Render cAMP XDMF/VTK outputs to MP4 or GIF with PyVista."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv


def parse_clim(value: str | None) -> tuple[float, float] | None:
    if value is None:
        return None
    low, high = value.split(",", maxsplit=1)
    return float(low), float(high)


def load_frames(input_path: Path, scalar: str) -> tuple[list[pv.DataSet], list[float]]:
    if input_path.is_dir():
        files = sorted(
            list(input_path.glob("*.vtu"))
            + list(input_path.glob("*.vtk"))
            + list(input_path.glob("*.vtp"))
            + list(input_path.glob("*.xdmf"))
        )
        if not files:
            raise FileNotFoundError(f"No VTK/XDMF files found in {input_path}")
        return [pv.read(path) for path in files], [float(i) for i in range(len(files))]

    reader = pv.get_reader(str(input_path))
    time_values = getattr(reader, "time_values", None)
    if time_values:
        frames = []
        for time_value in time_values:
            reader.set_active_time_value(time_value)
            frames.append(reader.read())
        return frames, [float(t) for t in time_values]

    return [pv.read(input_path)], [0.0]


def ensure_scalar(mesh: pv.DataSet, scalar: str) -> str:
    if scalar in mesh.array_names:
        return scalar
    if mesh.active_scalars_name:
        return mesh.active_scalars_name
    if mesh.array_names:
        return mesh.array_names[0]
    raise ValueError("No scalar arrays found in the input mesh.")


def compute_clim(frames: list[pv.DataSet], scalar: str) -> tuple[float, float]:
    mins = []
    maxs = []
    for frame in frames:
        scalar_name = ensure_scalar(frame, scalar)
        values = np.asarray(frame[scalar_name])
        mins.append(float(np.nanmin(values)))
        maxs.append(float(np.nanmax(values)))
    return min(mins), max(maxs)


def parse_camera(value: str | None):
    if value is None:
        return None
    parts = [float(part) for part in value.split(",")]
    if len(parts) != 9:
        raise ValueError("--camera must have 9 comma-separated values: pos3,focal3,viewup3")
    return parts[0:3], parts[3:6], parts[6:9]


def render_video(
    frames: list[pv.DataSet],
    times: list[float],
    output_path: Path,
    scalar: str,
    clim: tuple[float, float],
    camera,
    fps: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(off_screen=True, window_size=(1200, 900))
    if output_path.suffix.lower() == ".gif":
        plotter.open_gif(str(output_path), fps=fps)
    else:
        plotter.open_movie(str(output_path), framerate=fps)

    scalar_name = ensure_scalar(frames[0], scalar)
    actor = plotter.add_mesh(
        frames[0],
        scalars=scalar_name,
        cmap="inferno",
        clim=clim,
        scalar_bar_args={"title": "cAMP concentration (uM)"},
    )
    plotter.add_text(f"t = {times[0]:.3g}", position="upper_left", font_size=12, name="time_label")
    if camera is not None:
        plotter.camera_position = camera
    else:
        plotter.view_isometric()
        plotter.camera.zoom(1.1)
    fixed_camera = plotter.camera_position
    plotter.write_frame()

    for frame, time_value in zip(frames[1:], times[1:]):
        plotter.remove_actor(actor)
        scalar_name = ensure_scalar(frame, scalar)
        actor = plotter.add_mesh(
            frame,
            scalars=scalar_name,
            cmap="inferno",
            clim=clim,
            scalar_bar_args={"title": "cAMP concentration (uM)"},
        )
        plotter.add_text(f"t = {time_value:.3g}", position="upper_left", font_size=12, name="time_label")
        plotter.camera_position = fixed_camera
        plotter.write_frame()
    plotter.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render cAMP XDMF/VTK output to MP4/GIF.")
    parser.add_argument("input", help="Path to cAMP.xdmf, a VTK file, or a directory of VTK/XDMF frames.")
    parser.add_argument("--output", default="figures/camp_render.mp4")
    parser.add_argument("--scalar", default="cAMP")
    parser.add_argument("--clim", default=None, help="Fixed color limits as min,max. Auto-computed if omitted.")
    parser.add_argument("--camera", default=None, help="Fixed camera as posx,posy,posz,fx,fy,fz,ux,uy,uz.")
    parser.add_argument("--fps", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    frames, times = load_frames(input_path, args.scalar)
    clim = parse_clim(args.clim) or compute_clim(frames, args.scalar)
    camera = parse_camera(args.camera)
    render_video(frames, times, output_path, args.scalar, clim, camera, args.fps)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
