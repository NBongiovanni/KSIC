from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from KSIC_v6.rendering import render_trajectory_sequence, create_snapshots_figure
from KSIC_v6.utils import to_numpy
from KSIC_v6.models import VisionValForwardOutputs

from .utils.extract_images import (
    extract_open_loop_pred_gt_imgs_2d,
    extract_open_loop_pred_gt_imgs_3d,
)
from .plotters.single_plotter import SinglePlotExtractors, SingleStateInputPlotter
from .plotters.basic_plotter import BasicStatePlotter

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RenderOpenLoopConfig:
    modality: str                 # "vision" | "sensor"
    drone_dim: int                # 2 | 3
    dt: float
    phase: str
    epoch: int

    # plotting / layout
    layout: str = "two_columns"
    only_position: bool = False

    # rollouts
    num_rollouts: int = 10

    # images
    render_images: bool = True
    num_steps: int = 10           # used for snapshots
    snapshots: bool = True
    snapshots_n_rows: int = 2
    label: str = "Model"

    # directories
    eval_dir: Optional[Path] = None   # can be set if you want; else passed to function


# -----------------------------------------------------------------------------
# Public entrypoint
# -----------------------------------------------------------------------------
def render_open_loop_rollouts(
        config: RenderOpenLoopConfig,
        output: VisionValForwardOutputs,
        eval_dir: Path,
        extract_one_rollout: Callable[[VisionValForwardOutputs, int], VisionValForwardOutputs],
) -> None:
    """
    Render open-loop rollouts with plots (and optionally images + snapshots).

    Args:
        config: RenderOpenLoopConfig containing rendering parameters.
        output: full output container (all rollouts).
        eval_dir: directory where results will be written.
        extract_one_rollout: function to select rollout k from `output`.
    """
    eval_dir.mkdir(parents=True, exist_ok=True)
    extractors = _make_extractors(config.modality)

    for k in range(config.num_rollouts):
        rollout_dir = eval_dir / f"rollout_{k}"
        rollout_dir.mkdir(parents=True, exist_ok=True)

        output_k = extract_one_rollout(output, k)

        # 1) State/input plots
        plotter = SingleStateInputPlotter(
            drone_dim=config.drone_dim,
            dt=config.dt,
            layout=config.layout,
            only_position=config.only_position,
            path=rollout_dir,
            extractors=extractors,
        )
        plotter.pipeline(output_k)

        plotter_2 = BasicStatePlotter(
            drone_dim=config.drone_dim,
            dt=config.dt,
            path=rollout_dir,
            x_gt = to_numpy(output_k.g_t.x_data)
        )
        plotter_2.pipeline(output_k)

        if config.render_images:
            _render_rollout_images(
                output_k,
                rollout_dir=rollout_dir,
                drone_dim=config.drone_dim
            )

            if config.snapshots:
                left_dir = rollout_dir / "images" / "left"
                out_path = rollout_dir / "snapshots.png"
                create_snapshots_figure(
                    im_dir=left_dir,
                    out_path=out_path,
                    step_count=config.num_steps,
                    label=config.label,
                    n_rows=config.snapshots_n_rows,
                )


# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------
def _render_rollout_images(
        output_k: VisionValForwardOutputs,
        rollout_dir: Path,
        drone_dim: int
) -> None:
    if drone_dim == 2:
        _render_rollout_images_2d(output_k, rollout_dir)
    elif drone_dim == 3:
        _render_rollout_images_3d(output_k, rollout_dir)
    else:
        raise ValueError(f"Invalid drone dimension: {drone_dim}")


def _render_rollout_images_2d(
        output_k: VisionValForwardOutputs,
        rollout_dir: Path
) -> None:

    pred_imgs, gt_imgs = extract_open_loop_pred_gt_imgs_2d(output_k)
    # Important: keep same folder structure as 3D for downstream code (snapshots, etc.)
    out_dir_left = rollout_dir / "images" / "left"
    out_dir_left.mkdir(parents=True, exist_ok=True)

    render_trajectory_sequence(
        pred_imgs=pred_imgs,
        gt_imgs=gt_imgs,
        path_results=out_dir_left,
    )


def _render_rollout_images_3d(
        output_k: VisionValForwardOutputs,
        rollout_dir: Path
) -> None:
    pred_left, gt_left, pred_right, gt_right = extract_open_loop_pred_gt_imgs_3d(output_k)

    out_dir_left = rollout_dir / "images" / "left"
    out_dir_left.mkdir(parents=True, exist_ok=True)
    render_trajectory_sequence(
        pred_imgs=pred_left,
        gt_imgs=gt_left,
        path_results=out_dir_left
    )
    out_dir_right = rollout_dir / "images" / "right"
    out_dir_right.mkdir(parents=True, exist_ok=True)
    render_trajectory_sequence(
        pred_imgs=pred_right,
        gt_imgs=gt_right,
        path_results=out_dir_right
    )


def _make_extractors(modality: str) -> SinglePlotExtractors:
    """Select extractor functions based on modality, keeping the plotter class agnostic."""
    if modality == "vision":
        return SinglePlotExtractors(
            get_x_gt=lambda out: to_numpy(out.g_t.state),
            get_x_pred=lambda out: to_numpy(out.pred.state),
            get_u=lambda out: to_numpy(out.inputs_physical),
        )

    elif modality == "sensor":
        return SinglePlotExtractors(
            get_x_gt=lambda out: to_numpy(out.state_gt_physical),
            get_x_pred=lambda out: to_numpy(out.pred.state),
            get_u=lambda out: to_numpy(out.inputs_physical),
        )
    else:
        raise ValueError(f"Unknown modality: {modality}")