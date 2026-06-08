from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from .baselines import behavior_distributed, rl_guided_behavior_score
from .formation import adaptive_radius, circular_formation_targets
from .model import Agent, Task
from .neural_q import NeuralQScorer
from .scenario import Scenario, generate_scenario
from .utility import distance, utility


Position = Tuple[float, float]
WORLD_EXTENT = (0.0, 100.0, -40.0, 50.0)
PLOT_FORMATS = ("png", "pdf")


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.0,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def run_vision_explainability(output_dir: Path, seed: int = 77) -> Dict[str, float | str]:
    """Generate visual-perception explainability figures for the paper."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario = generate_scenario(seed=seed, scenario_id=0, regime="standard")
    tracking_scenario = generate_scenario(seed=seed + 19, scenario_id=1, regime="stress")

    summary: Dict[str, float | str] = {}
    summary.update(_plot_confidence_allocation(scenario, output_dir, seed))
    summary.update(_plot_tracking_radius(tracking_scenario, output_dir, seed + 1))
    summary.update(_plot_attention_bidding(scenario, output_dir, seed + 2))

    with (output_dir / "vision_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _plot_confidence_allocation(scenario: Scenario, output_dir: Path, seed: int) -> Dict[str, float | str]:
    rng = np.random.default_rng(seed)
    x_grid, y_grid, confidence = _task_surface(
        scenario.search_tasks,
        lambda task: task.target_probability,
        sigma=7.4,
        noise=0.012,
        rng=rng,
    )
    _, _, information = _task_surface(
        scenario.search_tasks,
        lambda task: task.information_gain,
        sigma=8.6,
        noise=0.010,
        rng=rng,
    )
    ocean = _ocean_frame(x_grid, y_grid, rng)
    assignment = behavior_distributed(scenario.agents, scenario.search_tasks)
    task_lookup = {task.task_id: task for task in scenario.search_tasks}

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.35))
    _show_ocean(axes[0], ocean, "Simulated UAV visual frame")
    _draw_candidates(axes[0], scenario.search_tasks, scenario.true_search_task_id)
    _draw_agents(axes[0], scenario.agents)
    axes[0].scatter(
        [scenario.target_position[0]],
        [scenario.target_position[1]],
        marker="x",
        s=34,
        linewidths=1.2,
        color="#b21f2d",
        label="Target cue",
        zorder=5,
    )
    axes[0].legend(frameon=True, loc="lower left", fontsize=7.2)

    im_conf = axes[1].imshow(
        confidence,
        extent=WORLD_EXTENT,
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",
    )
    _draw_candidates(axes[1], scenario.search_tasks, scenario.true_search_task_id)
    _setup_map_axis(axes[1], "Target-confidence heatmap")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    cbar_conf = fig.colorbar(
        im_conf,
        ax=axes[1],
        orientation="horizontal",
        fraction=0.050,
        pad=0.105,
        shrink=0.82,
        aspect=24,
    )
    cbar_conf.set_label("Confidence", fontsize=8.0, labelpad=1.5)
    cbar_conf.ax.tick_params(labelsize=7.2, length=2.4, width=0.6)

    im_info = axes[2].imshow(
        information,
        extent=WORLD_EXTENT,
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",
    )
    _draw_candidates(axes[2], scenario.search_tasks, scenario.true_search_task_id)
    _draw_agents(axes[2], scenario.agents)
    for agent in scenario.agents:
        task_id = assignment.get(agent.agent_id)
        if task_id is None:
            continue
        task = task_lookup[task_id]
        axes[2].annotate(
            "",
            xy=task.position,
            xytext=agent.position,
            arrowprops={
                "arrowstyle": "->",
                "lw": 1.0,
                "color": "#202020",
                "alpha": 0.78,
                "shrinkA": 3,
                "shrinkB": 3,
            },
            zorder=4,
        )
        axes[2].text(
            task.position[0] + 1.0,
            task.position[1] + 1.0,
            agent.agent_id.replace("_", ""),
            fontsize=7.2,
            color="#101010",
            zorder=5,
    )
    _setup_map_axis(axes[2], "Information gain and assigned regions")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("")
    cbar_info = fig.colorbar(
        im_info,
        ax=axes[2],
        orientation="horizontal",
        fraction=0.050,
        pad=0.105,
        shrink=0.82,
        aspect=24,
    )
    cbar_info.set_label("Normalized information gain", fontsize=8.0, labelpad=1.5)
    cbar_info.ax.tick_params(labelsize=7.2, length=2.4, width=0.6)
    fig.subplots_adjust(left=0.055, right=0.990, top=0.905, bottom=0.205, wspace=0.235)
    _save_figure(fig, output_dir, "vision_confidence_allocation")
    plt.close(fig)

    ranked_by_confidence = sorted(scenario.search_tasks, key=lambda task: task.target_probability, reverse=True)
    ranked_by_gain = sorted(scenario.search_tasks, key=lambda task: task.information_gain, reverse=True)
    true_task = task_lookup[scenario.true_search_task_id]
    assigned_true = [agent_id for agent_id, task_id in assignment.items() if task_id == scenario.true_search_task_id]
    return {
        "confidence_top_region": ranked_by_confidence[0].task_id,
        "information_gain_top_region": ranked_by_gain[0].task_id,
        "true_region": scenario.true_search_task_id,
        "true_region_confidence": round(true_task.target_probability, 4),
        "true_region_information_gain": round(true_task.information_gain, 4),
        "true_region_assigned_agent": assigned_true[0] if assigned_true else "none",
    }


def _plot_tracking_radius(scenario: Scenario, output_dir: Path, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    times = np.arange(0, 9, dtype=float)
    heading = 0.62
    step_scale = 3.15
    unit = np.asarray([np.cos(heading), np.sin(heading)])
    cross = np.asarray([-np.sin(heading), np.cos(heading)])
    start = np.asarray(scenario.target_position, dtype=float)
    wobble = np.sin(times * 0.85)[:, None] * cross[None, :] * 1.4
    true_track = start[None, :] + scenario.target_speed * step_scale * times[:, None] * unit[None, :] + wobble

    view_specs = [
        ("UAV nadir", 1.35, np.asarray([0.35, -0.25]), "#1666b0"),
        ("UAV oblique", 2.20, np.asarray([1.10, -0.80]), "#7a52cc"),
        ("USV close", 1.05, np.asarray([-0.55, 0.45]), "#1e9d74"),
    ]
    observations: Dict[str, np.ndarray] = {}
    weights = []
    for label, noise_scale, bias, _ in view_specs:
        noise = rng.normal(0.0, noise_scale, size=true_track.shape)
        observations[label] = true_track + bias[None, :] + noise
        weights.append(1.0 / (noise_scale**2))
    weight_arr = np.asarray(weights, dtype=float)
    fused = sum(weight_arr[idx] * observations[label] for idx, (label, _, _, _) in enumerate(view_specs)) / np.sum(weight_arr)
    speed_est = np.zeros_like(times)
    speed_est[1:] = np.linalg.norm(np.diff(fused, axis=0), axis=1) / step_scale
    speed_est[0] = speed_est[1]
    sea_state = scenario.sea_state + 0.18 * np.sin(times * 0.7)
    radii = 25.0 + 3.0 * speed_est + 1.5 * sea_state
    final_radius = float(radii[-1])
    fixed_radius = 25.0
    final_slots = circular_formation_targets(
        center=tuple(fused[-1]),
        num_agents=4,
        base_radius=25.0,
        target_speed=float(speed_est[-1]),
        sea_state=float(sea_state[-1]),
    )

    fig = plt.figure(figsize=(13.3, 4.25))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.14, 1.0, 1.02], wspace=0.36)
    ax_track = fig.add_subplot(grid[0, 0])
    ax_radius = fig.add_subplot(grid[0, 1])
    ax_slots = fig.add_subplot(grid[0, 2])

    ax_track.plot(true_track[:, 0], true_track[:, 1], color="#202020", linewidth=2.0, label="True drift")
    ax_track.plot(fused[:, 0], fused[:, 1], color="#c84b39", linewidth=2.0, linestyle="--", label="Fused estimate")
    for label, _, _, color in view_specs:
        obs = observations[label]
        ax_track.scatter(obs[:, 0], obs[:, 1], s=14, alpha=0.52, color=color, label=label)
    ax_track.annotate(
        "",
        xy=fused[-1],
        xytext=fused[-3],
        arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#c84b39"},
    )
    _setup_map_axis(ax_track, "Multi-view target tracking")
    _set_dynamic_map_limits(ax_track, [true_track, fused, *observations.values()], pad=8.0)
    ax_track.legend(frameon=False, loc="upper left", fontsize=7.0)

    ax_radius.plot(times, radii, color="#1666b0", linewidth=2.0, marker="o", label="Adaptive radius")
    ax_radius.axhline(fixed_radius, color="#8c8c8c", linewidth=1.4, linestyle="--", label="Fixed radius")
    true_speed_radius = 25.0 + 3.0 * np.full_like(times, scenario.target_speed) + 1.5 * sea_state
    ax_radius.plot(
        times,
        true_speed_radius,
        color="#5f8d3a",
        linewidth=1.5,
        linestyle=":",
        label="Radius from true speed",
    )
    ax_radius.set_xlabel("Frame index")
    ax_radius.set_ylabel("Encircling radius")
    _setup_plain_axis(ax_radius, "Drift estimate drives radius")
    ax_radius.set_xlim(-0.35, 9.25)
    ax_radius.set_ylim(24.2, max(43.3, float(np.max(radii)) + 0.7))
    ax_radius.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#d4d4d4",
        framealpha=0.92,
        loc="center",
        bbox_to_anchor=(0.56, 0.42),
        fontsize=7.0,
        borderpad=0.28,
        handlelength=1.8,
    )

    ax_slots.plot(true_track[:, 0], true_track[:, 1], color="#202020", linewidth=1.5, alpha=0.65)
    ax_slots.scatter([fused[-1, 0]], [fused[-1, 1]], color="#c84b39", s=42, zorder=4, label="Final estimate")
    ax_slots.add_patch(Circle(tuple(fused[-1]), fixed_radius, fill=False, linestyle="--", linewidth=1.2, edgecolor="#8c8c8c"))
    ax_slots.add_patch(Circle(tuple(fused[-1]), final_radius, fill=False, linewidth=2.0, edgecolor="#1666b0"))
    for idx, slot in enumerate(final_slots):
        ax_slots.scatter([slot[0]], [slot[1]], marker="s", s=42, color="#1666b0", zorder=4)
        ax_slots.text(slot[0] + 0.7, slot[1] + 0.7, f"slot {idx + 1}", fontsize=7.2)
    ax_slots.text(
        fused[-1, 0] + 1.0,
        fused[-1, 1] - 4.2,
        f"r*={final_radius:.1f}",
        fontsize=8.2,
        color="#1666b0",
    )
    _setup_map_axis(ax_slots, "Adaptive protection geometry")
    ax_slots.set_ylabel("")
    circle_extreme_points = np.asarray(
        [
            [fused[-1, 0] + final_radius, fused[-1, 1]],
            [fused[-1, 0] - final_radius, fused[-1, 1]],
            [fused[-1, 0], fused[-1, 1] + final_radius],
            [fused[-1, 0], fused[-1, 1] - final_radius],
        ]
    )
    _set_dynamic_map_limits(ax_slots, [true_track, fused, np.asarray(final_slots), circle_extreme_points], pad=7.0)
    ax_slots.legend(frameon=False, loc="upper left", fontsize=7.2)

    _save_figure(fig, output_dir, "vision_tracking_radius")
    plt.close(fig)

    mean_tracking_error = float(np.mean(np.linalg.norm(fused - true_track, axis=1)))
    return {
        "tracking_final_radius": round(final_radius, 4),
        "tracking_radius_increase_over_fixed": round(final_radius - fixed_radius, 4),
        "tracking_final_speed_estimate": round(float(speed_est[-1]), 4),
        "tracking_mean_fusion_error": round(mean_tracking_error, 4),
    }


def _plot_attention_bidding(scenario: Scenario, output_dir: Path, seed: int) -> Dict[str, float | str]:
    rng = np.random.default_rng(seed)
    x_grid, y_grid, confidence = _task_surface(
        scenario.search_tasks,
        lambda task: task.target_probability,
        sigma=7.0,
        noise=0.012,
        rng=rng,
    )
    _, _, information = _task_surface(
        scenario.search_tasks,
        lambda task: task.information_gain,
        sigma=8.0,
        noise=0.010,
        rng=rng,
    )
    ocean = _ocean_frame(x_grid, y_grid, rng)
    uavs = [agent for agent in scenario.agents if agent.kind == "UAV"]
    attention_maps = []
    for idx, agent in enumerate(uavs[:2]):
        visibility = _visibility_from_agent(x_grid, y_grid, agent.position)
        saliency = 0.58 * confidence + 0.24 * information + 0.18 * visibility
        saliency += rng.normal(0.0, 0.018 + 0.006 * idx, size=saliency.shape)
        attention_maps.append(_normalize(saliency))
    mean_attention = np.mean(attention_maps, axis=0) if attention_maps else confidence
    task_attention = {task.task_id: _sample_surface(mean_attention, x_grid, y_grid, task.position) for task in scenario.search_tasks}

    scorer = NeuralQScorer.load()
    bid_matrix = np.asarray(
        [
            [rl_guided_behavior_score(agent, task, scorer) for task in scenario.search_tasks]
            for agent in scenario.agents
        ],
        dtype=float,
    )
    assignment = behavior_distributed(scenario.agents, scenario.search_tasks)
    assigned_tasks = set(assignment.values())
    max_bid_by_task = bid_matrix.max(axis=0)
    attention_values = np.asarray([task_attention[task.task_id] for task in scenario.search_tasks], dtype=float)
    corr = float(np.corrcoef(attention_values, max_bid_by_task)[0, 1])

    _write_region_table(scenario, output_dir / "vision_region_table.csv", task_attention, max_bid_by_task, assignment)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    _show_ocean(axes[0], ocean, "Visual attention overlay")
    axes[0].imshow(mean_attention, extent=WORLD_EXTENT, origin="lower", cmap="inferno", alpha=0.55, vmin=0.0, vmax=1.0)
    _draw_candidates(axes[0], scenario.search_tasks, scenario.true_search_task_id)
    for task in scenario.search_tasks:
        value = task_attention[task.task_id]
        axes[0].text(task.position[0] + 1.0, task.position[1] - 3.4, f"A={value:.2f}", fontsize=7.0, color="#ffffff")

    im_bid = axes[1].imshow(bid_matrix, cmap="YlGnBu", aspect="auto")
    axes[1].set_xticks(range(len(scenario.search_tasks)), [task.task_id.replace("SEARCH_", "S") for task in scenario.search_tasks])
    axes[1].set_yticks(range(len(scenario.agents)), [agent.agent_id.replace("_", "") for agent in scenario.agents])
    axes[1].set_title("RL-guided task-bid matrix")
    for row_idx in range(bid_matrix.shape[0]):
        for col_idx in range(bid_matrix.shape[1]):
            text_color = "#ffffff" if bid_matrix[row_idx, col_idx] > np.median(bid_matrix) else "#202020"
            axes[1].text(col_idx, row_idx, f"{bid_matrix[row_idx, col_idx]:.1f}", ha="center", va="center", fontsize=6.8, color=text_color)
    fig.colorbar(im_bid, ax=axes[1], fraction=0.046, pad=0.02, label="Bid")
    _setup_matrix_axis(axes[1])

    axes[2].scatter(attention_values, max_bid_by_task, s=70, color="#1666b0", edgecolor="#202020", linewidth=0.6)
    for idx, task in enumerate(scenario.search_tasks):
        label = task.task_id.replace("SEARCH_", "S")
        color = "#b21f2d" if task.task_id == scenario.true_search_task_id else "#202020"
        weight = "bold" if task.task_id in assigned_tasks else "normal"
        axes[2].text(attention_values[idx] + 0.008, max_bid_by_task[idx], label, fontsize=8.0, color=color, weight=weight)
    slope, intercept = np.polyfit(attention_values, max_bid_by_task, deg=1)
    line_x = np.linspace(float(attention_values.min()), float(attention_values.max()), 40)
    axes[2].plot(line_x, slope * line_x + intercept, color="#c84b39", linewidth=1.6)
    axes[2].set_xlabel("Candidate-box visual attention")
    axes[2].set_ylabel("Maximum task bid")
    _setup_plain_axis(axes[2], f"Attention-bid link (r={corr:.2f})")

    fig.subplots_adjust(wspace=0.28)
    _save_figure(fig, output_dir, "vision_attention_bidding")
    plt.close(fig)

    attention_ranked = sorted(scenario.search_tasks, key=lambda task: task_attention[task.task_id], reverse=True)
    bid_ranked = sorted(
        scenario.search_tasks,
        key=lambda task: max_bid_by_task[scenario.search_tasks.index(task)],
        reverse=True,
    )
    return {
        "attention_bid_correlation": round(corr, 4),
        "attention_top_region": attention_ranked[0].task_id,
        "bid_top_region": bid_ranked[0].task_id,
    }


def _task_surface(
    tasks: Sequence[Task],
    value_fn,
    *,
    sigma: float,
    noise: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_grid = np.linspace(WORLD_EXTENT[0], WORLD_EXTENT[1], 220)
    y_grid = np.linspace(WORLD_EXTENT[2], WORLD_EXTENT[3], 180)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    surface = np.zeros_like(x_mesh, dtype=float)
    for task in tasks:
        value = float(value_fn(task))
        dx = x_mesh - task.position[0]
        dy = y_mesh - task.position[1]
        surface += value * np.exp(-0.5 * (dx * dx + dy * dy) / (sigma * sigma))
    if noise:
        surface += rng.normal(0.0, noise, size=surface.shape)
    return x_grid, y_grid, _normalize(surface)


def _ocean_frame(x_grid: np.ndarray, y_grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    waves = 0.45 + 0.13 * np.sin(0.20 * x_mesh + 0.55 * np.sin(0.12 * y_mesh))
    waves += 0.08 * np.sin(0.33 * y_mesh + 0.05 * x_mesh)
    glare = np.exp(-((y_mesh - 18.0 - 0.20 * (x_mesh - 54.0)) ** 2) / (2.0 * 4.0**2))
    noise = rng.normal(0.0, 0.025, size=waves.shape)
    texture = _normalize(waves + 0.45 * glare + noise)
    frame = np.zeros((len(y_grid), len(x_grid), 3), dtype=float)
    frame[..., 0] = 0.035 + 0.08 * texture + 0.12 * glare
    frame[..., 1] = 0.22 + 0.30 * texture + 0.16 * glare
    frame[..., 2] = 0.34 + 0.42 * texture + 0.16 * glare
    return np.clip(frame, 0.0, 1.0)


def _visibility_from_agent(x_grid: np.ndarray, y_grid: np.ndarray, position: Position) -> np.ndarray:
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    dx = x_mesh - position[0]
    dy = y_mesh - position[1]
    view = np.exp(-0.5 * (dx * dx + dy * dy) / (34.0**2))
    scan = 0.5 + 0.5 * np.cos(np.arctan2(dy, dx) - 0.35)
    return _normalize(0.72 * view + 0.28 * scan)


def _normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    min_value = float(np.min(arr))
    max_value = float(np.max(arr))
    if max_value - min_value < 1e-12:
        return np.zeros_like(arr)
    return (arr - min_value) / (max_value - min_value)


def _sample_surface(surface: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray, position: Position) -> float:
    x_idx = int(np.argmin(np.abs(x_grid - position[0])))
    y_idx = int(np.argmin(np.abs(y_grid - position[1])))
    y0 = max(0, y_idx - 5)
    y1 = min(surface.shape[0], y_idx + 6)
    x0 = max(0, x_idx - 5)
    x1 = min(surface.shape[1], x_idx + 6)
    return float(np.mean(surface[y0:y1, x0:x1]))


def _show_ocean(axis, ocean: np.ndarray, title: str) -> None:
    axis.imshow(ocean, extent=WORLD_EXTENT, origin="lower", interpolation="bilinear")
    _setup_map_axis(axis, title)


def _draw_candidates(axis, tasks: Sequence[Task], true_task_id: str) -> None:
    for task in tasks:
        x0 = task.position[0] - 5.8
        y0 = task.position[1] - 4.8
        is_true = task.task_id == true_task_id
        edge = "#f7f7f7" if not is_true else "#b21f2d"
        linewidth = 1.0 if not is_true else 1.7
        axis.add_patch(Rectangle((x0, y0), 11.6, 9.6, fill=False, edgecolor=edge, linewidth=linewidth, zorder=4))
        axis.text(
            task.position[0] - 4.7,
            task.position[1] + 5.7,
            task.task_id.replace("SEARCH_", "S"),
            fontsize=7.4,
            color=edge,
            weight="bold" if is_true else "normal",
            zorder=5,
        )


def _draw_agents(axis, agents: Sequence[Agent]) -> None:
    for agent in agents:
        marker = "^" if agent.kind == "UAV" else "s"
        color = "#f2c84b" if agent.kind == "UAV" else "#72c7b0"
        axis.scatter(
            [agent.position[0]],
            [agent.position[1]],
            marker=marker,
            s=42,
            color=color,
            edgecolor="#202020",
            linewidth=0.6,
            zorder=6,
        )
        axis.text(agent.position[0] + 0.9, agent.position[1] + 0.9, agent.agent_id.replace("_", ""), fontsize=7.0, color="#101010", zorder=6)


def _setup_map_axis(axis, title: str) -> None:
    axis.set_title(title, weight="bold", pad=5)
    axis.set_xlim(WORLD_EXTENT[0], WORLD_EXTENT[1])
    axis.set_ylim(WORLD_EXTENT[2], WORLD_EXTENT[3])
    axis.set_xlabel("East position")
    axis.set_ylabel("North position")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _set_dynamic_map_limits(axis, point_sets: Sequence[np.ndarray], pad: float = 6.0) -> None:
    points = np.vstack([np.asarray(point_set, dtype=float).reshape(-1, 2) for point_set in point_sets])
    xmin, ymin = np.min(points, axis=0)
    xmax, ymax = np.max(points, axis=0)
    span = max(float(xmax - xmin), float(ymax - ymin), 1.0)
    extra = max(pad, 0.08 * span)
    axis.set_xlim(float(xmin - extra), float(xmax + extra))
    axis.set_ylim(float(ymin - extra), float(ymax + extra))


def _setup_plain_axis(axis, title: str) -> None:
    axis.set_title(title, weight="bold", pad=5)
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="both", direction="out", length=3.0, width=0.7)


def _setup_matrix_axis(axis) -> None:
    axis.tick_params(axis="x", rotation=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _write_region_table(
    scenario: Scenario,
    csv_path: Path,
    task_attention: Dict[str, float],
    max_bid_by_task: np.ndarray,
    assignment: Dict[str, str],
) -> None:
    assigned_agent_by_task = {task_id: agent_id for agent_id, task_id in assignment.items()}
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task_id",
                "is_true_region",
                "target_probability",
                "information_gain",
                "visual_attention",
                "max_bid",
                "assigned_agent",
            ],
        )
        writer.writeheader()
        for idx, task in enumerate(scenario.search_tasks):
            writer.writerow(
                {
                    "task_id": task.task_id,
                    "is_true_region": task.task_id == scenario.true_search_task_id,
                    "target_probability": f"{task.target_probability:.4f}",
                    "information_gain": f"{task.information_gain:.4f}",
                    "visual_attention": f"{task_attention[task.task_id]:.4f}",
                    "max_bid": f"{max_bid_by_task[idx]:.4f}",
                    "assigned_agent": assigned_agent_by_task.get(task.task_id, ""),
                }
            )


def _save_figure(fig, output_dir: Path, stem: str) -> None:
    for fmt in PLOT_FORMATS:
        fig.savefig(output_dir / f"{stem}.{fmt}")
