import csv
import inspect
import json
from dataclasses import asdict
from math import comb
from pathlib import Path
from random import Random
from statistics import mean, median, pstdev
from typing import Dict, List, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .evaluation import ABLATIONS, METHODS, evaluate_method
from .scenario import generate_scenario


METRIC_FIELDS = [
    "detection_time",
    "verification_time",
    "encirclement_time",
    "mission_time",
    "total_path_length",
    "utility_score",
    "formation_error",
    "rescue_success",
    "communication_load",
]

METHOD_LABELS = {
    "behavior_distributed": "Proposed",
    "cbba": "CBBA-style",
    "rl_contextual_bandit": "RL-bandit",
    "neural_q": "Neural-Q",
    "homogeneous_distributed": "Homogeneous",
    "distance_only": "Distance-only",
    "centralized_optimal": "Centralized",
}

METHOD_COLORS = {
    "behavior_distributed": "#1666b0",
    "cbba": "#7a52cc",
    "rl_contextual_bandit": "#5f8d3a",
    "neural_q": "#b75aa1",
    "homogeneous_distributed": "#1e9d74",
    "distance_only": "#f0a11e",
    "centralized_optimal": "#c84b39",
}

HETEROGENEITY_LEVELS = [0.0, 0.5, 1.0, 1.5]
SCALABILITY_CONFIGS = [
    {"label": "compact", "num_uavs": 1, "num_usvs": 3, "num_search_tasks": 4},
    {"label": "nominal", "num_uavs": 2, "num_usvs": 4, "num_search_tasks": 6},
    {"label": "expanded", "num_uavs": 3, "num_usvs": 5, "num_search_tasks": 8},
    {"label": "dense", "num_uavs": 3, "num_usvs": 6, "num_search_tasks": 8},
]
PLOT_DPI = 300
PLOT_FORMATS = ("png", "pdf")

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#2f2f2f",
        "axes.linewidth": 0.7,
        "axes.grid": False,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.dpi": PLOT_DPI,
        "savefig.dpi": PLOT_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "stix",
    }
)


def _base_methods(rows: Sequence[Dict[str, float]]) -> List[str]:
    return _ordered_methods({row["method"] for row in rows if "__" not in row["method"]})


def _ordered_methods(methods: Sequence[str]) -> List[str]:
    order = [
        "behavior_distributed",
        "cbba",
        "rl_contextual_bandit",
        "neural_q",
        "centralized_optimal",
        "distance_only",
        "homogeneous_distributed",
    ]
    return [method for method in order if method in methods]


def _scenario_rows(rows: Sequence[Dict[str, float]], method: str) -> List[Dict[str, float]]:
    return sorted(
        [row for row in rows if row["method"] == method],
        key=lambda row: (int(row["scenario_id"]), row["regime"], float(row.get("heterogeneity_scale", 1.0))),
    )


def _bootstrap_ci(values: Sequence[float], iters: int = 2000, seed: int = 11) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = Random(seed)
    n = len(values)
    samples = []
    for _ in range(iters):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(mean(draw))
    samples.sort()
    low_idx = int(0.025 * (iters - 1))
    high_idx = int(0.975 * (iters - 1))
    return {
        "mean": mean(values),
        "ci_low": samples[low_idx],
        "ci_high": samples[high_idx],
    }


def _two_sided_sign_test(values: Sequence[float]) -> float:
    nonzero = [value for value in values if abs(value) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return 1.0
    positive = sum(value > 0.0 for value in nonzero)
    tail = sum(comb(n, k) for k in range(0, min(positive, n - positive) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def run_batch(
    num_scenarios: int,
    output_dir: Path,
    seed: int = 42,
    regime: str = "standard",
    include_ablations: bool = False,
    heterogeneity_scale: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, float]] = []
    for scenario_idx in range(num_scenarios):
        scenario = generate_scenario(
            seed=seed + scenario_idx,
            scenario_id=scenario_idx,
            regime=regime,
            heterogeneity_scale=heterogeneity_scale,
        )
        for method in METHODS:
            rows.append(asdict(evaluate_method(scenario, method)))
        if include_ablations:
            for ablation in [name for name in ABLATIONS if name != "full"]:
                rows.append(asdict(evaluate_method(scenario, "behavior_distributed", ablation=ablation)))

    _write_rows(rows, output_dir / "results.csv")

    summary = summarize(rows)
    _write_json(summary, output_dir / "summary.json")

    difficulty_summary = summarize_by_difficulty(rows)
    _write_json(difficulty_summary, output_dir / "difficulty_summary.json")

    significance = compute_significance(rows)
    _write_json(significance, output_dir / "significance_summary.json")

    distribution_summary = summarize_distributions(rows)
    _write_json(distribution_summary, output_dir / "distribution_summary.json")

    win_summary = summarize_wins(rows)
    _write_json(win_summary, output_dir / "wins_summary.json")

    sensitivity_summary = summarize_sensitivity(rows)
    _write_json(sensitivity_summary, output_dir / "sensitivity_summary.json")

    regime_summary = summarize_by_regime(rows)
    _write_json(regime_summary, output_dir / "regime_summary.json")

    ablation_summary = summarize_ablation(rows)
    _write_json(ablation_summary, output_dir / "ablation_summary.json")

    export_latex_tables(
        summary=summary,
        difficulty_summary=difficulty_summary,
        regime_summary=regime_summary,
        ablation_summary=ablation_summary,
        distribution_summary=distribution_summary,
        win_summary=win_summary,
        significance_summary=significance,
        output_dir=output_dir,
    )

    plot_summary(summary, significance, output_dir)
    plot_distributions(rows, output_dir)
    plot_cdf(rows, output_dir, metric_key="mission_time", title="Mission-Time CDF", filename="mission_time_cdf.png")
    plot_sensitivity(rows, output_dir, x_key="sea_state", y_key="mission_time", xlabel="Sea-State Level", filename="mission_time_vs_sea_state.png")
    plot_sensitivity(rows, output_dir, x_key="communication_quality", y_key="mission_time", xlabel="Communication Quality", filename="mission_time_vs_communication.png")
    plot_wins(win_summary, output_dir)
    plot_ablation(ablation_summary, output_dir)
    plot_robustness_panel(rows, win_summary, output_dir)
    return summary


def run_heterogeneity_sweep(
    output_dir: Path,
    num_scenarios: int = 40,
    seed: int = 200,
    regime: str = "standard",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, float]] = []
    for level_idx, level in enumerate(HETEROGENEITY_LEVELS):
        for scenario_idx in range(num_scenarios):
            scenario = generate_scenario(
                seed=seed + scenario_idx,
                scenario_id=scenario_idx,
                regime=regime,
                heterogeneity_scale=level,
            )
            for method in METHODS:
                rows.append(asdict(evaluate_method(scenario, method)))

    _write_rows(rows, output_dir / "results.csv")
    summary = summarize_by_heterogeneity(rows)
    _write_json(summary, output_dir / "heterogeneity_summary.json")
    export_heterogeneity_table(summary, output_dir / "heterogeneity_table.tex")
    plot_heterogeneity(summary, output_dir)
    return summary


def run_scalability_sweep(
    output_dir: Path,
    num_scenarios: int = 24,
    seed: int = 500,
    regime: str = "standard",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, float]] = []
    for config_idx, config in enumerate(SCALABILITY_CONFIGS):
        for scenario_idx in range(num_scenarios):
            scenario = generate_scenario(
                seed=seed + 1000 * config_idx + scenario_idx,
                scenario_id=scenario_idx,
                num_uavs=config["num_uavs"],
                num_usvs=config["num_usvs"],
                num_search_tasks=config["num_search_tasks"],
                regime=regime,
            )
            for method in METHODS:
                row = asdict(evaluate_method(scenario, method))
                row["scale_label"] = config["label"]
                row["num_uavs"] = config["num_uavs"]
                row["num_usvs"] = config["num_usvs"]
                row["num_search_tasks"] = config["num_search_tasks"]
                rows.append(row)

    _write_rows(rows, output_dir / "results.csv")
    summary = summarize_by_scale(rows)
    _write_json(summary, output_dir / "scalability_summary.json")
    export_scalability_table(summary, output_dir / "scalability_table.tex")
    plot_scalability(summary, output_dir)
    return summary


def _write_rows(rows: Sequence[Dict[str, float]], csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(payload: Dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def summarize(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for method in _ordered_methods({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        summary[method] = {
            metric: mean(float(row[metric]) for row in method_rows)
            for metric in METRIC_FIELDS
        }
    return summary


def summarize_by_difficulty(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    methods = _ordered_methods({row["method"] for row in rows})
    difficulties = sorted({row["difficulty"] for row in rows})
    for difficulty in difficulties:
        summary[difficulty] = {}
        for method in methods:
            method_rows = [row for row in rows if row["method"] == method and row["difficulty"] == difficulty]
            if not method_rows:
                continue
            summary[difficulty][method] = {
                metric: mean(float(row[metric]) for row in method_rows)
                for metric in METRIC_FIELDS
            }
    return summary


def summarize_by_regime(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    methods = _ordered_methods({row["method"] for row in rows})
    regimes = sorted({row["regime"] for row in rows})
    for regime in regimes:
        summary[regime] = {}
        for method in methods:
            method_rows = [row for row in rows if row["method"] == method and row["regime"] == regime]
            if not method_rows:
                continue
            summary[regime][method] = {
                metric: mean(float(row[metric]) for row in method_rows)
                for metric in METRIC_FIELDS
            }
    return summary


def summarize_ablation(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    targets = [row for row in rows if row["method"].startswith("behavior_distributed__")]
    methods = sorted({row["method"] for row in targets})
    summary: Dict[str, Dict[str, float]] = {}
    for method in methods:
        method_rows = [row for row in targets if row["method"] == method]
        summary[method] = {
            metric: mean(float(row[metric]) for row in method_rows)
            for metric in METRIC_FIELDS
        }
    return summary


def summarize_distributions(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    metrics = ["mission_time", "rescue_success", "formation_error"]
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method in _base_methods(rows):
        method_rows = [row for row in rows if row["method"] == method]
        summary[method] = {}
        for metric in metrics:
            values = sorted(float(row[metric]) for row in method_rows)
            count = len(values)
            if count == 0:
                continue
            q1 = values[max(0, int(0.25 * (count - 1)))]
            q3 = values[max(0, int(0.75 * (count - 1)))]
            summary[method][metric] = {
                "mean": mean(values),
                "median": median(values),
                "min": values[0],
                "max": values[-1],
                "q1": q1,
                "q3": q3,
            }
    return summary


def summarize_wins(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    relevant_rows = [row for row in rows if "__" not in row["method"]]
    scenario_ids = sorted({int(row["scenario_id"]) for row in relevant_rows})
    methods = _base_methods(relevant_rows)
    win_counts = {method: 0 for method in methods}
    rescue_wins = {method: 0 for method in methods}

    for scenario_id in scenario_ids:
        scenario_rows = [row for row in relevant_rows if int(row["scenario_id"]) == scenario_id]
        best_time = min(scenario_rows, key=lambda row: float(row["mission_time"]))["method"]
        best_rescue = max(scenario_rows, key=lambda row: float(row["rescue_success"]))["method"]
        win_counts[best_time] += 1
        rescue_wins[best_rescue] += 1

    total = max(len(scenario_ids), 1)
    return {
        method: {
            "mission_time_wins": float(win_counts[method]),
            "mission_time_win_rate": win_counts[method] / total,
            "rescue_success_wins": float(rescue_wins[method]),
            "rescue_success_win_rate": rescue_wins[method] / total,
        }
        for method in methods
    }


def summarize_sensitivity(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    relevant_rows = [row for row in rows if "__" not in row["method"]]
    specs = {
        "sea_state": [(0.0, 2.0, "low"), (2.0, 3.5, "moderate"), (3.5, 10.0, "high")],
        "communication_quality": [(0.0, 0.45, "poor"), (0.45, 0.7, "degraded"), (0.7, 1.01, "strong")],
    }
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for key, bins in specs.items():
        summary[key] = {}
        for lower, upper, label in bins:
            bucket_rows = [row for row in relevant_rows if lower <= float(row[key]) < upper]
            if not bucket_rows:
                continue
            summary[key][label] = {}
            for method in _base_methods(bucket_rows):
                method_rows = [row for row in bucket_rows if row["method"] == method]
                summary[key][label][method] = {
                    "mission_time": mean(float(row["mission_time"]) for row in method_rows),
                    "rescue_success": mean(float(row["rescue_success"]) for row in method_rows),
                    "formation_error": mean(float(row["formation_error"]) for row in method_rows),
                }
    return summary


def summarize_by_heterogeneity(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for level in sorted({float(row["heterogeneity_scale"]) for row in rows}):
        key = f"{level:.2f}"
        level_rows = [row for row in rows if abs(float(row["heterogeneity_scale"]) - level) < 1e-9]
        summary[key] = {}
        for method in _base_methods(level_rows):
            method_rows = [row for row in level_rows if row["method"] == method]
            summary[key][method] = {
                "mission_time": mean(float(row["mission_time"]) for row in method_rows),
                "rescue_success": mean(float(row["rescue_success"]) for row in method_rows),
                "formation_error": mean(float(row["formation_error"]) for row in method_rows),
            }
    return summary


def summarize_by_scale(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    labels = [config["label"] for config in SCALABILITY_CONFIGS]
    for label in labels:
        scale_rows = [row for row in rows if row["scale_label"] == label]
        if not scale_rows:
            continue
        representative = scale_rows[0]
        summary[label] = {}
        for method in _base_methods(scale_rows):
            method_rows = [row for row in scale_rows if row["method"] == method]
            summary[label][method] = {
                "mission_time": mean(float(row["mission_time"]) for row in method_rows),
                "rescue_success": mean(float(row["rescue_success"]) for row in method_rows),
                "formation_error": mean(float(row["formation_error"]) for row in method_rows),
                "communication_load": mean(float(row["communication_load"]) for row in method_rows),
                "num_uavs": float(representative["num_uavs"]),
                "num_usvs": float(representative["num_usvs"]),
                "num_search_tasks": float(representative["num_search_tasks"]),
            }
    return summary


def compute_significance(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    baseline_rows = _scenario_rows(rows, "behavior_distributed")
    for method in [item for item in _base_methods(rows) if item != "behavior_distributed"]:
        compare_rows = _scenario_rows(rows, method)
        result[method] = {}
        for metric in ["mission_time", "rescue_success"]:
            if metric == "mission_time":
                deltas = [float(other[metric]) - float(base[metric]) for base, other in zip(baseline_rows, compare_rows)]
            else:
                deltas = [float(base[metric]) - float(other[metric]) for base, other in zip(baseline_rows, compare_rows)]
            ci = _bootstrap_ci(deltas)
            result[method][f"{metric}_delta_mean"] = ci["mean"]
            result[method][f"{metric}_delta_ci_low"] = ci["ci_low"]
            result[method][f"{metric}_delta_ci_high"] = ci["ci_high"]
            result[method][f"{metric}_delta_std"] = pstdev(deltas) if len(deltas) > 1 else 0.0
            result[method][f"{metric}_sign_p"] = _two_sided_sign_test(deltas)
    return result


def export_latex_tables(
    summary: Dict[str, Dict[str, float]],
    difficulty_summary: Dict[str, Dict[str, Dict[str, float]]],
    regime_summary: Dict[str, Dict[str, Dict[str, float]]],
    ablation_summary: Dict[str, Dict[str, float]],
    distribution_summary: Dict[str, Dict[str, Dict[str, float]]],
    win_summary: Dict[str, Dict[str, float]],
    significance_summary: Dict[str, Dict[str, float]],
    output_dir: Path,
) -> None:
    main_table = output_dir / "results_table.tex"
    with main_table.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lcccc}\n")
        handle.write("\\toprule\n")
        handle.write("Method & Mission Time & Rescue Success & Path Len. & Comm.\\\\\n")
        handle.write("\\midrule\n")
        for method, metrics in summary.items():
            handle.write(
                f"{METHOD_LABELS.get(method, method)} & {metrics['mission_time']:.2f} & "
                f"{metrics['rescue_success']:.3f} & {metrics['total_path_length']:.2f} & "
                f"{metrics['communication_load']:.2f}\\\\\n"
            )
        handle.write("\\bottomrule\n\\end{tabular*}\n")

    distribution_table = output_dir / "distribution_table.tex"
    with distribution_table.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lccc}\n")
        handle.write("\\toprule\n")
        handle.write("Method & Median $T_{\\mathrm{mis}}$ & IQR $T_{\\mathrm{mis}}$ & Median $S_{\\mathrm{res}}$\\\\\n")
        handle.write("\\midrule\n")
        for method, metrics in distribution_summary.items():
            mission = metrics["mission_time"]
            rescue = metrics["rescue_success"]
            iqr = mission["q3"] - mission["q1"]
            handle.write(
                f"{METHOD_LABELS.get(method, method)} & {mission['median']:.2f} & "
                f"{iqr:.2f} & {rescue['median']:.3f}\\\\\n"
            )
        handle.write("\\bottomrule\n\\end{tabular*}\n")

    wins_table = output_dir / "wins_table.tex"
    with wins_table.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lcc}\n")
        handle.write("\\toprule\n")
        handle.write("Method & Mission-time wins & Rescue-success wins\\\\\n")
        handle.write("\\midrule\n")
        for method, metrics in win_summary.items():
            handle.write(
                f"{METHOD_LABELS.get(method, method)} & {metrics['mission_time_wins']:.0f} & "
                f"{metrics['rescue_success_wins']:.0f}\\\\\n"
            )
        handle.write("\\bottomrule\n\\end{tabular*}\n")

    difficulty_table = output_dir / "difficulty_table.tex"
    with difficulty_table.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}llcc}\n")
        handle.write("\\toprule\n")
        handle.write("Difficulty & Method & Mission Time & Rescue Success\\\\\n")
        handle.write("\\midrule\n")
        for difficulty, method_data in difficulty_summary.items():
            for method, metrics in method_data.items():
                handle.write(
                    f"{difficulty.title()} & {METHOD_LABELS.get(method, method)} & "
                    f"{metrics['mission_time']:.2f} & {metrics['rescue_success']:.3f}\\\\\n"
                )
        handle.write("\\bottomrule\n\\end{tabular*}\n")

    regime_table = output_dir / "regime_table.tex"
    with regime_table.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}llcc}\n")
        handle.write("\\toprule\n")
        handle.write("Regime & Method & Mission Time & Rescue Success\\\\\n")
        handle.write("\\midrule\n")
        for regime, method_data in regime_summary.items():
            for method, metrics in method_data.items():
                handle.write(
                    f"{regime.title()} & {METHOD_LABELS.get(method, method)} & "
                    f"{metrics['mission_time']:.2f} & {metrics['rescue_success']:.3f}\\\\\n"
                )
        handle.write("\\bottomrule\n\\end{tabular*}\n")

    ablation_table = output_dir / "ablation_table.tex"
    with ablation_table.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lcc}\n")
        handle.write("\\toprule\n")
        handle.write("Ablation & Mission Time & Rescue Success\\\\\n")
        handle.write("\\midrule\n")
        for method, metrics in ablation_summary.items():
            label = method.replace("behavior_distributed__", "").replace("_", " ").title()
            handle.write(f"{label} & {metrics['mission_time']:.2f} & {metrics['rescue_success']:.3f}\\\\\n")
        handle.write("\\bottomrule\n\\end{tabular*}\n")

    significance_table = output_dir / "significance_table.tex"
    with significance_table.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lcc}\n")
        handle.write("\\toprule\n")
        handle.write("Baseline & $\\Delta T_{\\mathrm{mis}}$ & Bootstrap 95\\% CI\\\\\n")
        handle.write("\\midrule\n")
        for method, metrics in significance_summary.items():
            handle.write(
                f"{METHOD_LABELS.get(method, method)} & {metrics['mission_time_delta_mean']:.2f} & "
                f"[{metrics['mission_time_delta_ci_low']:.2f}, {metrics['mission_time_delta_ci_high']:.2f}]\\\\\n"
            )
        handle.write("\\bottomrule\n\\end{tabular*}\n")


def export_heterogeneity_table(summary: Dict[str, Dict[str, Dict[str, float]]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}llcc}\n")
        handle.write("\\toprule\n")
        handle.write("Scale & Method & Mission Time & Rescue Success\\\\\n")
        handle.write("\\midrule\n")
        for level, method_data in summary.items():
            for method, metrics in method_data.items():
                handle.write(
                    f"{level} & {METHOD_LABELS.get(method, method)} & "
                    f"{metrics['mission_time']:.2f} & {metrics['rescue_success']:.3f}\\\\\n"
                )
        handle.write("\\bottomrule\n\\end{tabular*}\n")


def export_scalability_table(summary: Dict[str, Dict[str, Dict[str, float]]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}llcccc}\n")
        handle.write("\\toprule\n")
        handle.write("Scale & Method & Team & Tasks & Mission Time & Comm.\\\\\n")
        handle.write("\\midrule\n")
        for label, method_data in summary.items():
            for method, metrics in method_data.items():
                team = int(metrics["num_uavs"] + metrics["num_usvs"])
                tasks = int(metrics["num_search_tasks"])
                handle.write(
                    f"{label.title()} & {METHOD_LABELS.get(method, method)} & {team} & {tasks} & "
                    f"{metrics['mission_time']:.2f} & {metrics['communication_load']:.2f}\\\\\n"
                )
        handle.write("\\bottomrule\n\\end{tabular*}\n")


def _setup_axis(axis, title: str) -> None:
    axis.set_facecolor("white")
    axis.set_title(title, fontsize=10.5, weight="bold", pad=6)
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#2f2f2f")
    axis.spines["bottom"].set_color("#2f2f2f")
    axis.spines["left"].set_linewidth(0.7)
    axis.spines["bottom"].set_linewidth(0.7)
    axis.tick_params(axis="both", direction="out", length=3.0, width=0.7, color="#2f2f2f")


def _bar_value_format(metric_key: str) -> str:
    if metric_key == "rescue_success":
        return "{:.3f}"
    if metric_key in {"communication_load", "formation_error"}:
        return "{:.2f}"
    if metric_key.endswith("_wins"):
        return "{:.0f}"
    return "{:.1f}"


def _add_bar_labels(axis, bars, values: Sequence[float], value_format: str) -> None:
    if not values:
        return
    ymin, ymax = axis.get_ylim()
    span = max(ymax - ymin, 1e-9)
    offset = span * 0.018
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + offset,
            value_format.format(value),
            ha="center",
            va="bottom",
            fontsize=7.4,
            color="#222222",
        )
    axis.set_ylim(ymin, ymax + span * 0.08)


def _plot_bars(axis, labels: Sequence[str], values: Sequence[float], colors, metric_key: str) -> None:
    bars = axis.bar(labels, values, color=colors, edgecolor="#2f2f2f", linewidth=0.45, width=0.72)
    _add_bar_labels(axis, bars, values, _bar_value_format(metric_key))
    axis.margins(x=0.02)


def _smoothstep_interp(xs: Sequence[float], ys: Sequence[float], points_per_segment: int = 28) -> tuple[np.ndarray, np.ndarray]:
    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    if len(xs_arr) <= 1:
        return xs_arr, ys_arr
    out_x = []
    out_y = []
    for idx in range(len(xs_arr) - 1):
        t = np.linspace(0.0, 1.0, points_per_segment, endpoint=False)
        eased = t * t * (3.0 - 2.0 * t)
        out_x.extend(xs_arr[idx] + (xs_arr[idx + 1] - xs_arr[idx]) * t)
        out_y.extend(ys_arr[idx] + (ys_arr[idx + 1] - ys_arr[idx]) * eased)
    out_x.append(xs_arr[-1])
    out_y.append(ys_arr[-1])
    return np.asarray(out_x), np.asarray(out_y)


def _moving_trend(
    xs: Sequence[float],
    ys: Sequence[float],
    grid_count: int = 90,
    bandwidth_ratio: float = 0.18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    order = np.argsort(xs_arr)
    xs_arr = xs_arr[order]
    ys_arr = ys_arr[order]
    if len(xs_arr) == 0:
        return xs_arr, ys_arr, np.zeros_like(ys_arr)
    if float(xs_arr[-1] - xs_arr[0]) < 1e-12:
        return xs_arr, ys_arr, np.zeros_like(ys_arr)
    grid = np.linspace(xs_arr[0], xs_arr[-1], grid_count)
    bandwidth = max((xs_arr[-1] - xs_arr[0]) * bandwidth_ratio, 1e-6)
    trend = []
    spread = []
    for center in grid:
        weights = np.exp(-0.5 * ((xs_arr - center) / bandwidth) ** 2)
        weights = weights / max(float(np.sum(weights)), 1e-12)
        mu = float(np.sum(weights * ys_arr))
        var = float(np.sum(weights * (ys_arr - mu) ** 2))
        trend.append(mu)
        spread.append(0.55 * np.sqrt(max(var, 0.0)))
    return grid, np.asarray(trend), np.asarray(spread)


def _empirical_cdf(values: Sequence[float], grid_count: int = 140) -> tuple[np.ndarray, np.ndarray]:
    values_arr = np.sort(np.asarray(values, dtype=float))
    if len(values_arr) == 0:
        return values_arr, values_arr
    if len(values_arr) == 1 or float(values_arr[-1] - values_arr[0]) < 1e-12:
        grid = values_arr
    else:
        grid = np.linspace(values_arr[0], values_arr[-1], grid_count)
    cdf = np.searchsorted(values_arr, grid, side="right") / len(values_arr)
    return grid, cdf


def _cdf_support(
    values: Sequence[float],
    grid_count: int = 160,
    *,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
) -> np.ndarray:
    values_arr = np.sort(np.asarray(values, dtype=float))
    if len(values_arr) == 0:
        return values_arr
    left = values_arr[0] if x_min is None else float(x_min)
    right = values_arr[-1] if x_max is None else float(x_max)
    if right < left:
        left, right = right, left
    span = float(right - left)
    margin = max(span * 0.015, 1e-6)
    if len(values_arr) == 1 or span < 1e-12:
        return np.asarray([left - margin, left, right + margin])
    grid = np.linspace(left, right, grid_count)
    return np.unique(np.concatenate(([left - margin], grid, values_arr, [right + margin])))


def _empirical_cdf_on_grid(values: Sequence[float], grid: np.ndarray) -> np.ndarray:
    values_arr = np.sort(np.asarray(values, dtype=float))
    if len(values_arr) == 0:
        return np.zeros_like(grid, dtype=float)
    return np.searchsorted(values_arr, grid, side="right") / len(values_arr)


def _bootstrap_cdf_band(
    values: Sequence[float],
    grid: np.ndarray,
    *,
    seed: int,
    iters: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    values_arr = np.asarray(values, dtype=float)
    if len(values_arr) < 2 or len(grid) == 0:
        cdf = _empirical_cdf_on_grid(values_arr, grid)
        return cdf, cdf
    rng = np.random.default_rng(seed)
    samples = np.empty((iters, len(grid)), dtype=float)
    for idx in range(iters):
        draw = np.sort(rng.choice(values_arr, size=len(values_arr), replace=True))
        samples[idx] = np.searchsorted(draw, grid, side="right") / len(draw)
    return np.percentile(samples, 10, axis=0), np.percentile(samples, 90, axis=0)


def _plot_cdf_with_band(
    axis,
    values: Sequence[float],
    *,
    color: str,
    label: str,
    seed: int,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
) -> None:
    if not values:
        return
    grid = _cdf_support(values, x_min=x_min, x_max=x_max)
    cdf = _empirical_cdf_on_grid(values, grid)
    lower, upper = _bootstrap_cdf_band(values, grid, seed=seed)
    axis.fill_between(grid, lower, upper, step="post", color=color, alpha=0.10, linewidth=0, zorder=1)
    axis.step(grid, cdf, where="post", linewidth=1.8, color=color, label=label, zorder=3)


def _format_cdf_axis(axis) -> None:
    axis.set_ylim(-0.03, 1.05)
    axis.set_yticks(np.linspace(0.0, 1.0, 6))
    axis.margins(x=0.03)


def _plot_smooth_line(
    axis,
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    color: str,
    label: str,
    shaded: Optional[Sequence[float]] = None,
    linewidth: float = 2.0,
) -> None:
    x_line, y_line = _smoothstep_interp(xs, ys)
    axis.plot(x_line, y_line, linewidth=linewidth, color=color, label=label)
    axis.scatter(xs, ys, s=16, color=color, zorder=3, linewidth=0.0)
    if shaded is not None:
        shade = np.asarray(shaded, dtype=float)
        if len(shade) == len(xs):
            _, shade_line = _smoothstep_interp(xs, shade)
            axis.fill_between(x_line, y_line - shade_line, y_line + shade_line, color=color, alpha=0.13, linewidth=0)


def _dataset_title(output_dir: Path) -> str:
    name = output_dir.name.lower()
    if name == "stress":
        return "Stress Scenarios"
    if name == "ablation":
        return "Ablation Scenarios"
    return "Standard Scenarios"


def _save_figure(fig, output_dir: Path, stem: str) -> None:
    for fmt in PLOT_FORMATS:
        fig.savefig(output_dir / f"{stem}.{fmt}")


def _boxplot_with_labels(axis, data, labels: Sequence[str], widths: float = 0.55):
    kwargs = {"patch_artist": True, "widths": widths}
    if "tick_labels" in inspect.signature(axis.boxplot).parameters:
        kwargs["tick_labels"] = labels
    else:
        kwargs["labels"] = labels
    return axis.boxplot(data, **kwargs)


def plot_summary(summary: Dict[str, Dict[str, float]], significance: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    methods = list(summary.keys())
    labels = [METHOD_LABELS.get(method, method) for method in methods]
    colors = [METHOD_COLORS.get(method, "#777777") for method in methods]
    mission_ci = [0.0]
    for method in methods[1:]:
        mission_ci.append(abs(significance[method]["mission_time_delta_ci_high"] - significance[method]["mission_time_delta_mean"]))

    fig = plt.figure(figsize=(10.8, 8.4))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.82], hspace=0.48, wspace=0.26)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, 0])
    ax4 = fig.add_subplot(grid[1, 1])
    ax5 = fig.add_subplot(grid[2, :])

    metrics = [
        (ax1, "mission_time", "Mission Time"),
        (ax2, "rescue_success", "Rescue Success"),
        (ax3, "total_path_length", "Path Length"),
        (ax4, "communication_load", "Communication Load"),
    ]
    for axis, metric_key, title in metrics:
        values = [summary[method][metric_key] for method in methods]
        _plot_bars(axis, labels, values, colors, metric_key)
        _setup_axis(axis, title)
        axis.tick_params(axis="x", rotation=22, labelsize=8.2)
        for tick in axis.get_xticklabels():
            tick.set_horizontalalignment("right")

    baseline = summary["behavior_distributed"]["mission_time"]
    comparators = methods[1:]
    delta_labels = [METHOD_LABELS.get(method, method) for method in comparators]
    deltas = [summary[method]["mission_time"] - baseline for method in comparators]
    ci_lows = [significance[method]["mission_time_delta_ci_low"] for method in comparators]
    ci_highs = [significance[method]["mission_time_delta_ci_high"] for method in comparators]
    y_positions = list(range(len(comparators)))
    ax5.axvline(0.0, color="#c4c4c4", linestyle="--", linewidth=1.0)
    for idx, method in enumerate(comparators):
        color = METHOD_COLORS.get(method, "#777777")
        ax5.plot([ci_lows[idx], ci_highs[idx]], [idx, idx], color=color, linewidth=3.0, solid_capstyle="round")
        ax5.scatter([deltas[idx]], [idx], color=color, s=70, zorder=3)
    ax5.set_yticks(y_positions, delta_labels)
    ax5.invert_yaxis()
    ax5.set_xlabel("Mission-time delta vs Proposed")
    _setup_axis(ax5, "Paired bootstrap comparison")
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.08, top=0.965, hspace=0.48, wspace=0.26)
    _save_figure(fig, output_dir, "summary")
    _save_figure(fig, output_dir, "main_results_summary")
    _save_figure(fig, output_dir, "core_results_panel")
    plt.close(fig)


def plot_distributions(rows: List[Dict[str, float]], output_dir: Path) -> None:
    methods = _base_methods(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    for axis, (metric_key, title) in zip(axes, [("mission_time", "Mission Time"), ("rescue_success", "Rescue Success")]):
        data = [[float(row[metric_key]) for row in rows if row["method"] == method] for method in methods]
        bp = _boxplot_with_labels(axis, data, [METHOD_LABELS.get(method, method) for method in methods])
        for patch, method in zip(bp["boxes"], methods):
            patch.set_facecolor(METHOD_COLORS.get(method, "#777777"))
            patch.set_alpha(0.68)
            patch.set_linewidth(1.0)
        for median_line in bp["medians"]:
            median_line.set_color("#1f1f1f")
            median_line.set_linewidth(1.2)
        _setup_axis(axis, title)
        axis.tick_params(axis="x", rotation=15, labelsize=8.5)
    fig.tight_layout()
    _save_figure(fig, output_dir, "distribution_boxplots")
    _save_figure(fig, output_dir, "distribution_results_boxplots")
    plt.close(fig)


def plot_cdf(rows: List[Dict[str, float]], output_dir: Path, metric_key: str, title: str, filename: str) -> None:
    methods = _base_methods(rows)
    all_values = [float(row[metric_key]) for row in rows if row["method"] in methods]
    x_min = min(all_values) if all_values else None
    x_max = max(all_values) if all_values else None
    fig, axis = plt.subplots(figsize=(6.8, 4.6))
    for method_idx, method in enumerate(methods):
        values = [float(row[metric_key]) for row in rows if row["method"] == method]
        _plot_cdf_with_band(
            axis,
            values,
            color=METHOD_COLORS.get(method, "#777777"),
            label=METHOD_LABELS.get(method, method),
            seed=3100 + method_idx,
            x_min=x_min,
            x_max=x_max,
        )
    axis.set_xlabel(metric_key.replace("_", " ").title())
    axis.set_ylabel("Empirical CDF")
    _setup_axis(axis, title)
    _format_cdf_axis(axis)
    axis.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, output_dir, Path(filename).stem)
    plt.close(fig)


def plot_sensitivity(rows: List[Dict[str, float]], output_dir: Path, x_key: str, y_key: str, xlabel: str, filename: str) -> None:
    methods = _base_methods(rows)
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        xs = [float(row[x_key]) for row in method_rows]
        ys = [float(row[y_key]) for row in method_rows]
        if xs:
            grid, trend, spread = _moving_trend(xs, ys)
            color = METHOD_COLORS.get(method, "#777777")
            axis.plot(grid, trend, linewidth=2.0, color=color, label=METHOD_LABELS.get(method, method))
            axis.fill_between(grid, trend - spread, trend + spread, color=color, alpha=0.12, linewidth=0)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(y_key.replace("_", " ").title())
    _setup_axis(axis, xlabel + " Sensitivity")
    axis.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, output_dir, Path(filename).stem)
    plt.close(fig)


def plot_wins(win_summary: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    methods = list(win_summary.keys())
    labels = [METHOD_LABELS.get(method, method) for method in methods]
    colors = [METHOD_COLORS.get(method, "#777777") for method in methods]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    mission_values = [win_summary[method]["mission_time_wins"] for method in methods]
    rescue_values = [win_summary[method]["rescue_success_wins"] for method in methods]
    _plot_bars(axes[0], labels, mission_values, colors, "mission_time_wins")
    _plot_bars(axes[1], labels, rescue_values, colors, "rescue_success_wins")
    _setup_axis(axes[0], "Mission-Time Wins")
    _setup_axis(axes[1], "Rescue-Success Wins")
    for axis in axes:
        axis.set_ylabel("Wins")
        axis.tick_params(axis="x", rotation=22, labelsize=8.2)
        for tick in axis.get_xticklabels():
            tick.set_horizontalalignment("right")
    fig.tight_layout()
    _save_figure(fig, output_dir, "win_counts")
    _save_figure(fig, output_dir, "scenario_wise_wins")
    plt.close(fig)


def plot_ablation(ablation_summary: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    if not ablation_summary:
        return
    methods = list(ablation_summary.keys())
    labels = [
        method.replace("behavior_distributed__", "").replace("_", " ").title().replace("Rl", "RL")
        for method in methods
    ]
    mission_values = [ablation_summary[method]["mission_time"] for method in methods]
    rescue_values = [ablation_summary[method]["rescue_success"] for method in methods]
    formation_values = [ablation_summary[method]["formation_error"] for method in methods]

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2))
    _plot_bars(axes[0], labels, mission_values, "#1666b0", "mission_time")
    _plot_bars(axes[1], labels, rescue_values, "#1e9d74", "rescue_success")
    _plot_bars(axes[2], labels, formation_values, "#d18219", "formation_error")
    _setup_axis(axes[0], "Mission Time")
    _setup_axis(axes[1], "Rescue Success")
    _setup_axis(axes[2], "Slot Error")
    axes[0].set_ylabel("Mission Time")
    axes[1].set_ylabel("Rescue Success")
    axes[2].set_ylabel("Slot Error")
    for axis in axes:
        axis.tick_params(axis="x", rotation=24, labelsize=8.0)
        for tick in axis.get_xticklabels():
            tick.set_horizontalalignment("right")
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.92, wspace=0.22)
    _save_figure(fig, output_dir, "ablation_bar")
    _save_figure(fig, output_dir, "ablation_results")
    _save_figure(fig, output_dir, "ablation_results_panel")
    plt.close(fig)

    column_specs = [
        (mission_values, "#1666b0", "Mission Time", "mission_time"),
        (rescue_values, "#1e9d74", "Rescue Success", "rescue_success"),
        (formation_values, "#d18219", "Slot Error", "formation_error"),
    ]
    y_positions = np.arange(len(labels))
    fig, axes = plt.subplots(3, 1, figsize=(4.2, 5.4))
    for axis, (values, color, title, metric_key) in zip(axes, column_specs):
        axis.barh(y_positions, values, color=color, edgecolor="#343434", linewidth=0.35)
        axis.set_yticks(y_positions, labels, fontsize=8.2)
        axis.invert_yaxis()
        axis.set_xlabel(title)
        _setup_axis(axis, title)
        offset = (max(values) - min(values)) * 0.03 if max(values) > min(values) else max(values) * 0.03
        value_format = _bar_value_format(metric_key)
        for idx, value in enumerate(values):
            axis.text(value + offset, idx, value_format.format(value), va="center", fontsize=7.6)
        axis.margins(x=0.18)
    fig.subplots_adjust(left=0.29, right=0.97, bottom=0.08, top=0.95, hspace=0.55)
    _save_figure(fig, output_dir, "ablation_results_panel_column")
    plt.close(fig)


def plot_robustness_panel(rows: List[Dict[str, float]], win_summary: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    methods = _base_methods(rows)
    all_mission_times = [float(row["mission_time"]) for row in rows if row["method"] in methods]
    cdf_x_min = min(all_mission_times) if all_mission_times else None
    cdf_x_max = max(all_mission_times) if all_mission_times else None
    fig = plt.figure(figsize=(10.8, 8.8))
    grid = fig.add_gridspec(3, 2, height_ratios=[0.95, 0.95, 1.05], hspace=0.46, wspace=0.28)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, 0])
    ax4 = fig.add_subplot(grid[1, 1])
    ax5 = fig.add_subplot(grid[2, :])

    for method_idx, method in enumerate(methods):
        values = [float(row["mission_time"]) for row in rows if row["method"] == method]
        _plot_cdf_with_band(
            ax1,
            values,
            color=METHOD_COLORS.get(method, "#777777"),
            label=METHOD_LABELS.get(method, method),
            seed=4100 + method_idx,
            x_min=cdf_x_min,
            x_max=cdf_x_max,
        )
    ax1.legend(frameon=False, fontsize=8)
    _setup_axis(ax1, "Mission-Time CDF")
    _format_cdf_axis(ax1)
    ax1.set_xlabel("Mission Time")
    ax1.set_ylabel("Empirical CDF")

    labels = [METHOD_LABELS.get(method, method) for method in methods]
    colors = [METHOD_COLORS.get(method, "#777777") for method in methods]
    _plot_bars(ax2, labels, [win_summary[method]["mission_time_wins"] for method in methods], colors, "mission_time_wins")
    _setup_axis(ax2, "Scenario-wise wins")
    ax2.set_ylabel("Wins")
    ax2.tick_params(axis="x", rotation=22, labelsize=8.2)
    for tick in ax2.get_xticklabels():
        tick.set_horizontalalignment("right")

    for axis, x_key, xlabel in [(ax3, "communication_quality", "Communication Quality"), (ax4, "sea_state", "Sea-State Level")]:
        for method in methods:
            method_rows = [row for row in rows if row["method"] == method]
            xs = [float(row[x_key]) for row in method_rows]
            ys = [float(row["mission_time"]) for row in method_rows]
            if xs:
                grid, trend, spread = _moving_trend(xs, ys)
                color = METHOD_COLORS.get(method, "#777777")
                axis.plot(grid, trend, linewidth=1.9, color=color, label=METHOD_LABELS.get(method, method))
                axis.fill_between(grid, trend - spread, trend + spread, color=color, alpha=0.10, linewidth=0)
        _setup_axis(axis, xlabel + " Sensitivity")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Mission Time")

    dist_rows = [row for row in rows if "__" not in row["method"]]
    data = [[float(row["rescue_success"]) for row in dist_rows if row["method"] == method] for method in methods]
    bp = _boxplot_with_labels(ax5, data, labels)
    for patch, method in zip(bp["boxes"], methods):
        patch.set_facecolor(METHOD_COLORS.get(method, "#777777"))
        patch.set_alpha(0.68)
    _setup_axis(ax5, "Rescue-success distribution")
    ax5.tick_params(axis="x", rotation=15, labelsize=8.5)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.07, top=0.96, hspace=0.46, wspace=0.28)
    _save_figure(fig, output_dir, "robustness_panel")
    plt.close(fig)


def plot_heterogeneity(summary: Dict[str, Dict[str, Dict[str, float]]], output_dir: Path) -> None:
    levels = [float(level) for level in summary.keys()]
    methods = _ordered_methods({method for level in summary.values() for method in level.keys()})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    for method in methods:
        mission = [summary[f"{level:.2f}"][method]["mission_time"] for level in levels]
        rescue = [summary[f"{level:.2f}"][method]["rescue_success"] for level in levels]
        color = METHOD_COLORS.get(method, "#777777")
        _plot_smooth_line(axes[0], levels, mission, color=color, label=METHOD_LABELS.get(method, method), shaded=[0.22] * len(levels), linewidth=2.0)
        _plot_smooth_line(axes[1], levels, rescue, color=color, label=METHOD_LABELS.get(method, method), shaded=[0.004] * len(levels), linewidth=2.0)
    _setup_axis(axes[0], "Mission Time vs Heterogeneity")
    _setup_axis(axes[1], "Rescue Success vs Heterogeneity")
    axes[0].set_xlabel("Heterogeneity Scale")
    axes[1].set_xlabel("Heterogeneity Scale")
    axes[0].set_ylabel("Mission Time")
    axes[1].set_ylabel("Rescue Success")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save_figure(fig, output_dir, "heterogeneity_sweep")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(4.2, 5.0), sharex=True)
    for method in methods:
        mission = [summary[f"{level:.2f}"][method]["mission_time"] for level in levels]
        rescue = [summary[f"{level:.2f}"][method]["rescue_success"] for level in levels]
        color = METHOD_COLORS.get(method, "#777777")
        label = METHOD_LABELS.get(method, method)
        _plot_smooth_line(axes[0], levels, mission, color=color, label=label, shaded=[0.22] * len(levels), linewidth=1.9)
        _plot_smooth_line(axes[1], levels, rescue, color=color, label=label, shaded=[0.004] * len(levels), linewidth=1.9)
    _setup_axis(axes[0], "Mission Time")
    _setup_axis(axes[1], "Rescue Success")
    axes[0].set_ylabel("Mission Time")
    axes[1].set_ylabel("Rescue Success")
    axes[1].set_xlabel("Heterogeneity Scale")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False, fontsize=6.7)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.20, top=0.94, hspace=0.34)
    _save_figure(fig, output_dir, "heterogeneity_sweep_column")
    plt.close(fig)


def plot_scalability(summary: Dict[str, Dict[str, Dict[str, float]]], output_dir: Path) -> None:
    labels = list(summary.keys())
    methods = _ordered_methods(_base_methods([{"method": method} for method in summary[labels[0]]]))
    x_positions = list(range(len(labels)))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    for axis, metric_key, ylabel in [
        (axes[0], "mission_time", "Mission Time"),
        (axes[1], "communication_load", "Communication Load"),
    ]:
        for method in methods:
            values = [summary[label][method][metric_key] for label in labels]
            color = METHOD_COLORS.get(method, "#777777")
            shade = [0.28] * len(values) if metric_key == "mission_time" else [0.20] * len(values)
            _plot_smooth_line(axis, x_positions, values, color=color, label=METHOD_LABELS.get(method, method), shaded=shade)
        axis.set_xticks(x_positions, [label.title() for label in labels])
        axis.set_ylabel(ylabel)
        _setup_axis(axis, ylabel)
    axes[0].set_xlabel("Scenario scale")
    axes[1].set_xlabel("Scenario scale")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save_figure(fig, output_dir, "scalability_sweep")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(4.2, 5.0), sharex=True)
    for axis, metric_key, ylabel in [
        (axes[0], "mission_time", "Mission Time"),
        (axes[1], "communication_load", "Communication Load"),
    ]:
        for method in methods:
            values = [summary[label][method][metric_key] for label in labels]
            color = METHOD_COLORS.get(method, "#777777")
            shade = [0.28] * len(values) if metric_key == "mission_time" else [0.20] * len(values)
            _plot_smooth_line(axis, x_positions, values, color=color, label=METHOD_LABELS.get(method, method), shaded=shade, linewidth=1.9)
        axis.set_xticks(x_positions, [label.title() for label in labels])
        axis.set_ylabel(ylabel)
        _setup_axis(axis, ylabel)
    axes[1].set_xlabel("Scenario scale")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False, fontsize=6.7)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.20, top=0.94, hspace=0.34)
    _save_figure(fig, output_dir, "scalability_sweep_column")
    plt.close(fig)
