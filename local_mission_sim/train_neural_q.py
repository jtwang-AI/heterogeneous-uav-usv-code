from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.msar_sim.neural_q import FEATURE_NAMES, agent_task_features, expert_q_target, neural_q_artifact_dir
from src.msar_sim.scenario import generate_scenario


def build_dataset(num_scenarios: int, seed: int, regime: str) -> Tuple[np.ndarray, np.ndarray]:
    features: List[np.ndarray] = []
    targets: List[float] = []
    for idx in range(num_scenarios):
        scenario = generate_scenario(seed=seed + idx, scenario_id=idx, regime=regime)
        tasks = scenario.search_tasks + [scenario.verify_task]
        for agent in scenario.agents:
            for task in tasks:
                features.append(agent_task_features(agent, task))
                targets.append(expert_q_target(agent, task))
    return np.vstack(features), np.asarray(targets, dtype=np.float64)


def train_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    hidden: int = 16,
    epochs: int = 1,
    lr: float = 0.0,
    seed: int = 3,
) -> Tuple[Dict[str, object], List[Dict[str, float]]]:
    rng = np.random.default_rng(seed)
    mean = train_x.mean(axis=0)
    std = np.maximum(train_x.std(axis=0), 1e-6)
    x = (train_x - mean) / std
    xv = (val_x - mean) / std
    y_mean = float(train_y.mean())
    y_std = float(max(train_y.std(), 1e-6))
    y = (train_y - y_mean) / y_std
    yv = (val_y - y_mean) / y_std

    w1 = rng.normal(0.0, 0.35, size=(x.shape[1], hidden))
    b1 = np.zeros(hidden)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        hidden_train = np.tanh(x @ w1 + b1)
        hidden_val = np.tanh(xv @ w1 + b1)
        design = np.column_stack([hidden_train, np.ones(hidden_train.shape[0])])
        ridge = 1e-3
        coeff = np.linalg.solve(design.T @ design + ridge * np.eye(design.shape[1]), design.T @ y)
    w2 = coeff[:-1]
    b2 = float(coeff[-1])
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        train_pred = hidden_train @ w2 + b2
        val_pred = hidden_val @ w2 + b2
    final_train = float(np.mean((train_pred - y) ** 2))
    final_val = float(np.mean((val_pred - yv) ** 2))
    history: List[Dict[str, float]] = [
        {"epoch": 0.0, "train_mse": float(np.mean(y**2)), "val_mse": float(np.mean(yv**2))},
        {"epoch": 1.0, "train_mse": final_train, "val_mse": final_val},
    ]

    payload = {
        "model": "one_hidden_layer_tanh_q_network_ridge_output",
        "feature_names": FEATURE_NAMES,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "target_mean": y_mean,
        "target_std": y_std,
        "w1": w1.tolist(),
        "b1": b1.tolist(),
        "w2": (w2 * y_std).tolist(),
        "b2": float(b2 * y_std + y_mean),
        "hidden_units": hidden,
        "epochs": epochs,
        "learning_rate": lr,
    }
    return payload, history


def main() -> None:
    out_dir = neural_q_artifact_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_std_x, train_std_y = build_dataset(num_scenarios=600, seed=8000, regime="standard")
    train_str_x, train_str_y = build_dataset(num_scenarios=260, seed=9000, regime="stress")
    val_x, val_y = build_dataset(num_scenarios=120, seed=10000, regime="standard")
    train_x = np.vstack([train_std_x, train_str_x])
    train_y = np.concatenate([train_std_y, train_str_y])

    payload, history = train_mlp(train_x, train_y, val_x, val_y)
    (out_dir / "neural_q_weights.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "train_pairs": int(train_x.shape[0]),
                "validation_pairs": int(val_x.shape[0]),
                "final_train_mse": history[-1]["train_mse"],
                "final_val_mse": history[-1]["val_mse"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (out_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_mse", "val_mse"])
        writer.writeheader()
        writer.writerows(history)

    fig, axis = plt.subplots(figsize=(5.8, 3.8))
    axis.plot([row["epoch"] for row in history], [row["train_mse"] for row in history], label="train")
    axis.plot([row["epoch"] for row in history], [row["val_mse"] for row in history], label="validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Normalized MSE")
    axis.grid(True, linestyle="--", alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curve.png", dpi=300)
    fig.savefig(out_dir / "training_curve.pdf")
    plt.close(fig)
    print(json.dumps({"output_dir": out_dir.as_posix(), "summary": history[-1]}, indent=2))


if __name__ == "__main__":
    main()
