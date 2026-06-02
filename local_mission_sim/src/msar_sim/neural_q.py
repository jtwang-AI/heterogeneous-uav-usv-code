from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .model import Agent, Task
from .utility import distance, travel_cost, utility


FEATURE_NAMES = [
    "information_gain",
    "rescue_value",
    "target_probability",
    "agent_energy",
    "is_uav",
    "is_usv",
    "risk_cost",
    "travel_time",
    "agent_speed",
    "is_search",
    "is_verify",
    "is_encircle",
]


def neural_q_artifact_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "outputs" / "neural_q"


def default_weight_path() -> Path:
    return neural_q_artifact_dir() / "neural_q_weights.json"


def agent_task_features(agent: Agent, task: Task) -> np.ndarray:
    task_type = task.task_type.lower()
    return np.asarray(
        [
            task.information_gain,
            task.rescue_value,
            task.target_probability,
            agent.energy,
            1.0 if agent.kind == "UAV" else 0.0,
            1.0 if agent.kind == "USV" else 0.0,
            task.risk_cost,
            travel_cost(agent, task),
            agent.speed,
            1.0 if task_type == "search" else 0.0,
            1.0 if task_type == "verify" else 0.0,
            1.0 if task_type == "encircle" else 0.0,
        ],
        dtype=np.float64,
    )


def expert_q_target(agent: Agent, task: Task) -> float:
    """Offline target used to train the neural assignment comparator."""
    target_bonus = 2.5 * task.target_probability
    kind_bonus = 0.45 * task.information_gain if agent.kind == "UAV" else 0.55 * task.rescue_value
    travel_penalty = 0.12 * distance(agent.position, task.position) / max(agent.speed, 1e-6)
    return utility(agent, task) + target_bonus + kind_bonus - travel_penalty


class NeuralQScorer:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self.mean = np.asarray(payload["feature_mean"], dtype=np.float64)
        self.std = np.asarray(payload["feature_std"], dtype=np.float64)
        self.w1 = np.asarray(payload["w1"], dtype=np.float64)
        self.b1 = np.asarray(payload["b1"], dtype=np.float64)
        self.w2 = np.asarray(payload["w2"], dtype=np.float64)
        self.b2 = np.asarray(payload["b2"], dtype=np.float64)

    @classmethod
    def load(cls, path: Path | None = None) -> "NeuralQScorer":
        weight_path = path or default_weight_path()
        if not weight_path.exists():
            return cls(_fallback_payload())
        return cls(json.loads(weight_path.read_text(encoding="utf-8")))

    def score(self, agent: Agent, task: Task) -> float:
        x = (agent_task_features(agent, task) - self.mean) / np.maximum(self.std, 1e-8)
        hidden = np.tanh(x @ self.w1 + self.b1)
        return float(hidden @ self.w2 + self.b2)


def _fallback_payload() -> Dict[str, Any]:
    rng = np.random.default_rng(7)
    w1 = rng.normal(0.0, 0.05, size=(len(FEATURE_NAMES), 16))
    b1 = np.zeros(16)
    w2 = rng.normal(0.0, 0.05, size=16)
    b2 = 0.0
    return {
        "feature_names": FEATURE_NAMES,
        "feature_mean": [0.0] * len(FEATURE_NAMES),
        "feature_std": [1.0] * len(FEATURE_NAMES),
        "w1": w1.tolist(),
        "b1": b1.tolist(),
        "w2": w2.tolist(),
        "b2": b2,
        "note": "Deterministic fallback; run code/train_neural_q.py to create trained weights.",
    }


def greedy_neural_q_assignment(agents: List[Agent], tasks: List[Task], scorer: NeuralQScorer | None = None) -> Dict[str, str]:
    scorer = scorer or NeuralQScorer.load()
    remaining_tasks = {task.task_id: task for task in tasks}
    assignment: Dict[str, str] = {}
    for agent in sorted(agents, key=lambda item: item.agent_id):
        if not remaining_tasks:
            break
        best_task = max(
            remaining_tasks.values(),
            key=lambda task: (
                scorer.score(agent, task),
                task.target_probability,
                -distance(agent.position, task.position),
            ),
        )
        assignment[agent.agent_id] = best_task.task_id
        remaining_tasks.pop(best_task.task_id, None)
    return assignment
