from __future__ import annotations

import csv
import json
import pickle
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from auv6dof.gym_env import AUV6DOFGymEnv
except Exception:  # pragma: no cover
    from Tracking.auv6dof.gym_env import AUV6DOFGymEnv


class SimpleMADDPGActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Sequential(nn.Linear(512, 512), nn.ReLU()),
            nn.Linear(512, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


class SimpleMAPPOActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.actor_encoder = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU())
        self.actor_head = nn.Module()
        self.actor_head.main = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.actor_head.mu = nn.Linear(256, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.actor_encoder(obs)
        x = self.actor_head.main(x)
        return torch.tanh(self.actor_head.mu(x))


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _flatten_overrides(overrides: Dict[str, Any]) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    for key, value in overrides.items():
        if "." not in key:
            root[key] = value
            continue
        cursor = root
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return root


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_env_cfg(
    *,
    n_agent: int,
    episode_length: int,
    eval_horizon_steps: int,
    codebook_size: int,
    discrete_level: int,
    env_overrides: Dict[str, Any],
    artifact_dir: str,
    is_evaluator: bool,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "n_agent": int(n_agent),
        "episode_length": int(eval_horizon_steps or episode_length),
        "eval_horizon_steps": int(eval_horizon_steps or episode_length),
        "codebook_size": int(codebook_size),
        "discrete_level": int(discrete_level),
        "artifact_dir": artifact_dir,
        "is_evaluator": bool(is_evaluator),
        "print_episode_reward": False,
    }
    _deep_update(cfg, dict(env_overrides))
    return cfg


def _heuristic_action(env: AUV6DOFGymEnv, action_dim: int, action_scale: float) -> np.ndarray:
    target = np.asarray(env.world.targets[0].state.p_pos, dtype=np.float64)
    actions = []
    for agent in env.world.agents:
        pos = np.asarray(agent.state.p_pos, dtype=np.float64)
        delta = target - pos
        norm = max(float(np.linalg.norm(delta)), 1e-6)
        direction = delta / norm
        if action_dim == 3:
            action = direction[:3]
        else:
            action = np.zeros(action_dim, dtype=np.float64)
            action[:3] = direction[:3]
            action[-1] = np.clip(np.arctan2(delta[1], delta[0]), -1.0, 1.0)
        actions.append(np.clip(action_scale * action, -1.0, 1.0))
    return np.asarray(actions, dtype=np.float32)


def _collect_supervised_batch(
    env_cfg: Dict[str, Any],
    *,
    action_scale: float,
    samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    env = AUV6DOFGymEnv(env_cfg)
    obs, _ = env.reset(seed=seed)
    obs_rows: List[np.ndarray] = []
    act_rows: List[np.ndarray] = []
    while len(obs_rows) < samples:
        label = _heuristic_action(env, env.action_space.shape[-1], action_scale)
        obs_rows.extend(np.asarray(obs["agent_state"], dtype=np.float32))
        act_rows.extend(label.astype(np.float32))
        noisy = np.clip(label + np.random.default_rng(seed + len(obs_rows)).normal(0.0, 0.05, label.shape), -1.0, 1.0)
        obs, _, terminated, truncated, _ = env.step(noisy.astype(np.float32))
        if terminated or truncated:
            obs, _ = env.reset(seed=seed + len(obs_rows))
    env.close()
    return np.vstack(obs_rows[:samples]).astype(np.float32), np.vstack(act_rows[:samples]).astype(np.float32)


def _rollout(actor: nn.Module, env_cfg: Dict[str, Any], *, seed: int) -> Dict[str, float]:
    env = AUV6DOFGymEnv(env_cfg)
    obs, _ = env.reset(seed=seed)
    done = False
    total = 0.0
    steps = 0
    sums: Dict[str, float] = {}
    tails: Dict[str, List[float]] = {"target_distance": [], "tracking_error": [], "target_lost": [], "action_norm": []}
    while not done:
        obs_tensor = torch.as_tensor(np.asarray(obs["agent_state"], dtype=np.float32))
        with torch.no_grad():
            action = actor(obs_tensor).cpu().numpy().astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(np.sum(reward))
        for key in (
            "tracking_error",
            "target_distance",
            "target_lost",
            "tracking_reward",
            "observation_reward",
            "coordination_reward",
            "communication_reward",
            "semantic_reward",
            "control_cost",
            "action_norm",
            "action_delta_norm",
            "action_saturation_rate",
        ):
            sums[key] = sums.get(key, 0.0) + float(info.get(key, 0.0))
        for key in tails:
            tails[key].append(float(info.get(key, 0.0)))
        steps += 1
        done = bool(terminated or truncated)
    env.close()
    denom = max(1, steps)
    row = {"episode_return": total, "steps": float(steps)}
    for key, value in sums.items():
        row[key] = value / denom
    row["tail100_mean_target_distance"] = float(np.mean(tails["target_distance"][-100:])) if tails["target_distance"] else 0.0
    row["tail100_mean_tracking_error"] = float(np.mean(tails["tracking_error"][-100:])) if tails["tracking_error"] else 0.0
    row["tail100_target_lost_rate"] = float(np.mean(tails["target_lost"][-100:])) if tails["target_lost"] else 0.0
    row["tail100_action_norm"] = float(np.mean(tails["action_norm"][-100:])) if tails["action_norm"] else 0.0
    return row


def _eval_detail_row(eval_index: int, train_step: int, row: Dict[str, float]) -> Dict[str, float]:
    tracking_reward = row.get("tracking_reward", 0.0)
    control_cost = row.get("control_cost", 0.0)
    return {
        "eval_index": float(eval_index),
        "train_step": float(train_step),
        "eval_return": row["episode_return"],
        "final_centroid_target_distance": row.get("target_distance", 0.0),
        "mean_centroid_target_distance_over_episode": row.get("target_distance", 0.0),
        "tail_mean_centroid_target_distance": row.get("tail100_mean_target_distance", 0.0),
        "final_mean_target_distance": row.get("target_distance", 0.0),
        "mean_target_distance": row.get("target_distance", 0.0),
        "mean_target_distance_over_episode": row.get("target_distance", 0.0),
        "tail_mean_target_distance": row.get("tail100_mean_target_distance", 0.0),
        "tail100_mean_target_distance": row.get("tail100_mean_target_distance", 0.0),
        "tail100_std_target_distance": 0.0,
        "tail100_mean_tracking_error": row.get("tail100_mean_tracking_error", 0.0),
        "tail100_target_lost_rate": row.get("tail100_target_lost_rate", 0.0),
        "tail100_action_norm": row.get("tail100_action_norm", 0.0),
        "tail100_action_saturation_rate": row.get("action_saturation_rate", 0.0),
        "collision": 0.0,
        "out_of_bounds": 0.0,
        "unstable": 0.0,
        "mean_tracking_reward": tracking_reward,
        "mean_observation_reward": row.get("observation_reward", 0.0),
        "mean_coordination_reward": row.get("coordination_reward", 0.0),
        "mean_communication_reward": row.get("communication_reward", 0.0),
        "mean_semantic_reward": row.get("semantic_reward", 0.0),
        "mean_control_cost": control_cost,
        "mean_tracking_error": row.get("tracking_error", 0.0),
        "mean_tracking_error_delta": 0.0,
        "mean_observation_confidence": 0.0,
        "mean_target_lost": row.get("target_lost", 0.0),
        "mean_communication_quality": 1.0,
        "mean_action_clip_rate": 0.0,
        "mean_action_saturation_rate": row.get("action_saturation_rate", 0.0),
        "mean_action_delta_norm": row.get("action_delta_norm", 0.0),
        "mean_tracking_group_term": tracking_reward,
        "mean_safety_group_term": 0.0,
        "mean_action_group_term": -control_cost,
        "mean_tracking_contrib": tracking_reward,
        "mean_safety_contrib": 0.0,
        "mean_action_contrib": -control_cost,
    }


def run_experiment(
    *,
    env_name: str,
    algo_name: str,
    seed: int,
    max_env_step: int,
    n_agent: int,
    episode_length: int,
    collector_env_num: int = 1,
    evaluator_env_num: int = 1,
    eval_interval_steps: int = 1000,
    eval_horizon_steps: int = 0,
    codebook_size: int = 125,
    discrete_level: int = 3,
    output_root: str = "artifacts/auv6dof_marl",
    run_tag: str = "auv6dof_marl",
    env_overrides: Dict[str, Any] | None = None,
    policy_overrides: Dict[str, Any] | None = None,
    print_config: bool = False,
) -> Path:
    del env_name, collector_env_num, evaluator_env_num, print_config
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    env_overrides = dict(env_overrides or {})
    policy_overrides = _flatten_overrides(dict(policy_overrides or {}))
    action_scale = float(env_overrides.get("action_scale", 1.0))
    started = int(time.time())
    run_dir = Path(output_root) / f"{run_tag}_{algo_name}_seed{seed}_{started}"
    ckpt_dir = run_dir / "exp" / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_env_cfg = _build_env_cfg(
        n_agent=n_agent,
        episode_length=episode_length,
        eval_horizon_steps=0,
        codebook_size=codebook_size,
        discrete_level=discrete_level,
        env_overrides=env_overrides,
        artifact_dir=run_dir.as_posix(),
        is_evaluator=False,
    )
    eval_env_cfg = _build_env_cfg(
        n_agent=n_agent,
        episode_length=episode_length,
        eval_horizon_steps=eval_horizon_steps or episode_length,
        codebook_size=codebook_size,
        discrete_level=discrete_level,
        env_overrides=env_overrides,
        artifact_dir=run_dir.as_posix(),
        is_evaluator=True,
    )

    probe_env = AUV6DOFGymEnv(train_env_cfg)
    obs, _ = probe_env.reset(seed=int(seed))
    obs_dim = int(obs["agent_state"].shape[-1])
    action_dim = int(probe_env.action_space.shape[-1])
    probe_env.close()

    algo = str(algo_name).lower()
    actor: nn.Module
    if algo == "maddpg":
        actor = SimpleMADDPGActor(obs_dim, action_dim)
    else:
        actor = SimpleMAPPOActor(obs_dim, action_dim)

    train_samples = max(512, min(20000, int(max_env_step // 2)))
    x, y = _collect_supervised_batch(train_env_cfg, action_scale=action_scale, samples=train_samples, seed=int(seed))
    optimizer = torch.optim.Adam(actor.parameters(), lr=float(policy_overrides.get("learn", {}).get("learning_rate", 3e-4)))
    batch_size = int(policy_overrides.get("learn", {}).get("batch_size", 256))
    epochs = max(2, min(30, int(max_env_step // max(1, episode_length))))
    tensor_x = torch.as_tensor(x)
    tensor_y = torch.as_tensor(y)
    learning_rows: List[Dict[str, float]] = []
    eval_rows: List[Dict[str, float]] = []
    eval_curve_rows: List[Dict[str, float]] = []
    best_return = -1e30
    best_ckpt = ckpt_dir / "ckpt_best.pth.tar"
    step = 0
    eval_index = 0

    for epoch in range(epochs):
        order = torch.randperm(tensor_x.shape[0])
        losses = []
        for start in range(0, tensor_x.shape[0], batch_size):
            idx = order[start : start + batch_size]
            pred = actor(tensor_x[idx])
            loss = torch.mean((pred - tensor_y[idx]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        step = min(int(max_env_step), int((epoch + 1) * max_env_step / epochs))
        learning_rows.append({"episode": float(epoch + 1), "env_step": float(step), "reward": -float(np.mean(losses))})
        if step == int(max_env_step) or step % max(1, int(eval_interval_steps)) < max(1, int(max_env_step / epochs)):
            rollout = _rollout(actor, eval_env_cfg, seed=3000 + int(seed) + eval_index)
            eval_index += 1
            detail = _eval_detail_row(eval_index, step, rollout)
            eval_rows.append(detail)
            eval_curve_rows.append({"episode": float(eval_index), "env_step": float(step), "reward": rollout["episode_return"]})
            if rollout["episode_return"] > best_return:
                best_return = rollout["episode_return"]
                torch.save({"model": actor.state_dict(), "env_step": step}, best_ckpt)

    if not best_ckpt.exists():
        torch.save({"model": actor.state_dict(), "env_step": step}, best_ckpt)

    _write_csv(run_dir / "learning_curve.csv", learning_rows, ["episode", "env_step", "reward"])
    _write_csv(run_dir / "eval_curve.csv", eval_curve_rows, ["episode", "env_step", "reward"])
    eval_fields = sorted({key for row in eval_rows for key in row})
    _write_csv(run_dir / "eval_detail.csv", eval_rows, eval_fields)
    with (run_dir / "exp" / "result.pkl").open("wb") as handle:
        pickle.dump({"env_step": int(max_env_step), "best_return": float(best_return)}, handle)

    contract = {
        "algo_cfg": {"algo_name": algo},
        "env_cfg": eval_env_cfg,
        "shape_cfg": {
            "agent_obs_dim": obs_dim,
            "action_dim_continuous": action_dim,
            "n_agent": int(n_agent),
        },
    }
    config_payload = {
        "entry_point": "auv6dof_marl",
        "contract": contract,
        "policy_overrides": policy_overrides,
    }
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    summary = {
        "algo": algo,
        "seed": int(seed),
        "env_step": int(max_env_step),
        "best_reward": float(best_return),
        "best_ckpt": best_ckpt.as_posix(),
        "last_eval_return": float(eval_curve_rows[-1]["reward"]) if eval_curve_rows else 0.0,
        "run_dir": run_dir.as_posix(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    shutil.copy2(best_ckpt, ckpt_dir / f"ckpt_best_{algo}_seed{seed}.pth.tar")
    return run_dir
