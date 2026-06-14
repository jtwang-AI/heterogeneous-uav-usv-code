from __future__ import annotations

from copy import deepcopy
from itertools import combinations, permutations
from typing import Callable, Dict, List, Tuple

from .assignment import distributed_greedy_assignment, rank_tasks
from .model import Agent, Task
from .neural_q import NeuralQScorer, greedy_neural_q_assignment
from .utility import distance, utility


AssignmentMethod = Callable[[List[Agent], List[Task]], Dict[str, str]]


def behavior_rule_distributed(agents: List[Agent], tasks: List[Task]) -> Dict[str, str]:
    return distributed_greedy_assignment(agents, tasks)


def behavior_distributed(agents: List[Agent], tasks: List[Task]) -> Dict[str, str]:
    return rl_guided_behavior_distributed(agents, tasks)


def rl_guided_behavior_score(agent: Agent, task: Task, scorer: NeuralQScorer | None = None) -> float:
    scorer = scorer or NeuralQScorer.load()
    base_score = utility(agent, task)
    q_score = scorer.score(agent, task)
    risk_gate = 1.0 / (1.0 + 0.08 * max(task.risk_cost, 0.0))
    confidence_gate = 0.75 + 0.25 * task.target_probability
    residual_weight = 0.72 * risk_gate * confidence_gate
    return (1.0 - residual_weight) * base_score + residual_weight * q_score


def rl_guided_behavior_distributed(
    agents: List[Agent],
    tasks: List[Task],
    scorer: NeuralQScorer | None = None,
) -> Dict[str, str]:
    scorer = scorer or NeuralQScorer.load()
    remaining_tasks = {task.task_id: task for task in tasks}
    assignment: Dict[str, str] = {}

    while remaining_tasks:
        bids = []
        for agent in agents:
            if agent.agent_id in assignment:
                continue
            best = max(
                remaining_tasks.values(),
                key=lambda task: (
                    rl_guided_behavior_score(agent, task, scorer),
                    utility(agent, task),
                    task.target_probability,
                    -distance(agent.position, task.position),
                ),
                default=None,
            )
            if best is None:
                continue
            bids.append(
                (
                    rl_guided_behavior_score(agent, best, scorer),
                    utility(agent, best),
                    agent.agent_id,
                    best.task_id,
                )
            )

        if not bids:
            break

        bids.sort(reverse=True)
        _, _, winner_agent, winner_task = bids[0]
        assignment[winner_agent] = winner_task
        remaining_tasks.pop(winner_task, None)

    return assignment


def homogeneous_distributed(agents: List[Agent], tasks: List[Task]) -> Dict[str, str]:
    cloned = deepcopy(agents)
    avg = {
        key: sum(agent.behavior.get(key, 0.0) for agent in agents) / len(agents)
        for key in agents[0].behavior
    }
    for agent in cloned:
        agent.behavior.update(avg)
    return distributed_greedy_assignment(cloned, tasks)


def cbba_assignment(agents: List[Agent], tasks: List[Task]) -> Dict[str, str]:
    remaining = {task.task_id: task for task in tasks}
    assignment: Dict[str, str] = {}
    winners: Dict[str, Tuple[str, float]] = {}

    # Single-task CBBA-style surrogate:
    # local best bids are proposed in parallel and a consensus winner is committed
    # per task before the next round.
    while remaining:
        proposals: Dict[str, Tuple[str, float]] = {}
        for agent in sorted(agents, key=lambda item: item.agent_id):
            if agent.agent_id in assignment:
                continue
            ranked = rank_tasks(agent, remaining.values())
            if not ranked:
                continue
            best = ranked[0]
            bid_value = utility(agent, best)
            incumbent = proposals.get(best.task_id)
            if incumbent is None or bid_value > incumbent[1] or (
                abs(bid_value - incumbent[1]) < 1e-12 and agent.agent_id < incumbent[0]
            ):
                proposals[best.task_id] = (agent.agent_id, bid_value)

        if not proposals:
            break

        committed = False
        for task_id in sorted(
            proposals,
            key=lambda item: (proposals[item][1], -len(item), item),
            reverse=True,
        ):
            candidate_agent, candidate_bid = proposals[task_id]
            incumbent = winners.get(task_id)
            if incumbent is not None and incumbent[1] > candidate_bid:
                continue
            if candidate_agent in assignment:
                continue
            winners[task_id] = (candidate_agent, candidate_bid)
            assignment[candidate_agent] = task_id
            remaining.pop(task_id, None)
            committed = True
            break

        if not committed:
            break
    return assignment


def rl_contextual_bandit_assignment(agents: List[Agent], tasks: List[Task]) -> Dict[str, str]:
    """Greedy assignment using a fixed contextual-bandit Q surrogate.

    The weights emulate an offline-learned rescue policy over local agent-task
    features. This keeps the experiment deterministic while adding a learning-
    based policy comparator with different inductive bias from hand-coded
    utilities and auctions.
    """
    remaining_tasks = {task.task_id: task for task in tasks}
    assignment: Dict[str, str] = {}

    def q_value(agent: Agent, task: Task) -> float:
        is_uav = 1.0 if agent.kind == "UAV" else 0.0
        is_usv = 1.0 if agent.kind == "USV" else 0.0
        travel = distance(agent.position, task.position) / max(agent.speed, 1e-6)
        learned_score = (
            1.18 * task.information_gain
            + 1.05 * task.rescue_value
            + 3.20 * task.target_probability
            + 1.10 * agent.energy
            + 0.72 * is_uav * task.information_gain
            + 0.82 * is_usv * task.rescue_value
            - 0.92 * task.risk_cost
            - 0.58 * travel
        )
        return learned_score

    for agent in sorted(agents, key=lambda item: item.agent_id):
        if not remaining_tasks:
            break
        best_task = max(
            remaining_tasks.values(),
            key=lambda task: (q_value(agent, task), task.target_probability, -distance(agent.position, task.position)),
        )
        assignment[agent.agent_id] = best_task.task_id
        remaining_tasks.pop(best_task.task_id, None)
    return assignment


def neural_q_assignment(agents: List[Agent], tasks: List[Task]) -> Dict[str, str]:
    """Greedy assignment using an offline-trained one-hidden-layer Q network."""
    return greedy_neural_q_assignment(agents, tasks)


def distance_only_assignment(agents: List[Agent], tasks: List[Task]) -> Dict[str, str]:
    remaining_tasks = {task.task_id: task for task in tasks}
    assignment: Dict[str, str] = {}
    for agent in sorted(agents, key=lambda item: item.agent_id):
        if not remaining_tasks:
            break
        best_task = min(
            remaining_tasks.values(),
            key=lambda task: distance(agent.position, task.position),
        )
        assignment[agent.agent_id] = best_task.task_id
        remaining_tasks.pop(best_task.task_id, None)
    return assignment


def centralized_optimal_assignment(agents: List[Agent], tasks: List[Task]) -> Dict[str, str]:
    best_score = None
    best_assignment: Dict[str, str] = {}
    pick_count = min(len(tasks), len(agents))
    for task_subset in combinations(tasks, pick_count):
        for perm in permutations(task_subset, pick_count):
            score = 0.0
            assignment: Dict[str, str] = {}
            for agent, task in zip(agents, perm):
                score += utility(agent, task)
                assignment[agent.agent_id] = task.task_id
            if best_score is None or score > best_score:
                best_score = score
                best_assignment = assignment
    return best_assignment
