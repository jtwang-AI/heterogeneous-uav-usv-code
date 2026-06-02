from src.msar_sim.baselines import behavior_distributed
from src.msar_sim.formation import circular_formation_targets
from src.msar_sim.scenario import build_demo_scenario


def main() -> None:
    agents, tasks, target = build_demo_scenario()
    assignment = behavior_distributed(agents, tasks)

    print("=== Assignment Result ===")
    for agent_id, task_id in assignment.items():
        print(f"{agent_id} -> {task_id}")

    usvs = [agent for agent in agents if agent.kind == "USV"]
    slots = circular_formation_targets(
        center=target,
        num_agents=len(usvs),
        base_radius=25.0,
        target_speed=1.8,
        sea_state=3.0,
    )

    print("\n=== Formation Targets ===")
    for idx, slot in enumerate(slots):
        print(f"slot_{idx}: ({slot[0]:.2f}, {slot[1]:.2f})")


if __name__ == "__main__":
    main()
