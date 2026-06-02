import unittest
from pathlib import Path

from src.msar_sim.experiments import run_batch, run_scalability_sweep
from src.msar_sim.formation import circular_formation_targets
from src.msar_sim.scenario import build_demo_scenario, generate_scenario


class DemoPipelineTest(unittest.TestCase):
    def test_demo_pipeline(self) -> None:
        agents, tasks, target = build_demo_scenario()
        self.assertTrue(agents)
        self.assertTrue(tasks)
        slots = circular_formation_targets(target, 3, 25.0, 1.5, 2.0)
        self.assertEqual(len(slots), 3)

    def test_batch_experiments(self) -> None:
        scenario = generate_scenario(seed=10, scenario_id=1)
        self.assertTrue(scenario.true_search_task_id.startswith("SEARCH_"))

        output_dir = Path(__file__).resolve().parent / "tmp_outputs"
        summary = run_batch(num_scenarios=3, output_dir=output_dir, seed=100)
        self.assertIn("behavior_distributed", summary)
        self.assertIn("rl_contextual_bandit", summary)
        self.assertIn("neural_q", summary)
        self.assertTrue((output_dir / "results.csv").exists())
        self.assertTrue((output_dir / "summary.png").exists())

    def test_scalability_sweep(self) -> None:
        output_dir = Path(__file__).resolve().parent / "tmp_scalability"
        summary = run_scalability_sweep(num_scenarios=1, output_dir=output_dir, seed=300)
        self.assertIn("compact", summary)
        self.assertIn("behavior_distributed", summary["compact"])
        self.assertTrue((output_dir / "scalability_sweep.png").exists())


if __name__ == "__main__":
    unittest.main()
