# Heterogeneous UAV-USV Code

This repository contains the executable code used for the heterogeneous UAV-USV marine search-and-rescue experiments.

## Contents

- `local_mission_sim/`: task-level UAV-USV mission simulator, baselines, Neural-Q training script, and tests.
- `remote_auv6dof_marl/`: AUV-6DOF MARL entry file for a compatible high-fidelity underwater tracking environment.

## Local Mission Simulator

```bash
cd local_mission_sim
python3 -m pip install -r requirements.txt
python3 run_demo.py
python3 run_experiments.py
```

Neural-Q training:

```bash
cd local_mission_sim
PYTHONPATH=. python3 train_neural_q.py
```

Unit test:

```bash
cd local_mission_sim
PYTHONPATH=. python3 -m unittest tests/test_demo.py
```

Generated outputs are written to `local_mission_sim/outputs/`. This directory is ignored by Git so that regenerated experiment artifacts do not clutter the repository.

## Remote AUV-6DOF MARL Entry

The file `remote_auv6dof_marl/auv6dof_marl_entry.py` is intended to be placed inside a compatible AUV-6DOF/DI-engine tracking codebase where `auv6dof.gym_env.AUV6DOFGymEnv` or `Tracking.auv6dof.gym_env.AUV6DOFGymEnv` is available.

It provides a callable `run_experiment(...)` entry that writes checkpoints, learning curves, evaluation curves, evaluation-detail tables, and configuration files.

## Notes

The local simulator is a task-level comparative simulator. It is designed for controlled coordination-policy evaluation, ablation, sensitivity, and scalability checks. It is not a hydrodynamic simulator, sensor stack, communication stack, or hardware-in-the-loop implementation.
