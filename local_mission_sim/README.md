# Local Mission Simulator

This directory contains a task-level simulator for heterogeneous UAV-USV marine search and rescue.

## Components

- Behavior-aware task utility.
- Distributed bidding assignment.
- CBBA-style, centralized, distance-only, homogeneous, RL-bandit, and Neural-Q baselines.
- Situation-triggered verification and circular protection model.
- Standard, stress, ablation, heterogeneity, and scalability experiment suites.
- CSV, JSON, LaTeX table, and figure export.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 run_demo.py
python3 run_experiments.py
```

## Neural-Q Training

```bash
PYTHONPATH=. python3 train_neural_q.py
```

## Test

```bash
PYTHONPATH=. python3 -m unittest tests/test_demo.py
```

## Outputs

Results are written to `outputs/`. This directory is ignored by Git and can be recreated by running the experiment scripts.
