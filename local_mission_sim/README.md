# Local Mission Simulator

This directory contains a task-level simulator for heterogeneous UAV-USV marine search and rescue.

## Components

- Behavior-aware task utility.
- Distributed bidding assignment.
- CBBA-style, centralized, distance-only, homogeneous, RL-bandit, and Neural-Q baselines.
- Situation-triggered verification and circular protection model.
- Visual explainability experiments for target-confidence heatmaps, search-region allocation, multi-view target tracking, adaptive encircling radius, and attention-bid linkage.
- Standard, stress, ablation, heterogeneity, and scalability experiment suites.
- CSV, JSON, LaTeX table, and figure export.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 run_demo.py
python3 run_experiments.py
python3 run_vision_experiments.py
```

Full reproduction entry point:

```bash
./REPRODUCE_ALL.sh
```

See `REPRODUCIBILITY.md` for fixed seeds, expected key outputs, and the mapping
from generated outputs to manuscript figures and tables.

`run_vision_experiments.py` produces deterministic diagnostics for the vision-to-decision interface:

- `vision_confidence_allocation.*`: simulated UAV overhead frame, target-confidence heatmap, information gain, and assigned search regions.
- `vision_tracking_radius.*`: multi-view visual tracking, target drift estimate, adaptive protection radius, and USV slot geometry.
- `vision_attention_bidding.*`: visual attention overlay, learning-augmented bid matrix, and attention-bid consistency plot.
- `vision_summary.json` and `vision_region_table.csv`: numerical summaries used to verify the generated figures.

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
