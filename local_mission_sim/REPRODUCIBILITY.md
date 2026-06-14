# Reproducibility Notes

This directory contains the task-level simulator used for the manuscript's
standard, stress, ablation, heterogeneity, scalability, Neural-Q, and visual
explainability experiments.

## Environment

- Python 3.9 or newer
- `numpy`
- `matplotlib`

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## One-command reproduction

```bash
./REPRODUCE_ALL.sh
```

The script runs:

- `python3 -m unittest tests/test_demo.py`
- `python3 train_neural_q.py`
- `python3 run_experiments.py`
- `python3 run_vision_experiments.py`

All generated files are written to `outputs/`.

## Fixed seeds and expected key outputs

The experiment scripts use deterministic seeds in the source code:

- Standard/stress/ablation batches: seed `42`
- Heterogeneity sweep: seed `200`
- Scalability sweep: seed `500`
- Neural-Q training: scenario seeds `8000`, `9000`, and `10000`; hidden-layer seed `3`
- Visual explainability: deterministic scenario generator in `src/msar_sim/vision.py`

Expected key summaries after a full run:

- `outputs/standard/summary.json`: proposed mission time about `26.26`
- `outputs/stress/summary.json`: proposed mission time about `45.21`
- `outputs/ablation/ablation_summary.json`: no-mode-switch mission time about `30.47`
- `outputs/neural_q/training_summary.json`: validation MSE about `0.079`
- `outputs/vision/vision_summary.json`: true region `SEARCH_5`, attention-bid correlation about `0.921`

## Figure and table mapping

- `outputs/standard/robustness_panel.pdf` -> manuscript standard robustness figure
- `outputs/stress/core_results_panel.pdf` -> manuscript stress core figure
- `outputs/stress/robustness_panel.pdf` -> manuscript stress robustness figure
- `outputs/heterogeneity/heterogeneity_sweep.pdf` -> manuscript heterogeneity sweep figure
- `outputs/scalability/scalability_sweep.pdf` -> manuscript scalability sweep figure
- `outputs/ablation/ablation_results_panel.pdf` -> manuscript ablation figure
- `outputs/vision/vision_confidence_allocation.pdf` -> manuscript visual confidence/allocation figure
- `outputs/vision/vision_tracking_radius.pdf` -> manuscript visual tracking/radius figure
- `outputs/vision/vision_attention_bidding.pdf` -> manuscript attention/bidding figure

The generated CSV, JSON, and LaTeX-table files in each output subdirectory are
the raw numerical basis for the reported manuscript tables and figure panels.
