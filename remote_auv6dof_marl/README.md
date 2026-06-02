# Remote AUV-6DOF MARL Entry

This directory contains an executable entry file for a compatible AUV-6DOF/DI-engine underwater tracking environment.

## File

- `auv6dof_marl_entry.py`

## Expected Environment

The entry file expects one of the following imports to be available in the target codebase:

```python
from auv6dof.gym_env import AUV6DOFGymEnv
```

or:

```python
from Tracking.auv6dof.gym_env import AUV6DOFGymEnv
```

## Usage

Place `auv6dof_marl_entry.py` in the target tracking project and call:

```python
from auv6dof_marl_entry import run_experiment
```

The entry writes checkpoints, learning curves, evaluation curves, evaluation-detail tables, and configuration files under `artifacts/auv6dof_marl/` by default.

## Scope

This file is intended for reproducibility and interface checks in a compatible high-fidelity AUV tracking environment. Full benchmark claims should use long-horizon multi-seed training with the target project configuration documented separately.
