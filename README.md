# 异构 UAV-USV 协同代码

本仓库包含异构 UAV-USV 海上搜救协同实验的可运行代码。

## 目录说明

- `local_mission_sim/`：任务级 UAV-USV 搜救协同仿真器，包含对比方法、Neural-Q 训练脚本和测试代码。
- `remote_auv6dof_marl/`：用于兼容高保真水下跟踪环境的 AUV-6DOF MARL 运行入口。

## 本地任务级仿真

```bash
cd local_mission_sim
python3 -m pip install -r requirements.txt
python3 run_demo.py
python3 run_experiments.py
python3 run_vision_experiments.py
```

视觉解释性实验：

- `run_vision_experiments.py` 生成视觉目标置信度热力图、搜索区域分配、多视角视觉跟踪、自适应环绕半径和视觉注意力-任务竞价诊断结果。
- 输出位于 `local_mission_sim/outputs/vision/`，包括 `vision_summary.json`、`vision_region_table.csv` 以及对应的 PNG/PDF 图。
- `outputs/` 是可重复生成的过程目录，不纳入 Git 提交。

Neural-Q 训练：

```bash
cd local_mission_sim
PYTHONPATH=. python3 train_neural_q.py
```

单元测试：

```bash
cd local_mission_sim
PYTHONPATH=. python3 -m unittest tests/test_demo.py
```

运行结果会写入 `local_mission_sim/outputs/`。该目录已被 Git 忽略，用于避免重复生成的实验结果影响代码仓库。

## 远程 AUV-6DOF MARL 入口

`remote_auv6dof_marl/auv6dof_marl_entry.py` 用于放置在兼容的 AUV-6DOF/DI-engine 跟踪代码环境中。目标环境需要能够导入 `auv6dof.gym_env.AUV6DOFGymEnv` 或 `Tracking.auv6dof.gym_env.AUV6DOFGymEnv`。

该文件提供可调用的 `run_experiment(...)` 入口，用于生成 checkpoint、训练曲线、评估曲线、评估明细表和配置文件。

## 说明

本地仿真器是任务级对比仿真器，适用于协同策略评估、消融实验、敏感性分析、规模化测试和视觉感知到任务决策链路的解释性诊断。
