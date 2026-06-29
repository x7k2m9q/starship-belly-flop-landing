# Starship Belly-Flop Recovery 6-DOF Simulation | 星舰腹部翻转回收六自由度仿真

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![MC Success Rate](https://img.shields.io/badge/MC%20Success-100%25-brightgreen.svg)](#monte-carlo-results)
[![Unit Tests](https://img.shields.io/badge/Unit%20Tests-42%2F42-brightgreen.svg)](#test-suite)

> An engineering-grade six-degree-of-freedom simulator for the SpaceX Starship
> "Belly-Flop → Flip → Landing" recovery maneuver, with SCvx guidance,
> 4-flap differential control allocation, Bouc-Wen hysteresis actuators,
> and a 15-state Multiplicative Extended Kalman Filter.

---

## Overview | 项目简介

This repository provides a full 6-DOF engineering model of the Starship
recovery profile. Unlike the Falcon 9 first-stage propulsive landing, the
Starship upper stage uses aerodynamic deceleration in a near-horizontal
"belly-flop" attitude, followed by a rapid flip maneuver and a vertical
powered landing. The simulator covers all three phases with consistent
coordinate conventions, non-ideal actuator dynamics, sensor fusion, and
Monte Carlo robustness verification.

本仓库对星舰回收剖面进行完整的六自由度工程级建模。与猎鹰9号一级垂直推进着陆
不同，星舰采用腹部朝下水平姿态气动减速，随后快速翻转并通过动力下降垂直着陆。
仿真覆盖 BELLY → FLIP → LANDING 三个阶段，统一坐标系约定、非理想执行器、
状态估计与蒙特卡洛鲁棒性验证。

## Key Results | 核心成果

| Metric | Value |
|--------|-------|
| Monte Carlo success rate | **20 / 20 (100%)** |
| Landing velocity | 3.7 ± 0.3 m/s |
| Landing attitude | 13 ± 1.5° |
| Flight time | 65.7 ± 5 s |
| Unit tests | 42 / 42 passed |
| Engineering defects identified & fixed | 10 |

<a id="test-suite"></a>
## Test Suite | 测试套件

The repository ships with 42 unit tests covering physics, control allocation,
attitude control, MEKF, the phase controller, and the non-ideal actuator
model. A 20-run Monte Carlo batch demonstrates closed-loop robustness under
wind perturbations, sensor noise, and actuator hysteresis.

```
tests/
├── test_6dof_physics.py              # DOP853 参考解比对、变重力精度
├── test_control_allocation.py        # 4 襟翼分配矩阵物理推导
├── test_attitude_control_6dof.py     # 四元数 PD、万向锁修复
├── test_actuator_nonideal_6dof.py    # Bouc-Wen 滞环、死区、速率限制
├── test_mekf_6dof.py                 # 15 维 MEKF 收敛性
├── test_phase_controller_6dof.py     # BELLY/FLIP/LANDING 状态机
├── test_monte_carlo_6dof.py          # 20 次蒙特卡洛
├── belly_flop_closedloop.py          # 闭环集成
├── belly_flop_scvx_7c1.py            # SCvx 凸优化制导
├── belly_flop_scvx_7c2.py
├── belly_flop_scvx_7c3.py
├── belly_flop_flip.py                # 翻转机动
├── belly_flop_7e_integration.py      # 集成验证
├── belly_flop_9a1_bias_only.py       # 偏置扫描
├── belly_flop_9a2_9a3.py
├── belly_flop_9b_mc100.py            # 蒙特卡洛批量
└── debug_*.py                        # 调试脚本
```

<a id="monte-carlo-results"></a>
## Monte Carlo Results | 蒙特卡洛验证

Under 20 randomized seeds with wind perturbations, IMU/GPS noise, and
actuator hysteresis enabled, all runs achieve safe landing (velocity
below 10 m/s, attitude below 15°).

| Run | Velocity (m/s) | Attitude (°) | Result |
|-----|----------------|--------------|--------|
| Mean | 3.7 | 13.0 | ✓ |
| Std | 0.3 | 1.5 | ✓ |
| Worst | 4.3 | 15.0 | ✓ |

## System Architecture | 系统架构

```
PhaseController6DOF
├── BELLY  (h: 10km → 3km)   θ≈85°  aerodynamic deceleration, 4 flaps
├── FLIP   (h: ~3km → 500m)  θ: 85° → 0°  bang-bang + PD + feedforward
└── LANDING(h: 2.5km → 0)    θ≈0°   uniform deceleration, TVC + flaps
```

Each phase integrates:
- 14-state dynamics (position 3 + velocity 3 + quaternion 4 + angular rate 3 + fuel 1)
- RK4 integration with quaternion renormalization, dt = 0.01 s
- 4-flap differential control allocation (physics-derived, no pseudo-inverse)
- Bouc-Wen hysteresis + deadband + rate limit per flap
- TVC thrust vectoring with first-order lag and rate limit
- 15-state MEKF with GPS/radar/asynchronous update handling

## Coordinate Conventions | 坐标系约定

| Frame | Definition |
|-------|------------|
| NED (n) | X: north, Y: east, Z: down (gravity along +Z) |
| Body (b) | X_b: nose, Y_b: right, Z_b = X_b × Y_b |
| Quaternion | q = [q_w, q_x, q_y, q_z], b → n rotation, Hamilton product |
| Vertical attitude (θ=0°) | Q_VERT = [√2/2, 0, √2/2, 0] (rotation by 90° about Y) |
| Belly attitude (θ=85°) | Q_BELLY = euler_angle_to_quat(85°) |

## Control Allocation | 控制分配

Four flaps (FL, FR, RL, RR) control three moment channels (pitch, yaw, roll).
The allocation matrix is derived from first principles rather than from
`np.linalg.pinv`, eliminating the conditioning blow-up at low dynamic pressure.

```
[δ_FL]              [ 1  1  1]
[δ_FR] = (1/4) ·    [ 1 -1 -1] · [M_pitch] / (Q · L_ref · C_δ)
[δ_RL]              [-1  1 -1]   [M_yaw  ]
[δ_RR]              [-1 -1  1]   [M_roll ]
```

Flap deflection is bounded to ±30° with a 30°/s rate limit.

## Directory Layout | 目录结构

```
starship-belly-flop-landing/
├── src/
│   ├── belly_flop/                # Python 实现
│   │   ├── dynamics_6dof.py
│   │   ├── aero_model_6dof.py
│   │   ├── control_allocation_6dof.py
│   │   ├── attitude_control_6dof.py
│   │   ├── actuator_nonideal_6dof.py
│   │   ├── phase_controller_6dof.py
│   │   ├── flip_controller.py
│   │   ├── scvx.py                # Successive convexification
│   │   └── ...
│   ├── common/                    # 共享工具
│   │   ├── ekf.py                 # 15-state MEKF
│   │   ├── atmosphere.py
│   │   ├── sensors.py
│   │   ├── flex_dynamics.py
│   │   └── quaternion_utils.py
│   ├── starship_non_ideal.py
│   └── starship_safety_hsm.py
├── cpp_sim/                       # C++ 实时飞控仿真 (嵌入式预研)
│   ├── belly_flop/                # 3-DOF 简化版
│   ├── belly_flop_6dof/           # 6-DOF 工程版
│   │   ├── dynamics_6dof.hpp
│   │   ├── aero_6dof.hpp
│   │   ├── control_allocation_6dof.hpp
│   │   ├── attitude_control_6dof.hpp
│   │   ├── actuator_nonideal_6dof.hpp
│   │   ├── fault_injection.hpp
│   │   ├── flight_computer.hpp
│   │   └── phase_controller_6dof.hpp
│   ├── core/                      # 定点矩阵、四元数
│   ├── hal/                       # 硬件抽象层
│   ├── hal_sim/                   # UDP 仿真接口
│   ├── os/                        # 协作式调度、看门狗、环形缓冲
│   └── tests/
├── tests/                         # Python 测试
├── plots/                         # 仿真结果图
├── docs/
│   └── 星舰论文.md                 # Technical paper
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Quick Start | 快速开始

```bash
# Install dependencies
pip install -r requirements.txt

# Run the 6-DOF Monte Carlo batch (20 runs)
python tests/belly_flop_9b_mc100.py

# Run unit tests
python tests/test_6dof_physics.py
python tests/test_control_allocation.py
python tests/test_attitude_control_6dof.py
python tests/test_mekf_6dof.py
python tests/test_phase_controller_6dof.py

# Run the closed-loop integration test
python tests/belly_flop_closedloop.py
```

## Physical Parameters | 物理参数

| Parameter | Value |
|-----------|-------|
| Dry mass | 100,000 kg |
| Initial fuel | 50,000 kg |
| Diameter | 9 m |
| Height | 50 m |
| Engine | 3× Raptor, T_max = 4600 kN total |
| Isp | 380 s |
| Flap deflection limit | ±30° |
| Flap rate limit | 30°/s |
| TVC deflection limit | ±10° |
| TVC rate limit | 20°/s |
| Simulation step | 0.01 s (100 Hz) |
| IMU frequency | 100 Hz |
| GPS frequency | 10 Hz |
| Radar frequency | 50 Hz (h < 100 m) |

## Engineering Defects Documented | 已记录的工程缺陷

During the 3-DOF → 6-DOF upgrade, 10 critical engineering defects were
identified and fixed. They are documented in the technical paper under
"Defects 1-10" and cover:

1. Physical foundation (DOP853 reference solution for variable-gravity accuracy)
2. Control allocation (physics-derived matrix, no pseudo-inverse)
3. Quaternion PD and gimbal-lock handling
4. Bouc-Wen hysteresis execution order
5. TVC pure-delay startup handling
6. Three-phase controller state synchronization
7. Monte Carlo reproducibility

The full list with reproduction steps is in [`docs/星舰论文.md`](docs/星舰论文.md).

## Theoretical Plan Evolution | 理论方案演进

| Phase | Scope |
|-------|-------|
| Plan 1.x | 3-DOF longitudinal model, basic flap control |
| Plan 7.0 | 3-DOF → 6-DOF upgrade, 14-state vector, quaternion kinematics |
| Plan 7.c | SCvx successive convexification guidance |
| Plan 8.x | Closed-loop integration, phase controller hardening |
| Plan 9.x | Monte Carlo robustness, fault injection, MEKF tuning |

## Citation | 引用

```bibtex
@misc{starship_belly_flop_2026,
  title={Engineering-Grade 6-DOF Modeling and Control for Starship Belly-Flop Recovery: A Systematic Upgrade from 3-DOF Longitudinal to Full-Envelope Six-Degree-of-Freedom},
  author={x7k2m9q},
  year={2026},
  url={https://github.com/x7k2m9q/starship-belly-flop-landing}
}
```

## License | 许可证

MIT License — see [LICENSE](LICENSE).

## References | 参考文献

1. Szmuk, M., Reynolds, T. P., & Açıkmeşe, B. (2020). Successive convexification for real-time six-degree-of-freedom powered descent guidance. *JGCD*, 43(8), 1439–1455.
2. Açıkmeşe, B., & Ploen, S. R. (2007). Convex programming approach to powered descent guidance for Mars landing. *JGCD*, 30(5), 1353–1366.
3. Lefferts, E. J., Markley, F. L., & Shuster, M. D. (1982). Kalman filtering for spacecraft attitude estimation. *JGCD*, 5(5), 417–429.
4. Blackmore, L. (2016). Autonomous precision landing of space rockets. *Nordica Winter School*, 4(1), 1–17.
5. Bouc, R. (1971). Mathematical model for hysteresis. *Acustica*, 24, 16–25.
6. Wen, Y. K. (1976). Method for random vibration of hysteresis systems. *J. Eng. Mech. Div.*, 102(2), 249–263.

---

**Repository**: https://github.com/x7k2m9q/starship-belly-flop-landing
**Author**: x7k2m9q
**Completion Date**: June 2026
