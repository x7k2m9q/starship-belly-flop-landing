"""
星舰 6-DOF 4片襟翼控制分配 (Phase 7.0 战役二)
================================================
理论方案7.0 问题19: 控制分配矩阵病态
  - 4片襟翼控制3个力矩通道(俯仰/偏航/滚转), 存在1维零空间
  - 严禁使用pinv伪逆(前项目教训: 37倍力矩低估)
  - 必须基于物理推导直接分配矩阵

物理推导 (从aero_model_6dof.py的力矩公式反推):
  气动力矩公式:
    Mx = k_roll · (d_FL - d_FR - d_RL + d_RR)    # 滚转
    My = k_fwd · (d_FL + d_FR) - k_aft ·(d_RL + d_RR)  # 俯仰
    Mz = k_yaw · (d_FL - d_FR + d_RL - d_RR)      # 偏航

  其中:
    k_fwd  = Q·L·C_DELTA_FWD   (前翼效率)
    k_aft  = Q·L·C_DELTA_AFT   (后翼效率)
    k_roll = Q·L·C_DELTA_ROLL  (滚转效率)
    k_yaw  = Q·L·C_DELTA_YAW   (偏航效率)

  设: a=d_FL+d_FR, b=d_RL+d_RR, c=d_FL-d_FR, d=d_RL-d_RR
  则:
    Mx/k_roll = c - d
    Mz/k_yaw  = c + d
    My = k_fwd·a - k_aft·b

  解得 (等力矩分配: 前后翼各承担My/2):
    c = (Mx/k_roll + Mz/k_yaw) / 2
    d = (Mz/k_yaw - Mx/k_roll) / 2
    a = My / (2·k_fwd)
    b = -My / (2·k_aft)

  最终:
    d_FL = My/(4·k_fwd)  + Mx/(4·k_roll) + Mz/(4·k_yaw)
    d_FR = My/(4·k_fwd)  - Mx/(4·k_roll) - Mz/(4·k_yaw)
    d_RL = -My/(4·k_aft) - Mx/(4·k_roll) + Mz/(4·k_yaw)
    d_RR = -My/(4·k_aft) + Mx/(4·k_roll) - Mz/(4·k_yaw)

  符号验证 (与方案7.0的D矩阵一致):
    FL: My+, Mx+, Mz+ (俯仰+, 滚转+, 偏航+) ✓
    FR: My+, Mx-, Mz- (俯仰+, 滚转-, 偏航-) ✓
    RL: My-, Mx-, Mz+ (俯仰-, 滚转-, 偏航+) ✓
    RR: My-, Mx+, Mz- (俯仰-, 滚转+, 偏航-) ✓
"""
import numpy as np
from src.belly_flop.aero_model_6dof import (
    S_REF, L_REF, DELTA_MAX,
    C_DELTA_FWD, C_DELTA_AFT, C_DELTA_ROLL, C_DELTA_YAW,
)


# 符号分配矩阵 (4×3, 与方案7.0一致)
# 行: FL, FR, RL, RR
# 列: Mx(滚转), My(俯仰), Mz(偏航)
ALLOCATION_SIGN = np.array([
    [ 1,  1,  1],   # FL: 滚转+, 俯仰+, 偏航+
    [-1,  1, -1],   # FR: 滚转-, 俯仰+, 偏航-
    [-1, -1,  1],   # RL: 滚转-, 俯仰-, 偏航+
    [ 1, -1, -1],   # RR: 滚转+, 俯仰-, 偏航-
], dtype=float)


def allocate_flaps(M_cmd, Q, L=L_REF,
                   C_fwd=C_DELTA_FWD, C_aft=C_DELTA_AFT,
                   C_roll=C_DELTA_ROLL, C_yaw=C_DELTA_YAW):
    """
    4片襟翼控制分配 (物理推导, 禁pinv).

    参数:
      M_cmd: [Mx, My, Mz] 期望力矩 (N·m), body系
        Mx: 滚转力矩 (绕Xb)
        My: 俯仰力矩 (绕Yb)
        Mz: 偏航力矩 (绕Zb)
      Q: 动压 (Pa) = 0.5·ρ·V²·S_REF
      L: 参考长度 (m), 默认L_REF
      C_fwd, C_aft: 前/后翼效率系数
      C_roll, C_yaw: 滚转/偏航效率系数

    返回:
      delta_flaps: [d_FL, d_FR, d_RL, d_RR] (rad), 已钳位到±DELTA_MAX
    """
    Mx, My, Mz = M_cmd

    # 效率系数 (力矩 = k · δ)
    k_fwd  = Q * L * C_fwd
    k_aft  = Q * L * C_aft
    k_roll = Q * L * C_roll
    k_yaw  = Q * L * C_yaw

    # 防除零
    eps = 1e-10
    k_fwd  = max(abs(k_fwd),  eps)
    k_aft  = max(abs(k_aft),  eps)
    k_roll = max(abs(k_roll), eps)
    k_yaw  = max(abs(k_yaw),  eps)

    # 物理推导的分配公式
    d_FL =  My / (4.0 * k_fwd)  + Mx / (4.0 * k_roll) + Mz / (4.0 * k_yaw)
    d_FR =  My / (4.0 * k_fwd)  - Mx / (4.0 * k_roll) - Mz / (4.0 * k_yaw)
    d_RL = -My / (4.0 * k_aft)  - Mx / (4.0 * k_roll) + Mz / (4.0 * k_yaw)
    d_RR = -My / (4.0 * k_aft)  + Mx / (4.0 * k_roll) - Mz / (4.0 * k_yaw)

    delta = np.array([d_FL, d_FR, d_RL, d_RR])

    # 钳位到 ±DELTA_MAX
    delta = np.clip(delta, -DELTA_MAX, DELTA_MAX)

    return delta


def allocate_flaps_normalized(M_cmd, Q, L=L_REF,
                              C_fwd=C_DELTA_FWD, C_aft=C_DELTA_AFT,
                              C_roll=C_DELTA_ROLL, C_yaw=C_DELTA_YAW):
    """
    归一化分配: 如果任一襟翼超限, 等比缩小所有襟翼保持方向.

    用于避免钳位导致的力矩方向偏转.
    """
    Mx, My, Mz = M_cmd

    k_fwd  = max(abs(Q * L * C_fwd),  1e-10)
    k_aft  = max(abs(Q * L * C_aft),  1e-10)
    k_roll = max(abs(Q * L * C_roll), 1e-10)
    k_yaw  = max(abs(Q * L * C_yaw),  1e-10)

    d_FL =  My / (4.0 * k_fwd)  + Mx / (4.0 * k_roll) + Mz / (4.0 * k_yaw)
    d_FR =  My / (4.0 * k_fwd)  - Mx / (4.0 * k_roll) - Mz / (4.0 * k_yaw)
    d_RL = -My / (4.0 * k_aft)  - Mx / (4.0 * k_roll) + Mz / (4.0 * k_yaw)
    d_RR = -My / (4.0 * k_aft)  + Mx / (4.0 * k_roll) - Mz / (4.0 * k_yaw)

    delta = np.array([d_FL, d_FR, d_RL, d_RR])

    # 归一化: 如果超限, 等比缩小
    max_abs = np.max(np.abs(delta))
    if max_abs > DELTA_MAX:
        delta = delta * (DELTA_MAX / max_abs)

    return delta


def verify_allocation(M_cmd, delta, Q, L=L_REF,
                      C_fwd=C_DELTA_FWD, C_aft=C_DELTA_AFT,
                      C_roll=C_DELTA_ROLL, C_yaw=C_DELTA_YAW):
    """
    验证分配: 给定襟翼偏转, 计算实际力矩, 返回误差.

    用于开环测试: allocate -> verify 闭环验证.
    """
    d_FL, d_FR, d_RL, d_RR = delta

    # 正向计算力矩 (与aero_model_6dof.py一致)
    Mx_actual = Q * L * C_roll * (d_FL - d_FR - d_RL + d_RR)
    My_actual = Q * L * (C_fwd * (d_FL + d_FR) - C_aft * (d_RL + d_RR))
    Mz_actual = Q * L * C_yaw * (d_FL - d_FR + d_RL - d_RR)

    M_actual = np.array([Mx_actual, My_actual, Mz_actual])
    error = M_actual - M_cmd

    return M_actual, error
