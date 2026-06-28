"""
Phase 7.0 战役二: 4片襟翼控制分配开环验证
==========================================
验收标准 (理论方案7.0 问题19):
  1. 禁止pinv伪逆, 必须物理推导
  2. 给定期望力矩, 4片襟翼偏转方向符合物理直觉
  3. 分配->验证闭环: 实际力矩误差<1%
  4. 饱和处理: 归一化分配保持力矩方向
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.belly_flop.control_allocation_6dof import (
    allocate_flaps, allocate_flaps_normalized, verify_allocation,
    ALLOCATION_SIGN,
)
from src.belly_flop.aero_model_6dof import (
    S_REF, L_REF, DELTA_MAX,
    C_DELTA_FWD, C_DELTA_AFT, C_DELTA_ROLL, C_DELTA_YAW,
)


def test_1_pitch_allocation():
    """测试1: 纯俯仰力矩分配 — 前翼正偏(抬头), 后翼负偏(低头)."""
    print("\n=== 测试1: 纯俯仰力矩分配 ===")
    Q = 50000.0  # 动压 (Pa)
    M_cmd = np.array([0.0, 1e6, 0.0])  # 纯俯仰+1e6 N·m (抬头)

    delta = allocate_flaps(M_cmd, Q)
    d_FL, d_FR, d_RL, d_RR = delta

    print(f"  期望力矩: Mx=0, My=+1e6, Mz=0")
    print(f"  FL={np.rad2deg(d_FL):.2f}°, FR={np.rad2deg(d_FR):.2f}°")
    print(f"  RL={np.rad2deg(d_RL):.2f}°, RR={np.rad2deg(d_RR):.2f}°")

    # 物理直觉: 抬头 → 前翼正偏, 后翼负偏
    ok_fwd = d_FL > 0 and d_FR > 0  # 前翼正偏
    ok_aft = d_RL < 0 and d_RR < 0  # 后翼负偏
    # 前左=前右, 后左=后右 (无滚转/偏航)
    ok_sym = abs(d_FL - d_FR) < 1e-10 and abs(d_RL - d_RR) < 1e-10

    print(f"  前翼正偏: {'✓' if ok_fwd else '✗'}")
    print(f"  后翼负偏: {'✓' if ok_aft else '✗'}")
    print(f"  左右对称: {'✓' if ok_sym else '✗'}")

    # 验证力矩
    M_actual, err = verify_allocation(M_cmd, delta, Q)
    err_rel = np.linalg.norm(err) / np.linalg.norm(M_cmd)
    print(f"  实际力矩: {M_actual}")
    print(f"  力矩误差: {err_rel:.2e}")

    result = ok_fwd and ok_aft and ok_sym and err_rel < 0.01
    print(f"  结果: {'PASS' if result else 'FAIL'}")
    return result


def test_2_roll_allocation():
    """测试2: 纯滚转力矩分配 — 对角线差动."""
    print("\n=== 测试2: 纯滚转力矩分配 ===")
    Q = 50000.0
    M_cmd = np.array([1e6, 0.0, 0.0])  # 纯滚转+1e6 N·m

    delta = allocate_flaps(M_cmd, Q)
    d_FL, d_FR, d_RL, d_RR = delta

    print(f"  期望力矩: Mx=+1e6, My=0, Mz=0")
    print(f"  FL={np.rad2deg(d_FL):.2f}°, FR={np.rad2deg(d_FR):.2f}°")
    print(f"  RL={np.rad2deg(d_RL):.2f}°, RR={np.rad2deg(d_RR):.2f}°")

    # 物理直觉: 正滚转 → FL+RR正偏, FR+RL负偏 (对角线差动)
    ok_fl = d_FL > 0
    ok_rr = d_RR > 0
    ok_fr = d_FR < 0
    ok_rl = d_RL < 0

    print(f"  FL+ (对角+): {'✓' if ok_fl else '✗'}")
    print(f"  RR+ (对角+): {'✓' if ok_rr else '✗'}")
    print(f"  FR- (对角-): {'✓' if ok_fr else '✗'}")
    print(f"  RL- (对角-): {'✓' if ok_rl else '✗'}")

    M_actual, err = verify_allocation(M_cmd, delta, Q)
    err_rel = np.linalg.norm(err) / np.linalg.norm(M_cmd)
    print(f"  力矩误差: {err_rel:.2e}")

    result = ok_fl and ok_rr and ok_fr and ok_rl and err_rel < 0.01
    print(f"  结果: {'PASS' if result else 'FAIL'}")
    return result


def test_3_yaw_allocation():
    """测试3: 纯偏航力矩分配 — 左右差动."""
    print("\n=== 测试3: 纯偏航力矩分配 ===")
    Q = 50000.0
    M_cmd = np.array([0.0, 0.0, 1e6])  # 纯偏航+1e6 N·m

    delta = allocate_flaps(M_cmd, Q)
    d_FL, d_FR, d_RL, d_RR = delta

    print(f"  期望力矩: Mx=0, My=0, Mz=+1e6")
    print(f"  FL={np.rad2deg(d_FL):.2f}°, FR={np.rad2deg(d_FR):.2f}°")
    print(f"  RL={np.rad2deg(d_RL):.2f}°, RR={np.rad2deg(d_RR):.2f}°")

    # 物理直觉: 正偏航 → 左侧(FL+RL)正偏, 右侧(FR+RR)负偏
    ok_fl = d_FL > 0
    ok_rl = d_RL > 0
    ok_fr = d_FR < 0
    ok_rr = d_RR < 0

    print(f"  FL+ (左+): {'✓' if ok_fl else '✗'}")
    print(f"  RL+ (左+): {'✓' if ok_rl else '✗'}")
    print(f"  FR- (右-): {'✓' if ok_fr else '✗'}")
    print(f"  RR- (右-): {'✓' if ok_rr else '✗'}")

    M_actual, err = verify_allocation(M_cmd, delta, Q)
    err_rel = np.linalg.norm(err) / np.linalg.norm(M_cmd)
    print(f"  力矩误差: {err_rel:.2e}")

    result = ok_fl and ok_rl and ok_fr and ok_rr and err_rel < 0.01
    print(f"  结果: {'PASS' if result else 'FAIL'}")
    return result


def test_4_coupled_allocation():
    """测试4: 耦合力矩分配 — 俯仰+滚转+偏航同时."""
    print("\n=== 测试4: 耦合力矩分配 ===")
    Q = 50000.0
    M_cmd = np.array([5e5, 1e6, 3e5])  # 滚转+俯仰+偏航

    delta = allocate_flaps(M_cmd, Q)
    M_actual, err = verify_allocation(M_cmd, delta, Q)
    err_rel = np.linalg.norm(err) / np.linalg.norm(M_cmd)

    print(f"  期望力矩: Mx=+5e5, My=+1e6, Mz=+3e5")
    print(f"  FL={np.rad2deg(delta[0]):.2f}°, FR={np.rad2deg(delta[1]):.2f}°")
    print(f"  RL={np.rad2deg(delta[2]):.2f}°, RR={np.rad2deg(delta[3]):.2f}°")
    print(f"  实际力矩: {M_actual}")
    print(f"  力矩误差: {err_rel:.2e}")

    result = err_rel < 0.01
    print(f"  结果: {'PASS' if result else 'FAIL'}")
    return result


def test_5_saturation_normalized():
    """测试5: 饱和归一化 — 大力矩指令等比缩小保持方向."""
    print("\n=== 测试5: 饱和归一化 ===")
    Q = 50000.0
    # 极大力矩指令, 应触发饱和
    M_cmd = np.array([5e7, 1e8, 3e7])

    delta_clip = allocate_flaps(M_cmd, Q)  # 普通钳位
    delta_norm = allocate_flaps_normalized(M_cmd, Q)  # 归一化

    print(f"  期望力矩: Mx=+5e7, My=+1e8, Mz=+3e7")
    print(f"  钳位: FL={np.rad2deg(delta_clip[0]):.2f}°, FR={np.rad2deg(delta_clip[1]):.2f}°")
    print(f"  归一化: FL={np.rad2deg(delta_norm[0]):.2f}°, FR={np.rad2deg(delta_norm[1]):.2f}°")

    # 归一化: 最大偏转=DELTA_MAX, 其他等比
    max_norm = np.max(np.abs(delta_norm))
    ok_max = abs(max_norm - DELTA_MAX) < 1e-10

    # 归一化保持方向: 所有符号正确
    M_actual_norm, _ = verify_allocation(M_cmd, delta_norm, Q)
    # 归一化后力矩方向应与指令一致(虽然幅值小)
    dir_cmd = M_cmd / np.linalg.norm(M_cmd)
    dir_act = M_actual_norm / np.linalg.norm(M_actual_norm)
    cos_angle = np.dot(dir_cmd, dir_act)
    ok_dir = cos_angle > 0.999  # 方向几乎一致

    print(f"  归一化最大偏转=DELTA_MAX: {'✓' if ok_max else '✗'} ({np.rad2deg(max_norm):.2f}°)")
    print(f"  方向保持: {'✓' if ok_dir else '✗'} (cos={cos_angle:.6f})")

    result = ok_max and ok_dir
    print(f"  结果: {'PASS' if result else 'FAIL'}")
    return result


def test_6_no_pinv():
    """测试6: 确认未使用pinv伪逆."""
    print("\n=== 测试6: 禁用pinv验证 ===")
    # 检查ALLOCATION_SIGN矩阵符号正确
    expected = np.array([
        [ 1,  1,  1],
        [-1,  1, -1],
        [-1, -1,  1],
        [ 1, -1, -1],
    ])
    ok = np.array_equal(ALLOCATION_SIGN, expected)
    print(f"  分配矩阵符号: {'✓' if ok else '✗'}")
    print(f"  ALLOCATION_SIGN = \n{ALLOCATION_SIGN}")

    # 读取源文件确认无pinv/lstsq调用
    import os
    src_path = os.path.join(os.path.dirname(__file__), '..', 'src',
                            'belly_flop', 'control_allocation_6dof.py')
    with open(src_path, 'r', encoding='utf-8') as f:
        source = f.read()
    # 检查是否有pinv或lstsq的函数调用(排除注释和docstring中的提及)
    ok_no_pinv = 'np.linalg.pinv' not in source and 'np.linalg.lstsq' not in source
    print(f"  无pinv/lstsq调用: {'✓' if ok_no_pinv else '✗'}")

    result = ok and ok_no_pinv
    print(f"  结果: {'PASS' if result else 'FAIL'}")
    return result


def test_7_zero_moment():
    """测试7: 零力矩指令 — 所有襟翼归零."""
    print("\n=== 测试7: 零力矩指令 ===")
    Q = 50000.0
    M_cmd = np.array([0.0, 0.0, 0.0])

    delta = allocate_flaps(M_cmd, Q)
    ok = np.allclose(delta, 0.0)

    print(f"  期望力矩: [0, 0, 0]")
    print(f"  襟翼偏转: {delta}")
    print(f"  全零: {'✓' if ok else '✗'}")
    print(f"  结果: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == '__main__':
    print("=" * 60)
    print("Phase 7.0 战役二: 4片襟翼控制分配开环验证")
    print("=" * 60)

    results = []
    results.append(('俯仰分配', test_1_pitch_allocation()))
    results.append(('滚转分配', test_2_roll_allocation()))
    results.append(('偏航分配', test_3_yaw_allocation()))
    results.append(('耦合力矩', test_4_coupled_allocation()))
    results.append(('饱和归一化', test_5_saturation_normalized()))
    results.append(('禁用pinv', test_6_no_pinv()))
    results.append(('零力矩', test_7_zero_moment()))

    print("\n" + "=" * 60)
    print("验收汇总:")
    n_pass = 0
    for name, ok in results:
        status = 'PASS' if ok else 'FAIL'
        print(f"  {name}: {status}")
        if ok:
            n_pass += 1
    print(f"\n  {n_pass}/{len(results)} PASS")
    print("=" * 60)
