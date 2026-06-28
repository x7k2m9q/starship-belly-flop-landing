"""
四元数与旋转工具 (全局坐标系约定的唯一实现, 禁止 elsewhere 另写旋转)

约定 (锁死, 与开发日志一致):
  惯性系 NED: X北 Y东 Z地(+). 重力 g_n=[0,0,g].
  箭体系 b: Xb头部, Yb右, Zb=Xb×Yb.
  四元数 q=[w,x,y,z] 表示 b系->n系 旋转. v_n = C_b^n(q) @ v_b  (Hamilton 约定).
  火箭垂直(头朝上): q_vert = [sqrt2/2, 0, sqrt2/2, 0].
  推力沿 +Xb.
"""
import numpy as np

SQRT2 = np.sqrt(2.0)
# 火箭垂直(头部朝上)姿态, b->n. 验证: C@[1,0,0]=[0,0,-1](上)
Q_VERT = np.array([SQRT2 / 2.0, 0.0, SQRT2 / 2.0, 0.0])


def quat_multiply(q1, q2):
    """Hamilton 乘法 q1 ⊗ q2, 返回 [w,x,y,z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_inverse(q):
    # 单位四元数逆 = 共轭
    n = np.dot(q, q)
    if n < 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return quat_conjugate(q) / n


def quat_normalize(q):
    n = np.linalg.norm(q)
    if n < 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_to_rotmat(q):
    """b->n 旋转矩阵 C_b^n, 满足 v_n = C @ v_b. q=[w,x,y,z] Hamilton."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rotmat_to_quat(C):
    """n->? 这里 C 视为 b->n. 返回 q=[w,x,y,z] 使 C_b^n=quat_to_rotmat(q). Shepherd 方法."""
    tr = C[0, 0] + C[1, 1] + C[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (C[2, 1] - C[1, 2]) / s
        y = (C[0, 2] - C[2, 0]) / s
        z = (C[1, 0] - C[0, 1]) / s
    elif (C[0, 0] > C[1, 1]) and (C[0, 0] > C[2, 2]):
        s = np.sqrt(1.0 + C[0, 0] - C[1, 1] - C[2, 2]) * 2.0
        w = (C[2, 1] - C[1, 2]) / s
        x = 0.25 * s
        y = (C[0, 1] + C[1, 0]) / s
        z = (C[0, 2] + C[2, 0]) / s
    elif C[1, 1] > C[2, 2]:
        s = np.sqrt(1.0 + C[1, 1] - C[0, 0] - C[2, 2]) * 2.0
        w = (C[0, 2] - C[2, 0]) / s
        x = (C[0, 1] + C[1, 0]) / s
        y = 0.25 * s
        z = (C[1, 2] + C[2, 1]) / s
    else:
        s = np.sqrt(1.0 + C[2, 2] - C[0, 0] - C[1, 1]) * 2.0
        w = (C[1, 0] - C[0, 1]) / s
        x = (C[0, 2] + C[2, 0]) / s
        y = (C[1, 2] + C[2, 1]) / s
        z = 0.25 * s
    return quat_normalize(np.array([w, x, y, z]))


def quat_error(q_des, q_actual):
    """姿态误差 e_q = q_des ⊗ q_actual^{-1} (b->n约定下, 表示 actual->des 的旋转).
    处理双覆盖: 若 w<0 整体取反. 返回 e_q=[w,x,y,z]."""
    e = quat_multiply(q_des, quat_inverse(q_actual))
    if e[0] < 0.0:
        e = -e
    return e


def quat_error_vec(q_des, q_actual):
    """返回误差四元数矢量部分 [ex,ey,ez] (已处理双覆盖). 用于PD控制."""
    e = quat_error(q_des, q_actual)
    return e[1:4].copy()


def quat_kinematics(q, omega_body):
    """四元数运动学 qdot = 0.5 * q ⊗ [0,omega]. omega_body 为b系角速度(3,)."""
    omega_q = np.array([0.0, omega_body[0], omega_body[1], omega_body[2]])
    return 0.5 * quat_multiply(q, omega_q)


def tilt_angle_from_vertical(q):
    """体X轴与向上方向(世界-Z)的夹角(弧度). 剔除滚转干扰(修复2).
    旧版 2*acos(|<q,Q_VERT>|) 会把纯滚转误算成倾斜, 导致SAFE误触发.
    新版: 直接算体X轴在n系中的指向, 与up=[0,0,-1]的夹角."""
    C = quat_to_rotmat(q)
    x_body_world = C @ np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, -1.0])  # NED向上
    cos_tilt = np.clip(np.dot(x_body_world, up), -1.0, 1.0)
    return np.arccos(cos_tilt)


def body_to_nav(q, v_body):
    return quat_to_rotmat(q) @ v_body


def nav_to_body(q, v_nav):
    return quat_to_rotmat(q).T @ v_nav
