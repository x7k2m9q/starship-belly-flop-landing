// =============================================================================
// dynamics_6dof.hpp - 星舰 6-DOF 动力学 (Phase 11: Python→C++ 移植)
// =============================================================================
// 对应 Python: src/belly_flop/dynamics_6dof.py
//
// 状态向量 (14维):
//   X = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, p, q, r, m_fuel]
//        |--位置(NED)--| |--速度(NED)--| |----四元数----| |--角速度(b)--| |燃料|
//
// 坐标系 (与猎鹰9号一致, 锁死):
//   NED系 (n系): X=北, Y=东, Z=地(向下为正). 重力 g_n=[0,0,g].
//   箭体系 (b系): Xb=头部, Yb=右, Zb=Xb×Yb.
//   四元数 q=[w,x,y,z]: b系->n系旋转. v_n = C_b^n(q) @ v_b.
//   推力沿 +Xb (头部方向, 减速时朝下).
//
// 关键物理 (与 Python 严格一致):
//   1. 气动力在 body 系计算 (由 vel_b = C_bn^T @ vel_n 得到气流角)
//   2. 推力在 body 系沿 +Xb, 含 TVC 偏转; 转到 NED 系
//   3. 重力在 NED 系 [0,0,m*g]
//   4. 四元数运动学: qdot = 0.5 * q ⊗ [0, omega_b]
//   5. 欧拉方程: I * domega/dt = M - omega × (I * omega)   ← 暗礁33: 陀螺耦合
//   6. 质心漂移: 燃料消耗导致惯量变化
//
// 暗礁清单 (批判性参考 Phase 11 方案):
//   暗礁31: 变重力 g(h) = g0 * (R_E/(R_E+h))²  ← 已在 aero_6dof.hpp 实现
//   暗礁32: RK4 每步后四元数归一化 (防数值漂移)
//   暗礁33: 陀螺耦合项 omega × (I*omega) 必须保留 (禁简化为 I*domega=M)
//
// 猎鹰9号 C++ 移植血泪教训 (开发日志.md):
//   "omega 反馈通道丢失, PD 退化为 P, 姿态发散 (0.6°→65.8° at t=14)"
//   → 此处 omega 必须从 state 参数读取, 严禁使用外部缓存或置零.
//   → state_derivative(s, ...) 内部 omega = s.omega_b(), 完全由调用方控制.
//   → RK4 四个子步 k1..k4 各自传入不同的 state (含不同 omega), 物理正确.
//
// 积分: RK4, dt=0.01s (定步长, 拒绝变步长 — RK4 数值稳定性要求)
// =============================================================================
#ifndef FALCON9_BELLY_FLOP_6DOF_DYNAMICS_HPP
#define FALCON9_BELLY_FLOP_6DOF_DYNAMICS_HPP

#include <cmath>
#include "aero_6dof.hpp"
#include "../core/fixed_matrix.hpp"
#include "../core/quaternion.hpp"

namespace falcon9 {
namespace belly_flop_6dof {

// =============================================================================
// 14 维状态结构体
// 内部存储为 float[14], 与 Python state[14] 严格对齐, 便于 RK4 通用运算.
// 提供命名视图访问器 (pos_n/vel_n/q/omega_b/m_fuel) 提升可读性.
// =============================================================================
struct State6DOF {
    float data[14];
    // 索引: 0-2 pos_n, 3-5 vel_n, 6-9 quat[w,x,y,z], 10-12 omega_b, 13 m_fuel

    // ---- 索引访问 (RK4 通用化) ----
    float& operator[](int i) { return data[i]; }
    float  operator[](int i) const { return data[i]; }

    // ---- 命名视图访问器 (返回值拷贝, 避免 dangling) ----
    Vec3f pos_n() const {
        Vec3f v; v[0] = data[0]; v[1] = data[1]; v[2] = data[2]; return v;
    }
    Vec3f vel_n() const {
        Vec3f v; v[0] = data[3]; v[1] = data[4]; v[2] = data[5]; return v;
    }
    Quaternion q() const {
        return Quaternion(data[6], data[7], data[8], data[9]);
    }
    Vec3f omega_b() const {
        Vec3f v; v[0] = data[10]; v[1] = data[11]; v[2] = data[12]; return v;
    }
    float m_fuel() const { return data[13]; }

    // ---- 写入器 (用于初始状态构造) ----
    void set_pos_n(float px, float py, float pz) {
        data[0] = px; data[1] = py; data[2] = pz;
    }
    void set_vel_n(float vx, float vy, float vz) {
        data[3] = vx; data[4] = vy; data[5] = vz;
    }
    void set_q(const Quaternion& q) {
        data[6] = q.w; data[7] = q.x; data[8] = q.y; data[9] = q.z;
    }
    void set_omega_b(float p, float q, float r) {
        data[10] = p; data[11] = q; data[12] = r;
    }
    void set_m_fuel(float m) { data[13] = m; }

    // ---- RK4 算术运算 (逐元素, 14 维) ----
    State6DOF operator+(const State6DOF& rhs) const {
        State6DOF r;
        for (int i = 0; i < 14; ++i) r.data[i] = data[i] + rhs.data[i];
        return r;
    }
    State6DOF operator*(float s) const {
        State6DOF r;
        for (int i = 0; i < 14; ++i) r.data[i] = data[i] * s;
        return r;
    }
};

// 标量乘法对称形式: s * state
inline State6DOF operator*(float s, const State6DOF& st) {
    return st * s;
}

// =============================================================================
// Belly-Flop 欧拉角 → 四元数 (与 Python euler_angle_to_quat 一致)
// =============================================================================
// 星舰 Belly-Flop 俯仰角定义 (与 3DOF 一致):
//   θ=0°:  火箭垂直, 头朝上 → Q_VERT (绕 Y 轴 90°)
//   θ=90°: 火箭水平, 腹部朝下 → identity (绕 Y 轴 0°)
// 即: 绕 Y 轴的旋转角度 = 90° - θ_pitch
//
// 旋转顺序: ZYX (yaw->pitch->roll), q = q_yaw ⊗ q_pitch ⊗ q_roll
// 注意: 这里的 "pitch" 是绕 Y 轴的实际旋转, 与 Belly-Flop 的 θ 反向.
// =============================================================================
inline Quaternion euler_angle_to_quat(float theta_pitch_deg,
                                       float phi_roll_deg = 0.0f,
                                       float psi_yaw_deg = 0.0f) {
    constexpr float DEG2RAD = 0.017453292519943295f;
    // 绕 Y 轴旋转 (90°-θ): θ=0→90°旋转(垂直), θ=90→0°旋转(水平)
    float theta_rot = DEG2RAD * (90.0f - theta_pitch_deg);
    float phi = DEG2RAD * phi_roll_deg;
    float psi = DEG2RAD * psi_yaw_deg;

    // 绕 Y 轴(俯仰): [cos(theta_rot/2), 0, sin(theta_rot/2), 0]
    Quaternion q_pitch(std::cos(theta_rot * 0.5f), 0.0f,
                       std::sin(theta_rot * 0.5f), 0.0f);
    // 绕 X 轴(滚转): [cos(phi/2), sin(phi/2), 0, 0]
    Quaternion q_roll(std::cos(phi * 0.5f), std::sin(phi * 0.5f), 0.0f, 0.0f);
    // 绕 Z 轴(偏航): [cos(psi/2), 0, 0, sin(psi/2)]
    Quaternion q_yaw(std::cos(psi * 0.5f), 0.0f, 0.0f, std::sin(psi * 0.5f));

    Quaternion q = q_yaw * q_pitch * q_roll;
    q.normalize();
    return q;
}

// =============================================================================
// 从四元数提取 Belly-Flop 俯仰角 (与 Python get_pitch_angle_from_quat 一致)
// =============================================================================
// 体 X 轴在 NED 系中的方向: x_body_n = C_bn @ [1,0,0]
// 俯仰角 = x_body_n 与水平面(XY)的夹角
//   theta = atan2(horizontal, -x_body_n[z])
// theta=0°: 垂直(头朝上), x_body_n=[0,0,-1] (向上)
// theta=90°: 水平(腹部朝下), x_body_n=[1,0,0] (向北)
// =============================================================================
inline float get_pitch_angle_from_quat(const Quaternion& q) {
    Mat3f C = q.to_rotmat();
    // x_body_n = C @ [1,0,0] = C 的第一列
    float xb_n_x = C(0, 0);
    float xb_n_y = C(1, 0);
    float xb_n_z = C(2, 0);
    float horizontal = std::sqrt(xb_n_x * xb_n_x + xb_n_y * xb_n_y);
    return std::atan2(horizontal, -xb_n_z);
}

// =============================================================================
// 从四元数提取倾角 (与 Python get_tilt_angle_from_quat 一致)
// =============================================================================
// tilt = 体 X 轴与垂直方向(上)的夹角.
// tilt=0°: 垂直. tilt=90°: 水平.
// 与 get_pitch_angle_from_quat 在无滚转时一致.
// =============================================================================
inline float get_tilt_angle_from_quat(const Quaternion& q) {
    Mat3f C = q.to_rotmat();
    // x_body_n = C @ [1,0,0] = C 的第一列
    float xb_n_x = C(0, 0);
    float xb_n_y = C(1, 0);
    float xb_n_z = C(2, 0);
    // up = [0, 0, -1] (NED 向上)
    float cos_tilt = -xb_n_z;
    if (cos_tilt > 1.0f)  cos_tilt = 1.0f;
    if (cos_tilt < -1.0f) cos_tilt = -1.0f;
    return std::acos(cos_tilt);
}

// =============================================================================
// 四元数运动学: qdot = 0.5 * q ⊗ [0, omega_b]
// 返回 [w,x,y,z] 的导数 (4 维)
// =============================================================================
inline void quat_kinematics(const Quaternion& q, const Vec3f& omega,
                             float dq[4]) {
    // q ⊗ [0, p, q, r]
    // Hamilton 乘法:
    //   w' = w*0 - x*p - y*q - z*r = -x*p - y*q - z*r
    //   x' = w*p + x*0 + y*r - z*q =  w*p + y*r - z*q
    //   y' = w*q - x*r + y*0 + z*p =  w*q - x*r + z*p
    //   z' = w*r + x*q - y*p + z*0 =  w*r + x*q - y*p
    // qdot = 0.5 * q ⊗ [0, omega]
    float p = omega[0], qq = omega[1], r = omega[2];
    dq[0] = 0.5f * (-q.x * p - q.y * qq - q.z * r);
    dq[1] = 0.5f * ( q.w * p + q.y * r - q.z * qq);
    dq[2] = 0.5f * ( q.w * qq - q.x * r + q.z * p);
    dq[3] = 0.5f * ( q.w * r + q.x * qq - q.y * p);
}

// =============================================================================
// 6-DOF 状态导数 (与 Python state_derivative_6dof 严格一致)
// =============================================================================
// 参数:
//   state: 14 维状态
//   T_cmd: 推力指令 (N), 沿 +Xb (实际推力, 点火瞬态在外部处理)
//   delta_flaps: [d_FL, d_FR, d_RL, d_RR] 4 片襟翼偏转 (rad)
//   tvc_gimbal: [gimbal_y, gimbal_z] TVC 偏转 (rad), 绕 Yb 和 Zb
//   M_external: 外部力矩 (body 系, N·m), 用于测试/扰动注入 (可为 nullptr)
// 返回: dstate/dt (14 维)
//
// 物理推导:
//   1. body 系速度: vel_b = C_bn^T @ vel_n
//   2. 气动力/力矩: 由 aero_forces_and_moments(u,v,w,h,delta) 计算 (body 系)
//   3. 推力 (body 系, 含 TVC):
//      T_dir_b = [cos(gy)*cos(gz), sin(gz), sin(gy)*cos(gz)]
//      F_thrust_b = T * T_dir_b
//   4. 推力力矩 (TVC 偏转): M = r_TVC × F_thrust, r_TVC = [-L_REF*0.4, 0, 0]
//   5. 总力/力矩 (body 系): F = F_aero + F_thrust, M = M_aero + M_thrust + M_ext
//   6. 转到 NED 系: F_n = C_bn @ F_b
//   7. 重力 (NED 系): [0, 0, m*g]
//   8. 运动方程:
//      dp/dt = v
//      dv/dt = (F_n + F_gravity) / m
//      dq/dt = 0.5 * q ⊗ [0, omega]   ← 四元数运动学
//      domega/dt = I^{-1} (M - omega × (I*omega))   ← 欧拉方程 (暗礁33)
//      dm/dt = -T / (ISP * G0_ISP)
// =============================================================================
inline State6DOF state_derivative(const State6DOF& state, float T_cmd,
                                   const float delta_flaps[4],
                                   const float tvc_gimbal[2],
                                   const float M_external[3] = nullptr) {
    State6DOF d;
    for (int i = 0; i < 14; ++i) d.data[i] = 0.0f;

    // 解包状态
    Vec3f pos_n = state.pos_n();
    Vec3f vel_n = state.vel_n();
    Quaternion q = state.q();
    Vec3f omega_b = state.omega_b();   // ← 猎鹰9号教训: 必须从 state 读取
    float m_fuel = state.m_fuel();

    // 质量/惯量/重力
    float m = get_mass(m_fuel);
    InertiaTensor I = get_inertia_tensor(m_fuel);
    float h = -pos_n[2];                 // 高度 = -pz (NED: pz 负=高度正)
    float g = gravity(h);                // 暗礁31: 变重力

    // 旋转矩阵 C_bn (b->n)
    Mat3f C_bn = q.to_rotmat();
    // C_nb (n->b) = C_bn^T
    Mat3f C_nb = C_bn.transpose();

    // body 系速度: vel_b = C_nb @ vel_n
    Vec3f vel_b = C_nb * vel_n;
    float u = vel_b[0], v = vel_b[1], w = vel_b[2];

    // ---- 气动力/力矩 (6DOF, body 系) ----
    AeroResult6DOF aero = aero_forces_and_moments(u, v, w, h, delta_flaps);
    Vec3f F_aero_b;
    F_aero_b[0] = aero.F[0]; F_aero_b[1] = aero.F[1]; F_aero_b[2] = aero.F[2];
    Vec3f M_aero_b;
    M_aero_b[0] = aero.M[0]; M_aero_b[1] = aero.M[1]; M_aero_b[2] = aero.M[2];

    // ---- 推力 (body 系, 含 TVC 偏转) ----
    // TVC 偏转: gimbal_y>0 = 俯仰抬头(正力矩), gimbal_z>0 = 偏航
    // 推力方向: 绕 Yb 旋转 gy, 绕 Zb 旋转 gz
    // 正 gy → 推力向 +Zb 偏 → 尾部被推向 +Zb(下) → 抬头(正俯仰)
    float gy = tvc_gimbal[0];
    float gz = tvc_gimbal[1];
    float T_dir_b_x = std::cos(gy) * std::cos(gz);
    float T_dir_b_y = std::sin(gz);
    float T_dir_b_z = std::sin(gy) * std::cos(gz);
    float T_actual = T_cmd;   // 实际推力 (点火瞬态在外部处理)
    Vec3f F_thrust_b;
    F_thrust_b[0] = T_actual * T_dir_b_x;
    F_thrust_b[1] = T_actual * T_dir_b_y;
    F_thrust_b[2] = T_actual * T_dir_b_z;

    // 推力力矩 (TVC 偏转产生): M = r_TVC × F_thrust
    // 推力作用点在质心后方 x_TVC = -L_REF*0.4 (尾部)
    float x_tvc = -L_REF * 0.4f;
    Vec3f r_tvc_b;
    r_tvc_b[0] = x_tvc; r_tvc_b[1] = 0.0f; r_tvc_b[2] = 0.0f;
    Vec3f M_thrust_b = r_tvc_b.cross(F_thrust_b);

    // ---- 总力/力矩 (body 系) ----
    Vec3f F_total_b = F_aero_b + F_thrust_b;
    Vec3f M_total_b = M_aero_b + M_thrust_b;
    if (M_external != nullptr) {
        M_total_b[0] += M_external[0];
        M_total_b[1] += M_external[1];
        M_total_b[2] += M_external[2];
    }

    // ---- 转到 NED 系 ----
    Vec3f F_total_n = C_bn * F_total_b;
    // 重力 (NED 系): [0, 0, m*g]
    Vec3f F_gravity_n;
    F_gravity_n[0] = 0.0f; F_gravity_n[1] = 0.0f; F_gravity_n[2] = m * g;

    // ---- 运动方程 ----
    // 位置导数 = 速度
    d.data[0] = vel_n[0];
    d.data[1] = vel_n[1];
    d.data[2] = vel_n[2];

    // 速度导数 = (F_aero + F_thrust)/m + gravity (NED 系)
    d.data[3] = (F_total_n[0] + F_gravity_n[0]) / m;
    d.data[4] = (F_total_n[1] + F_gravity_n[1]) / m;
    d.data[5] = (F_total_n[2] + F_gravity_n[2]) / m;

    // 四元数运动学: qdot = 0.5 * q ⊗ [0, omega_b]
    float dq[4];
    quat_kinematics(q, omega_b, dq);
    d.data[6]  = dq[0];
    d.data[7]  = dq[1];
    d.data[8]  = dq[2];
    d.data[9]  = dq[3];

    // 欧拉方程: I * domega/dt = M - omega × (I * omega)   ← 暗礁33: 陀螺耦合
    // I 为对角阵, I*omega 可逐元素计算; I^{-1} 同理
    Vec3f I_omega;
    I_omega[0] = I.Ixx * omega_b[0];
    I_omega[1] = I.Iyy * omega_b[1];
    I_omega[2] = I.Izz * omega_b[2];
    Vec3f gyro_couple = omega_b.cross(I_omega);   // omega × (I*omega)
    Vec3f domega;
    domega[0] = (M_total_b[0] - gyro_couple[0]) / I.Ixx;
    domega[1] = (M_total_b[1] - gyro_couple[1]) / I.Iyy;
    domega[2] = (M_total_b[2] - gyro_couple[2]) / I.Izz;
    d.data[10] = domega[0];
    d.data[11] = domega[1];
    d.data[12] = domega[2];

    // 燃料消耗
    d.data[13] = -T_actual / (ISP * G0_ISP);

    return d;
}

// =============================================================================
// RK4 单步积分 (与 Python rk4_step_6dof 严格一致)
// =============================================================================
// 每步结束后:
//   1. 四元数归一化 (暗礁32: 防数值漂移)
//   2. 燃料非负
// =============================================================================
inline State6DOF rk4_step(const State6DOF& state, float T_cmd,
                           const float delta_flaps[4], float dt,
                           const float tvc_gimbal[2] = nullptr,
                           const float M_external[3] = nullptr) {
    // 默认 TVC 偏转为 0
    float gimbal_default[2] = {0.0f, 0.0f};
    const float* gimbal = (tvc_gimbal != nullptr) ? tvc_gimbal : gimbal_default;

    State6DOF k1 = state_derivative(state, T_cmd, delta_flaps, gimbal, M_external);
    State6DOF k2 = state_derivative(state + 0.5f * dt * k1, T_cmd, delta_flaps, gimbal, M_external);
    State6DOF k3 = state_derivative(state + 0.5f * dt * k2, T_cmd, delta_flaps, gimbal, M_external);
    State6DOF k4 = state_derivative(state + dt * k3, T_cmd, delta_flaps, gimbal, M_external);

    State6DOF new_state = state + (dt / 6.0f) * (k1 + 2.0f * k2 + 2.0f * k3 + k4);

    // 暗礁32: 四元数归一化 (防数值漂移)
    Quaternion q_new(new_state.data[6], new_state.data[7],
                     new_state.data[8], new_state.data[9]);
    q_new.normalize();
    new_state.data[6] = q_new.w;
    new_state.data[7] = q_new.x;
    new_state.data[8] = q_new.y;
    new_state.data[9] = q_new.z;

    // 燃料非负
    if (new_state.data[13] < 0.0f) new_state.data[13] = 0.0f;

    return new_state;
}

// =============================================================================
// 创建 6-DOF 初始状态 (与 Python make_initial_state_6dof 一致)
// =============================================================================
// 默认: 10km 高度, 300m/s 下降, 50m/s 水平, 85° 俯仰 (BELLY)
// =============================================================================
inline State6DOF make_initial_state(float h_init = 10000.0f,
                                     float vz_init = 300.0f,
                                     float vx_init = 50.0f,
                                     float theta_pitch_deg = 85.0f,
                                     float m_fuel = -1.0f) {
    State6DOF s;
    for (int i = 0; i < 14; ++i) s.data[i] = 0.0f;

    // NED 位置: pz = -h (高度取负)
    s.set_pos_n(0.0f, 0.0f, -h_init);
    // NED 速度: vz 正=下降
    s.set_vel_n(vx_init, 0.0f, vz_init);
    // 四元数: theta_pitch_deg 俯仰
    s.set_q(euler_angle_to_quat(theta_pitch_deg));
    // 角速度: 0
    s.set_omega_b(0.0f, 0.0f, 0.0f);
    // 燃料: 默认 70%
    float m_fuel_init = (m_fuel < 0.0f) ? (M_FUEL_INIT * 0.7f) : m_fuel;
    s.set_m_fuel(m_fuel_init);

    return s;
}

}  // namespace belly_flop_6dof
}  // namespace falcon9

#endif  // FALCON9_BELLY_FLOP_6DOF_DYNAMICS_HPP
