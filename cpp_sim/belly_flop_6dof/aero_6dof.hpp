// =============================================================================
// aero_6dof.hpp - 星舰 6-DOF 气动模型 (Phase 11: Python→C++ 移植)
// =============================================================================
// 对应 Python: src/belly_flop/aero_model_6dof.py
//
// 设计决策 (批判性参考 Phase 11 方案):
//   - 拒绝 gen_aero_table.py 查表化: 我们没有真实风洞数据, Python 解析公式
//     就是真理. 把解析公式采样成表再插值回来是纯工程降级 (用户铁律: 拒绝降级).
//   - 保留解析气动公式 (sin²α, sin(2α), Mach sigmoid), 与 Python 严格一致.
//   - 6-DOF 新增: 侧向力 CY(β), 滚转力矩 Cl(β), 偏航力矩 Cn(β).
//   - 4 片襟翼 (FL/FR/RL/RR) 差动分配公式在 control_allocation_6dof.hpp.
//   - 缺陷24: tanh 替代 sigmoid (数值稳定, 已在 belly_flop_aero.hpp 验证).
//
// 物理参数: 与 Python aero_model_6dof.py 严格一致 (星舰构型)
// =============================================================================
#ifndef FALCON9_BELLY_FLOP_6DOF_AERO_HPP
#define FALCON9_BELLY_FLOP_6DOF_AERO_HPP

#include <cmath>
#include <cstdint>

namespace falcon9 {
namespace belly_flop_6dof {

// =============================================================================
// 物理常数 (与 Python aero_model_6dof.py 严格一致)
// =============================================================================
constexpr float DIAMETER    = 9.0f;           // m, 直径
constexpr float L_REF       = 50.0f;          // m, 参考长度
constexpr float S_REF       = 3.14159265358979f * (DIAMETER / 2.0f) * (DIAMETER / 2.0f);  // ≈63.617 m²
constexpr float M_DRY       = 100000.0f;      // kg, 干重
constexpr float M_FUEL_INIT = 50000.0f;       // kg, 初始燃料
constexpr float T_MAX       = 4600e3f;        // N, 最大推力
constexpr float T_IDLE      = 460e3f;         // N, 怠速推力 (FLIP 阶段维持 TVC)
constexpr float ISP         = 380.0f;         // s, 比冲
constexpr float G0_ISP      = 9.80665f;       // Isp 参考重力

constexpr float DELTA_MAX   = 0.523598775598f; // rad, ±30° 襟翼最大偏转

// 襟翼效率系数 (与 Python 一致)
constexpr float C_DELTA_FWD  = 0.8f;   // 前翼俯仰效率
constexpr float C_DELTA_AFT  = 0.6f;   // 后翼俯仰效率
constexpr float C_DELTA_ROLL = 0.5f;   // 差动滚转效率
constexpr float C_DELTA_YAW  = 0.3f;   // 差动偏航效率

// 侧向气动导数 (6-DOF 新增, 工程估算)
constexpr float CYB = -1.2f;   // 侧向力对侧滑角导数 (方向稳定)
constexpr float CLB = -0.3f;   // 滚转力矩导数 (上反效应)
constexpr float CNB =  0.15f;  // 偏航力矩导数 (方向稳定, weathervane)

// 物理常数
constexpr float R_EARTH = 6371000.0f;  // m
constexpr float G0_SL   = 9.80665f;    // m/s²
constexpr float GAMMA_AIR = 1.4f;
constexpr float R_AIR    = 287.05287f;

// 转动惯量基准 (与 Python get_inertia_tensor 一致)
constexpr float IXX_DRY = (1.0f / 12.0f) * M_DRY * (DIAMETER / 2.0f) * (DIAMETER / 2.0f) * 2.0f;
constexpr float IYY_DRY = (1.0f / 12.0f) * M_DRY * L_REF * L_REF;
constexpr float IZZ_DRY = IYY_DRY;  // 轴对称
constexpr float L_FUEL  = L_REF / 3.0f;
constexpr float IXX_FUEL_INIT = (1.0f / 12.0f) * M_FUEL_INIT * (DIAMETER / 2.0f) * (DIAMETER / 2.0f) * 2.0f;
constexpr float IYY_FUEL_INIT = (1.0f / 12.0f) * M_FUEL_INIT * L_FUEL * L_FUEL;
constexpr float IZZ_FUEL_INIT = IYY_FUEL_INIT;

// =============================================================================
// 1976 标准大气 (与 Python atmosphere.py 一致, 复用 belly_flop_aero.hpp 实现)
// =============================================================================
struct AtmosphereResult {
    float rho;      // 密度 [kg/m³]
    float a_sound;  // 声速 [m/s]
    float p;        // 压力 [Pa]
    float T;        // 温度 [K]
};

inline AtmosphereResult atmosphere(float h) {
    AtmosphereResult r;
    if (h < 0.0f) h = 0.0f;
    if (h > 86000.0f) h = 86000.0f;

    // 温度分段
    if (h < 11000.0f) {
        r.T = 288.15f - 0.0065f * h;
    } else if (h < 20000.0f) {
        r.T = 216.65f;
    } else if (h < 32000.0f) {
        r.T = 216.65f + 0.001f * (h - 20000.0f);
    } else {
        r.T = 228.65f + 0.0028f * (h - 32000.0f);
    }

    // 压力 (简化指数衰减)
    const float p0 = 101325.0f;
    const float H_scale = 8500.0f;
    r.p = p0 * std::exp(-h / H_scale);

    // 密度
    r.rho = r.p / (R_AIR * r.T);

    // 声速
    r.a_sound = std::sqrt(GAMMA_AIR * R_AIR * r.T);

    return r;
}

// =============================================================================
// 质量 / 惯量 / 重力 (与 Python 一致)
// =============================================================================
inline float get_mass(float m_fuel) {
    return M_DRY + (m_fuel > 0.0f ? m_fuel : 0.0f);
}

// 缺陷31: 变重力 g(h) = g0 * (R_E/(R_E+h))²
inline float gravity(float h) {
    const float r = R_EARTH / (R_EARTH + h);
    return G0_SL * r * r;
}

// 转动惯量张量 (对角阵, 与 Python get_inertia_tensor 一致)
// Ixx=滚转, Iyy=俯仰, Izz=偏航 (轴对称 Izz=Iyy)
struct InertiaTensor {
    float Ixx, Iyy, Izz;
};

inline InertiaTensor get_inertia_tensor(float m_fuel) {
    float frac = m_fuel / M_FUEL_INIT;
    if (frac < 0.0f) frac = 0.0f;
    InertiaTensor I;
    I.Ixx = IXX_DRY + frac * IXX_FUEL_INIT;
    I.Iyy = IYY_DRY + frac * IYY_FUEL_INIT;
    I.Izz = IZZ_DRY + frac * IZZ_FUEL_INIT;
    return I;
}

// =============================================================================
// 缺陷24: tanh 替代 sigmoid (数值稳定, 已在 belly_flop_aero.hpp 验证)
// sigmoid(x) = 0.5 + 0.5*tanh(x/2)
// =============================================================================
inline float sigmoid_tanh(float x) {
    return 0.5f + 0.5f * std::tanh(0.5f * x);
}

inline void mach_sigmoid_weights(float M, float& w_trans, float& w_super) {
    w_trans = sigmoid_tanh(20.0f * (M - 1.0f));
    w_super = sigmoid_tanh(20.0f * (M - 1.5f));
}

// 气动系数 (与 Python aero_coefficients 一致)
struct AeroCoeffs {
    float CD0, CDa, CLa, Cma;
};

inline AeroCoeffs aero_coefficients(float M) {
    float w_trans, w_super;
    mach_sigmoid_weights(M, w_trans, w_super);
    AeroCoeffs c;
    c.CD0 = 0.3f * (1.0f - w_trans) + 0.8f * w_trans * (1.0f - w_super) + 0.5f * w_super;
    c.CDa = 1.5f * (1.0f - w_trans) + 2.5f * w_trans * (1.0f - w_super) + 2.0f * w_super;
    c.CLa = 1.2f * (1.0f - w_trans) + 0.8f * w_trans * (1.0f - w_super) + 0.6f * w_super;
    c.Cma = 0.15f * (1.0f - w_trans) + 0.25f * w_trans * (1.0f - w_super) + 0.20f * w_super;
    return c;
}

// =============================================================================
// 气流角计算 (6-DOF)
// =============================================================================
struct AirflowAngles {
    float alpha;  // 攻角 [rad]
    float beta;   // 侧滑角 [rad]
    float V;      // 总速度 [m/s]
};

inline AirflowAngles airflow_angles(float u, float v, float w) {
    AirflowAngles a;
    a.V = std::sqrt(u * u + v * v + w * w);
    if (a.V < 1e-6f) {
        a.alpha = 0.0f;
        a.beta = 0.0f;
        return a;
    }
    a.alpha = std::atan2(w, u);
    float v_over_V = v / a.V;
    if (v_over_V > 1.0f)  v_over_V = 1.0f;
    if (v_over_V < -1.0f) v_over_V = -1.0f;
    a.beta = std::asin(v_over_V);
    return a;
}

// =============================================================================
// 6-DOF 气动力/力矩 (与 Python aero_forces_and_moments_6dof 一致)
// =============================================================================
struct AeroResult6DOF {
    float F[3];   // body 系气动力 [Fx, Fy, Fz] (N)
    float M[3];   // body 系气动力矩 [Mx, My, Mz] (N·m)
};

// vel_b: [u, v, w] body 系速度
// h: 高度 (m)
// delta_flaps: [d_FL, d_FR, d_RL, d_RR] (rad)
inline AeroResult6DOF aero_forces_and_moments(float u, float v, float w,
                                              float h,
                                              const float delta_flaps[4]) {
    AeroResult6DOF r;
    r.F[0] = 0.0f; r.F[1] = 0.0f; r.F[2] = 0.0f;
    r.M[0] = 0.0f; r.M[1] = 0.0f; r.M[2] = 0.0f;

    float V = std::sqrt(u * u + v * v + w * w);
    if (V < 1e-6f || h < 0.0f) {
        return r;
    }

    AtmosphereResult atm = atmosphere(h);
    float M_mach = (atm.a_sound > 0.0f) ? (V / atm.a_sound) : 0.0f;

    AirflowAngles af = airflow_angles(u, v, w);
    AeroCoeffs ac = aero_coefficients(M_mach);

    // 气动系数 (与 Python 一致)
    float sin_a = std::sin(af.alpha);
    float sin_2a = std::sin(2.0f * af.alpha);
    float CD = ac.CD0 + ac.CDa * sin_a * sin_a;
    float CL = ac.CLa * sin_2a * 0.5f;
    // 俯仰力矩配平在 85° (与 Python 一致)
    float Cm = -ac.Cma * std::sin(af.alpha - 1.48352986419518f);  // 85° in rad

    // 侧向气动 (6-DOF 新增)
    float CY = CYB * std::sin(af.beta);
    float Cl = CLB * std::sin(af.beta);
    float Cn = CNB * std::sin(af.beta);

    // 动压 × 参考面积
    float Q = 0.5f * atm.rho * V * V * S_REF;

    // body 系气动力
    float cos_a = std::cos(af.alpha);
    r.F[0] = Q * (-CD * cos_a + CL * sin_a);
    r.F[1] = Q * CY;
    r.F[2] = Q * (-CD * sin_a - CL * cos_a);

    // 襟翼力矩 (4 片差动)
    float d_FL = delta_flaps[0];
    float d_FR = delta_flaps[1];
    float d_RL = delta_flaps[2];
    float d_RR = delta_flaps[3];

    // 俯仰: 前翼(FL+FR)·Cδf - 后翼(RL+RR)·Cδa
    float M_flap_pitch = Q * L_REF * (
        C_DELTA_FWD * (d_FL + d_FR) - C_DELTA_AFT * (d_RL + d_RR)
    );
    // 偏航: 左侧(FL+RL) - 右侧(FR+RR)
    float M_flap_yaw = Q * L_REF * C_DELTA_YAW * (
        (d_FL + d_RL) - (d_FR + d_RR)
    );
    // 滚转: 对角线 (FL+RR) - (FR+RL)
    float M_flap_roll = Q * L_REF * C_DELTA_ROLL * (
        (d_FL + d_RR) - (d_FR + d_RL)
    );

    // 总气动力矩 (气动 + 襟翼)
    r.M[0] = Q * L_REF * Cl + M_flap_roll;   // 滚转
    r.M[1] = Q * L_REF * Cm + M_flap_pitch;  // 俯仰
    r.M[2] = Q * L_REF * Cn + M_flap_yaw;    // 偏航

    return r;
}

// =============================================================================
// 配平襟翼 (4 片, 与 Python trim_flaps_6dof 一致)
// 简化: 前后翼等比偏转, 侧向/滚转为 0
// =============================================================================
inline void trim_flaps(float alpha, float mach,
                       float& d_FL, float& d_FR, float& d_RL, float& d_RR) {
    AeroCoeffs ac = aero_coefficients(mach);
    float Cm = -ac.Cma * std::sin(alpha - 1.48352986419518f);  // 85°

    // 等力矩分配: δ_fwd = -Cm/(2·Cδf), δ_aft = -Cm/(2·Cδa)
    float d_fwd = -Cm / (2.0f * C_DELTA_FWD);
    float d_aft = -Cm / (2.0f * C_DELTA_AFT);

    if (d_fwd > DELTA_MAX) d_fwd = DELTA_MAX;
    if (d_fwd < -DELTA_MAX) d_fwd = -DELTA_MAX;
    if (d_aft > DELTA_MAX) d_aft = DELTA_MAX;
    if (d_aft < -DELTA_MAX) d_aft = -DELTA_MAX;

    d_FL = d_fwd; d_FR = d_fwd;
    d_RL = d_aft; d_RR = d_aft;
}

}  // namespace belly_flop_6dof
}  // namespace falcon9

#endif  // FALCON9_BELLY_FLOP_6DOF_AERO_HPP
