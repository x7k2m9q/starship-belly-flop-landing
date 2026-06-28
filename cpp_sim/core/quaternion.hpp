// =============================================================================
// quaternion.hpp - 四元数类 (Hamilton 约定)
// 猎鹰9号火箭回收算法 C++ 翻译项目
//
// 约定 (与 Python 仿真严格一致):
//   q = [w, x, y, z], Hamilton convention
//   表示 b系->n系 旋转: v_n = C_b^n(q) @ v_b
//   火箭垂直(头朝上)时: q_vert = [sqrt(2)/2, 0, sqrt(2)/2, 0]
//   推力沿 +Xb
// =============================================================================
#pragma once

#include <cmath>
#include "fixed_matrix.hpp"

namespace falcon9 {

// ===========================================================================
// Quaternion - 四元数类 (Hamilton 约定)
// q = [w, x, y, z], 表示 b系->n系 旋转
// ===========================================================================
class Quaternion {
public:
    float w, x, y, z;

    // 默认构造: 单位四元数 [1, 0, 0, 0]
    Quaternion() : w(1.0f), x(0.0f), y(0.0f), z(0.0f) {}

    // 参数化构造
    Quaternion(float w_, float x_, float y_, float z_)
        : w(w_), x(x_), y(y_), z(z_) {}

    // b->n 旋转矩阵 C_b^n (3x3), 满足 v_n = C @ v_b
    // Hamilton 约定下的标准公式
    Mat3f to_rotmat() const {
        Mat3f C;
        C(0, 0) = 1.0f - 2.0f * (y * y + z * z);
        C(0, 1) = 2.0f * (x * y - w * z);
        C(0, 2) = 2.0f * (x * z + w * y);
        C(1, 0) = 2.0f * (x * y + w * z);
        C(1, 1) = 1.0f - 2.0f * (x * x + z * z);
        C(1, 2) = 2.0f * (y * z - w * x);
        C(2, 0) = 2.0f * (x * z - w * y);
        C(2, 1) = 2.0f * (y * z + w * x);
        C(2, 2) = 1.0f - 2.0f * (x * x + y * y);
        return C;
    }

    // 归一化 (就地)
    void normalize() {
        const float n = std::sqrt(w * w + x * x + y * y + z * z);
        if (n > 1e-15f) {
            const float inv_n = 1.0f / n;
            w *= inv_n;
            x *= inv_n;
            y *= inv_n;
            z *= inv_n;
        }
    }

    // 归一化 (返回新对象)
    Quaternion normalized() const {
        Quaternion q = *this;
        q.normalize();
        return q;
    }

    // 四元数乘法 q = q1 * q2 (Hamilton 约定)
    Quaternion operator*(const Quaternion& rhs) const {
        return Quaternion(
            w * rhs.w - x * rhs.x - y * rhs.y - z * rhs.z,  // w
            w * rhs.x + x * rhs.w + y * rhs.z - z * rhs.y,  // x
            w * rhs.y - x * rhs.z + y * rhs.w + z * rhs.x,  // y
            w * rhs.z + x * rhs.y - y * rhs.x + z * rhs.w   // z
        );
    }

    // 共轭 [w, -x, -y, -z]
    Quaternion conjugate() const {
        return Quaternion(w, -x, -y, -z);
    }

    // 逆 = 共轭 / |q|² (单位四元数逆 = 共轭)
    // 退化保护: |q|² < 1e-15 时返回 [1,0,0,0]
    Quaternion inverse() const {
        const float n2 = w * w + x * x + y * y + z * z;
        if (n2 < 1e-15f) {
            return Quaternion(1.0f, 0.0f, 0.0f, 0.0f);
        }
        const float inv_n2 = 1.0f / n2;
        return Quaternion(w * inv_n2, -x * inv_n2, -y * inv_n2, -z * inv_n2);
    }

    // 从旋转向量(axis-angle)构造
    // q = [cos(θ/2), sin(θ/2)*axis_normalized]
    static Quaternion from_axis_angle(const Vec3f& axis, float angle) {
        const float half = angle * 0.5f;
        const float s = std::sin(half);
        Vec3f n = axis;
        n.normalize();
        return Quaternion(std::cos(half), n[0] * s, n[1] * s, n[2] * s);
    }

    // 从欧拉角构造 (ZYX顺序, yaw-pitch-roll)
    // yaw: 绕Z轴; pitch: 绕Y轴; roll: 绕X轴
    // q = q_z ⊗ q_y ⊗ q_x
    static Quaternion from_euler(float yaw, float pitch, float roll) {
        const float cy = std::cos(yaw * 0.5f);
        const float sy = std::sin(yaw * 0.5f);
        const float cp = std::cos(pitch * 0.5f);
        const float sp = std::sin(pitch * 0.5f);
        const float cr = std::cos(roll * 0.5f);
        const float sr = std::sin(roll * 0.5f);
        return Quaternion(
            cy * cp * cr + sy * sp * sr,  // w
            cy * cp * sr - sy * sp * cr,  // x
            cy * sp * cr + sy * cp * sr,  // y
            sy * cp * cr - cy * sp * sr   // z
        );
    }

    // 倾角 (与垂直方向的夹角, 弧度)
    // tilt = arccos(2 * q.w * q.y)
    // q_vert 时 2*w*y = 2*(√2/2)*(√2/2) = 1, arccos(1) = 0 (垂直)
    float tilt_angle_from_vertical() const {
        float cos_tilt = 2.0f * w * y;
        // clip to [-1, 1] 防止浮点误差导致 acos 域错误
        if (cos_tilt > 1.0f)  cos_tilt = 1.0f;
        if (cos_tilt < -1.0f) cos_tilt = -1.0f;
        return std::acos(cos_tilt);
    }

    // 球面线性插值 (slerp)
    // q0, q1 为单位四元数, t ∈ [0,1]
    // 处理双覆盖: dot < 0 时取反 q1 以保证最短路径
    static Quaternion slerp(const Quaternion& q1, const Quaternion& q2, float t) {
        // 点乘
        float dot = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z;
        if (dot > 1.0f)  dot = 1.0f;
        if (dot < -1.0f) dot = -1.0f;

        // 处理双覆盖: 取反 q2 以保证最短路径
        Quaternion q2_adj = q2;
        if (dot < 0.0f) {
            q2_adj = Quaternion(-q2.w, -q2.x, -q2.y, -q2.z);
            dot = -dot;
        }

        // 近似平行: 线性插值 + 归一化 (避免 sin(θ) 除零)
        if (dot > 0.9995f) {
            Quaternion q(
                q1.w + t * (q2_adj.w - q1.w),
                q1.x + t * (q2_adj.x - q1.x),
                q1.y + t * (q2_adj.y - q1.y),
                q1.z + t * (q2_adj.z - q1.z)
            );
            q.normalize();
            return q;
        }

        const float theta_0     = std::acos(dot);
        const float sin_theta_0 = std::sin(theta_0);
        const float theta       = theta_0 * t;
        const float s0          = std::sin(theta_0 - theta) / sin_theta_0;
        const float s1          = std::sin(theta) / sin_theta_0;
        Quaternion q(
            s0 * q1.w + s1 * q2_adj.w,
            s0 * q1.x + s1 * q2_adj.x,
            s0 * q1.y + s1 * q2_adj.y,
            s0 * q1.z + s1 * q2_adj.z
        );
        q.normalize();
        return q;
    }
};

// ---------------------------------------------------------------------------
// 垂直姿态四元数 (火箭头朝上)
// q_vert = [sqrt(2)/2, 0, sqrt(2)/2, 0] (绕Y轴+90度, b->n)
// 验证: tilt_angle_from_vertical() = arccos(2 * (√2/2) * (√2/2)) = arccos(1) = 0
// C++17 inline 变量: 头文件中定义, 无 ODR 冲突
// ---------------------------------------------------------------------------
inline const Quaternion Q_VERT(0.7071067811865476f, 0.0f, 0.7071067811865476f, 0.0f);

}  // namespace falcon9
