// =============================================================================
// fixed_matrix.hpp - 编译期定长矩阵模板 (零动态内存)
// 猎鹰9号火箭回收算法 C++ 翻译项目
// 约束: 禁止 new/malloc/vector, 内部使用 std::array
// 所有运算返回值对象 (编译期已知大小, 编译器可优化为栈分配)
// =============================================================================
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>

namespace falcon9 {

// ---------------------------------------------------------------------------
// FixedMatrix<T, ROWS, COLS> - 编译期固定大小矩阵
// 内存布局: 行主序, 一维 std::array<T, ROWS*COLS>
// ---------------------------------------------------------------------------
template <typename T, std::size_t ROWS, std::size_t COLS>
class FixedMatrix {
    static_assert(ROWS > 0 && COLS > 0, "矩阵维度必须为正");
    static_assert(std::is_arithmetic<T>::value, "矩阵元素必须为算术类型");

public:
    using value_type = T;
    using size_type  = std::size_t;
    static constexpr size_type kRows = ROWS;
    static constexpr size_type kCols = COLS;
    static constexpr size_type kSize = ROWS * COLS;

    // 默认构造: 零初始化
    constexpr FixedMatrix() : data_{} {}

    // ------- 元素访问 -------
    // operator()(i,j): 0-indexed, 行主序
    T& operator()(size_type i, size_type j) {
        return data_[i * COLS + j];
    }
    const T& operator()(size_type i, size_type j) const {
        return data_[i * COLS + j];
    }

    // operator[](i): 向量式一维访问 (COLS==1 或 ROWS==1 时便捷)
    T& operator[](size_type i) {
        return data_[i];
    }
    const T& operator[](size_type i) const {
        return data_[i];
    }

    // 原始数据访问
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    // ------- 矩阵运算 -------
    // 加法
    FixedMatrix operator+(const FixedMatrix& rhs) const {
        FixedMatrix result;
        for (size_type i = 0; i < kSize; ++i) {
            result.data_[i] = data_[i] + rhs.data_[i];
        }
        return result;
    }

    // 减法
    FixedMatrix operator-(const FixedMatrix& rhs) const {
        FixedMatrix result;
        for (size_type i = 0; i < kSize; ++i) {
            result.data_[i] = data_[i] - rhs.data_[i];
        }
        return result;
    }

    // 矩阵乘法: A(ROWS×COLS) * B(COLS×K) = C(ROWS×K)
    template <std::size_t K>
    FixedMatrix<T, ROWS, K> operator*(const FixedMatrix<T, COLS, K>& rhs) const {
        FixedMatrix<T, ROWS, K> result;
        for (size_type i = 0; i < ROWS; ++i) {
            for (size_type j = 0; j < K; ++j) {
                T sum = T(0);
                for (size_type k = 0; k < COLS; ++k) {
                    sum += data_[i * COLS + k] * rhs(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }

    // 标量乘
    FixedMatrix operator*(T scalar) const {
        FixedMatrix result;
        for (size_type i = 0; i < kSize; ++i) {
            result.data_[i] = data_[i] * scalar;
        }
        return result;
    }

    // 转置: (ROWS×COLS) -> (COLS×ROWS)
    FixedMatrix<T, COLS, ROWS> transpose() const {
        FixedMatrix<T, COLS, ROWS> result;
        for (size_type i = 0; i < ROWS; ++i) {
            for (size_type j = 0; j < COLS; ++j) {
                result(j, i) = data_[i * COLS + j];
            }
        }
        return result;
    }

    // 单位矩阵 (仅方阵)
    static FixedMatrix Identity() {
        static_assert(ROWS == COLS, "Identity() 仅支持方阵");
        FixedMatrix result;
        for (size_type i = 0; i < ROWS; ++i) {
            result(i, i) = T(1);
        }
        return result;
    }

    // 3x3 矩阵求逆 (伴随矩阵法, 仅对 3x3 方阵启用)
    // 其余尺寸调用将触发 static_assert 编译错误
    FixedMatrix<T, 3, 3> inverse() const {
        static_assert(ROWS == 3 && COLS == 3,
                      "inverse() 仅支持 3x3 矩阵");
        const T a = data_[0], b = data_[1], c = data_[2];
        const T d = data_[3], e = data_[4], f = data_[5];
        const T g = data_[6], h = data_[7], k = data_[8];

        // 行列式
        const T det = a * (e * k - f * h)
                    - b * (d * k - f * g)
                    + c * (d * h - e * g);

        // 伴随矩阵的转置 (即逆矩阵)
        FixedMatrix<T, 3, 3> result;
        // 奇异矩阵保护: 返回零矩阵 (嵌入式环境不抛异常)
        if (std::abs(det) < T(1e-15)) {
            return result;  // 零矩阵
        }
        const T inv_det = T(1) / det;

        result(0, 0) =  (e * k - f * h) * inv_det;
        result(0, 1) = -(b * k - c * h) * inv_det;
        result(0, 2) =  (b * f - c * e) * inv_det;
        result(1, 0) = -(d * k - f * g) * inv_det;
        result(1, 1) =  (a * k - c * g) * inv_det;
        result(1, 2) = -(a * f - c * d) * inv_det;
        result(2, 0) =  (d * h - e * g) * inv_det;
        result(2, 1) = -(a * h - b * g) * inv_det;
        result(2, 2) =  (a * e - b * d) * inv_det;

        return result;
    }

    // 叉乘 (仅 3x1 向量)
    FixedMatrix<T, 3, 1> cross(const FixedMatrix<T, 3, 1>& rhs) const {
        static_assert(ROWS == 3 && COLS == 1,
                      "cross() 仅支持 3x1 向量");
        FixedMatrix<T, 3, 1> result;
        result[0] = data_[1] * rhs[2] - data_[2] * rhs[1];
        result[1] = data_[2] * rhs[0] - data_[0] * rhs[2];
        result[2] = data_[0] * rhs[1] - data_[1] * rhs[0];
        return result;
    }

    // 点乘
    T dot(const FixedMatrix& rhs) const {
        T sum = T(0);
        for (size_type i = 0; i < kSize; ++i) {
            sum += data_[i] * rhs.data_[i];
        }
        return sum;
    }

    // 范数 (L2)
    T norm() const {
        T sum = T(0);
        for (size_type i = 0; i < kSize; ++i) {
            sum += data_[i] * data_[i];
        }
        return std::sqrt(sum);
    }

    // 归一化 (就地)
    void normalize() {
        const T n = norm();
        if (n > T(0)) {
            const T inv_n = T(1) / n;
            for (size_type i = 0; i < kSize; ++i) {
                data_[i] *= inv_n;
            }
        }
    }

private:
    std::array<T, kSize> data_;
};

// ---------------------------------------------------------------------------
// 类型别名 (float32 默认; double 仅用于 EKF 协方差 P)
// ---------------------------------------------------------------------------
using Vec3f  = FixedMatrix<float, 3, 1>;
using Vec4f  = FixedMatrix<float, 4, 1>;
using Mat3f  = FixedMatrix<float, 3, 3>;
using Mat4f  = FixedMatrix<float, 4, 4>;
using Vec15d = FixedMatrix<double, 15, 1>;
using Mat15d = FixedMatrix<double, 15, 15>;

// 标量乘法的对称形式: scalar * matrix
template <typename T, std::size_t R, std::size_t C>
FixedMatrix<T, R, C> operator*(T scalar, const FixedMatrix<T, R, C>& m) {
    return m * scalar;
}

}  // namespace falcon9
