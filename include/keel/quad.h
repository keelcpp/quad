#pragma once

#include "keel/complex.h"
#include "keel/core.h"
#include "keel/function_ref.h"
#include "keel/math.h"
#include "keel/ref.h"

namespace keel::quad {

// ============================================================================
// 15-point Gauss-Kronrod (G7/K15) nodes and weights on [-1, 1]
// Source: Davis & Rabinowitz, "Methods of Numerical Integration".
// ============================================================================

inline constexpr f64 kKronrodX[15] = {
    -0.991455371120813, -0.949107912342759, -0.864864423359769, -0.741531185599394, -0.586087235467691,
    -0.405845151377397, -0.207784955007898, 0.0,                0.207784955007898,  0.405845151377397,
    0.586087235467691,  0.741531185599394,  0.864864423359769,  0.949107912342759,  0.991455371120813,
};

inline constexpr f64 kKronrodW[15] = {
    0.022935322010529, 0.063092092629979, 0.104790010322250, 0.140653259715525, 0.169004726639267,
    0.190350578064785, 0.204432940075298, 0.209482141084728, 0.204432940075298, 0.190350578064785,
    0.169004726639267, 0.140653259715525, 0.104790010322250, 0.063092092629979, 0.022935322010529,
};

inline constexpr f64 kGaussW[7] = {
    0.129484966168870, 0.279705391489277, 0.381830050505119, 0.417959183673469,
    0.381830050505119, 0.279705391489277, 0.129484966168870,
};

// ============================================================================
// Adaptive Gauss-Kronrod Quadrature (1D, 2D, 3D)
// ============================================================================

enum class Status : u8 {
    Ok = 0,
    InvalidBounds,    ///< One or more integration bounds are non-finite
    InvalidTolerance, ///< abs_tol <= 0 or NaN
    InvalidCallback,  ///< Required integrand callback is empty
};

template <typename T> struct Result {
    Status status = Status::Ok;
    T value{};

    [[nodiscard]] constexpr bool ok() const noexcept { return status == Status::Ok; }
};

using Fn1 = FunctionRef<f64(f64)>;
using Fn1C = FunctionRef<c64(f64)>;
using Fn2 = FunctionRef<f64(f64, f64)>;
using Fn2C = FunctionRef<c64(f64, f64)>;
/// Optional vectorized 15x15 (225-point) batch evaluator for 2D quadrature.
/// `xs`, `ys`, and `out` each have length 225, indexed as `j * 15 + i` where
/// `i` indexes the x-node and `j` indexes the y-node.
/// Return `true` if `out` was populated; return `false` to fall back to `scalar_fn`.
using BatchFn2C = FunctionRef<bool(ArrayRef<f64> xs, ArrayRef<f64> ys, MutArrayRef<c64> out)>;
using Fn3 = FunctionRef<f64(f64, f64, f64)>;
using Fn3C = FunctionRef<c64(f64, f64, f64)>;

/// Adaptive 15-point Gauss-Kronrod definite integral over [a, b] (real integrand).
[[nodiscard]] Result<f64> integral(Fn1 fn, f64 a, f64 b, f64 abs_tol = 1e-10, i32 max_depth = 20);

/// Adaptive 15-point Gauss-Kronrod definite integral over [a, b] (complex integrand).
[[nodiscard]] Result<c64> integral_c64(Fn1C fn, f64 a, f64 b, f64 abs_tol = 1e-10, i32 max_depth = 20);

/// Adaptive 2D tensor-product 15x15 Gauss-Kronrod integral over [a, b] x [c, d] (real).
[[nodiscard]] Result<f64> integral2(Fn2 fn, f64 a, f64 b, f64 c, f64 d, f64 abs_tol = 1e-10, i32 max_depth = 18);

/// Adaptive 2D tensor-product 15x15 Gauss-Kronrod integral over [a, b] x [c, d] (complex).
[[nodiscard]] Result<c64> integral2_c64(Fn2C fn, f64 a, f64 b, f64 c, f64 d, f64 abs_tol = 1e-10, i32 max_depth = 18);

/// Adaptive 2D tensor-product 15x15 Gauss-Kronrod integral with an optional
/// 225-point vectorized batch evaluator and a scalar fallback (complex).
[[nodiscard]] Result<c64> integral2_batch_c64(BatchFn2C batch_fn, Fn2C scalar_fn, f64 a, f64 b, f64 c, f64 d,
                                              f64 abs_tol = 1e-10, i32 max_depth = 18);

/// Adaptive 3D nested Gauss-Kronrod integral over [a, b] x [c, d] x [e, f] (real).
[[nodiscard]] Result<f64> integral3(Fn3 fn, f64 a, f64 b, f64 c, f64 d, f64 e, f64 f, f64 abs_tol = 1e-10,
                                    i32 max_depth_outer = 20, i32 max_depth_inner = 18);

/// Adaptive 3D nested Gauss-Kronrod integral over [a, b] x [c, d] x [e, f] (complex).
[[nodiscard]] Result<c64> integral3_c64(Fn3C fn, f64 a, f64 b, f64 c, f64 d, f64 e, f64 f, f64 abs_tol = 1e-10,
                                        i32 max_depth_outer = 20, i32 max_depth_inner = 18);

// ============================================================================
// Discrete Trapezoidal Integration (real & complex, contiguous & strided)
// ============================================================================

/// Trapezoidal integral of `y` along a strided line of length `n`.
/// If `x` is null, unit spacing (dx = 1) is used.
[[nodiscard]] f64 trapz_strided(const f64* y, usize n, usize y_stride, const f64* x = nullptr,
                                usize x_stride = 1) noexcept;

/// Complex trapezoidal integral of `y` along a strided line of length `n` (real `x`).
[[nodiscard]] c64 trapz_strided_c64(const c64* y, usize n, usize y_stride, const f64* x = nullptr,
                                    usize x_stride = 1) noexcept;

/// Contiguous trapezoidal integral. Empty `x` uses unit spacing.
[[nodiscard]] f64 trapz(ArrayRef<f64> y, ArrayRef<f64> x = {}) noexcept;

/// Contiguous complex trapezoidal integral. Empty `x` uses unit spacing.
[[nodiscard]] c64 trapz_c64(ArrayRef<c64> y, ArrayRef<f64> x = {}) noexcept;

/// Cumulative trapezoidal integral along a strided line of length `n`.
/// If `x` is null, unit spacing (dx = 1) is used.
void cumtrapz_strided(const f64* y, f64* dst, usize n, usize y_stride, usize dst_stride, const f64* x = nullptr,
                      usize x_stride = 1) noexcept;

/// Complex cumulative trapezoidal integral along a strided line of length `n` (real `x`).
void cumtrapz_strided_c64(const c64* y, c64* dst, usize n, usize y_stride, usize dst_stride, const f64* x = nullptr,
                          usize x_stride = 1) noexcept;

/// Contiguous cumulative trapezoidal integral (`dst.size()` must be >= `y.size()`).
void cumtrapz(ArrayRef<f64> y, MutArrayRef<f64> dst, ArrayRef<f64> x = {}) noexcept;

/// Contiguous complex cumulative trapezoidal integral (`dst.size()` must be >= `y.size()`).
void cumtrapz_c64(ArrayRef<c64> y, MutArrayRef<c64> dst, ArrayRef<f64> x = {}) noexcept;

// ============================================================================
// Finite-Difference Calculus (gradient, del2)
// ============================================================================

/// Forward/central/backward first derivative along a strided line of length `n`
/// with uniform spacing `h`. A line of length 1 yields 0.
void gradient_strided(const f64* src, f64* dst, usize n, usize stride, f64 h = 1.0) noexcept;

/// Contiguous 1D gradient (`dst.size()` must be >= `src.size()`).
void gradient_1d(ArrayRef<f64> src, MutArrayRef<f64> dst, f64 h = 1.0) noexcept;

/// One-dimensional discrete-Laplacian term along a strided line of length `n`
/// with uniform spacing `h`: centered second difference divided by `2*h*h` on
/// the interior, with linear boundary extrapolation (for `n > 3`), copied from
/// the interior point (for `n == 3`), or zero (for `n < 3`).
void del2_strided(const f64* src, f64* dst, usize n, usize stride, f64 h = 1.0) noexcept;

} // namespace keel::quad
