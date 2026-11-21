#include "keel/quad.h"

namespace keel::quad {

namespace {

// ============================================================================
// 1D Adaptive Gauss-Kronrod G7/K15
// ============================================================================

struct Gk15PairF64 {
    f64 k;
    f64 g;
};

struct Gk15PairC64 {
    c64 k;
    c64 g;
};

[[nodiscard]] Gk15PairF64 gk15_f64(Fn1 fn, f64 a, f64 b) {
    const f64 half = 0.5 * (b - a);
    const f64 mid = 0.5 * (b + a);
    f64 k_sum = 0.0;
    f64 g_sum = 0.0;
    for (usize i = 0; i < 15; ++i) {
        const f64 x = mid + half * kKronrodX[i];
        const f64 fv = fn(x);
        k_sum += kKronrodW[i] * fv;
        if ((i & 1u) == 1u) {
            g_sum += kGaussW[i / 2] * fv;
        }
    }
    return {half * k_sum, half * g_sum};
}

[[nodiscard]] f64 adaptive_1d_f64(Fn1 fn, f64 a, f64 b, f64 abs_tol, i32 depth, i32 max_depth) {
    const Gk15PairF64 p = gk15_f64(fn, a, b);
    const f64 err = abs(p.k - p.g);
    if (err < abs_tol || depth >= max_depth) {
        return p.k;
    }
    const f64 mid = 0.5 * (a + b);
    return adaptive_1d_f64(fn, a, mid, abs_tol * 0.5, depth + 1, max_depth) +
           adaptive_1d_f64(fn, mid, b, abs_tol * 0.5, depth + 1, max_depth);
}

[[nodiscard]] Gk15PairC64 gk15_c64(Fn1C fn, f64 a, f64 b) {
    const f64 half = 0.5 * (b - a);
    const f64 mid = 0.5 * (b + a);
    c64 k_sum(0.0, 0.0);
    c64 g_sum(0.0, 0.0);
    for (usize i = 0; i < 15; ++i) {
        const f64 x = mid + half * kKronrodX[i];
        const c64 fv = fn(x);
        k_sum += kKronrodW[i] * fv;
        if ((i & 1u) == 1u) {
            g_sum += kGaussW[i / 2] * fv;
        }
    }
    return {half * k_sum, half * g_sum};
}

[[nodiscard]] c64 adaptive_1d_c64(Fn1C fn, f64 a, f64 b, f64 abs_tol, i32 depth, i32 max_depth) {
    const Gk15PairC64 p = gk15_c64(fn, a, b);
    const f64 err = abs(p.k - p.g);
    if (err < abs_tol || depth >= max_depth) {
        return p.k;
    }
    const f64 mid = 0.5 * (a + b);
    return adaptive_1d_c64(fn, a, mid, abs_tol * 0.5, depth + 1, max_depth) +
           adaptive_1d_c64(fn, mid, b, abs_tol * 0.5, depth + 1, max_depth);
}

// ============================================================================
// 2D Adaptive Tensor-Product 15x15 Gauss-Kronrod
// ============================================================================

[[nodiscard]] f64 adaptive_2d_f64(Fn2 fn, f64 x0, f64 x1, f64 y0, f64 y1, f64 tol, i32 depth, i32 max_depth) {
    const f64 half_x = 0.5 * (x1 - x0);
    const f64 mid_x = 0.5 * (x1 + x0);
    const f64 half_y = 0.5 * (y1 - y0);
    const f64 mid_y = 0.5 * (y1 + y0);

    f64 z[15][15];
    for (usize i = 0; i < 15; ++i) {
        const f64 xv = mid_x + half_x * kKronrodX[i];
        for (usize j = 0; j < 15; ++j) {
            const f64 yv = mid_y + half_y * kKronrodX[j];
            z[i][j] = fn(xv, yv);
        }
    }

    f64 kxy = 0.0;
    f64 gxky = 0.0;
    f64 kxgy = 0.0;
    f64 gxy = 0.0;
    for (usize i = 0; i < 15; ++i) {
        f64 ky = 0.0;
        f64 gy = 0.0;
        for (usize j = 0; j < 15; ++j) {
            ky += kKronrodW[j] * z[i][j];
            if ((j & 1u) == 1u) {
                gy += kGaussW[j / 2] * z[i][j];
            }
        }
        kxy += kKronrodW[i] * ky;
        kxgy += kKronrodW[i] * gy;
        if ((i & 1u) == 1u) {
            gxky += kGaussW[i / 2] * ky;
            gxy += kGaussW[i / 2] * gy;
        }
    }

    const f64 area_scale = half_x * half_y;
    kxy *= area_scale;
    gxky *= area_scale;
    kxgy *= area_scale;
    gxy *= area_scale;

    const f64 err_x = abs(kxy - gxky);
    const f64 err_y = abs(kxy - kxgy);
    const f64 err = max(abs(kxy - gxy), max(err_x, err_y));
    if (err < tol || depth >= max_depth) {
        return kxy;
    }

    if (err_x > err_y * 1.25 || (err_y <= err_x * 1.25 && (x1 - x0) >= (y1 - y0))) {
        return adaptive_2d_f64(fn, x0, mid_x, y0, y1, tol * 0.5, depth + 1, max_depth) +
               adaptive_2d_f64(fn, mid_x, x1, y0, y1, tol * 0.5, depth + 1, max_depth);
    }
    return adaptive_2d_f64(fn, x0, x1, y0, mid_y, tol * 0.5, depth + 1, max_depth) +
           adaptive_2d_f64(fn, x0, x1, mid_y, y1, tol * 0.5, depth + 1, max_depth);
}

[[nodiscard]] c64 adaptive_2d_c64(BatchFn2C batch_fn, Fn2C scalar_fn, f64 x0, f64 x1, f64 y0, f64 y1, f64 tol,
                                  i32 depth, i32 max_depth, bool& batch_ok) {
    const f64 half_x = 0.5 * (x1 - x0);
    const f64 mid_x = 0.5 * (x1 + x0);
    const f64 half_y = 0.5 * (y1 - y0);
    const f64 mid_y = 0.5 * (y1 + y0);

    c64 z[15][15];
    bool evaluated = false;

    if (batch_ok && !batch_fn.empty()) {
        f64 xs[225];
        f64 ys[225];
        c64 out[225];
        for (usize j = 0; j < 15; ++j) {
            const f64 yv = mid_y + half_y * kKronrodX[j];
            for (usize i = 0; i < 15; ++i) {
                xs[j * 15 + i] = mid_x + half_x * kKronrodX[i];
                ys[j * 15 + i] = yv;
            }
        }
        if (batch_fn(ArrayRef<f64>(xs, 225), ArrayRef<f64>(ys, 225), MutArrayRef<c64>(out, 225))) {
            for (usize j = 0; j < 15; ++j) {
                for (usize i = 0; i < 15; ++i) {
                    z[i][j] = out[j * 15 + i];
                }
            }
            evaluated = true;
        } else {
            batch_ok = false;
        }
    }

    if (!evaluated) {
        for (usize i = 0; i < 15; ++i) {
            const f64 xv = mid_x + half_x * kKronrodX[i];
            for (usize j = 0; j < 15; ++j) {
                const f64 yv = mid_y + half_y * kKronrodX[j];
                z[i][j] = scalar_fn(xv, yv);
            }
        }
    }

    c64 kxy(0.0, 0.0);
    c64 gxky(0.0, 0.0);
    c64 kxgy(0.0, 0.0);
    c64 gxy(0.0, 0.0);
    for (usize i = 0; i < 15; ++i) {
        c64 ky(0.0, 0.0);
        c64 gy(0.0, 0.0);
        for (usize j = 0; j < 15; ++j) {
            ky += kKronrodW[j] * z[i][j];
            if ((j & 1u) == 1u) {
                gy += kGaussW[j / 2] * z[i][j];
            }
        }
        kxy += kKronrodW[i] * ky;
        kxgy += kKronrodW[i] * gy;
        if ((i & 1u) == 1u) {
            gxky += kGaussW[i / 2] * ky;
            gxy += kGaussW[i / 2] * gy;
        }
    }

    const f64 area_scale = half_x * half_y;
    kxy *= area_scale;
    gxky *= area_scale;
    kxgy *= area_scale;
    gxy *= area_scale;

    const f64 err_x = abs(kxy - gxky);
    const f64 err_y = abs(kxy - kxgy);
    const f64 err = max(abs(kxy - gxy), max(err_x, err_y));
    if (err < tol || depth >= max_depth) {
        return kxy;
    }

    if (err_x > err_y * 1.25 || (err_y <= err_x * 1.25 && (x1 - x0) >= (y1 - y0))) {
        return adaptive_2d_c64(batch_fn, scalar_fn, x0, mid_x, y0, y1, tol * 0.5, depth + 1, max_depth, batch_ok) +
               adaptive_2d_c64(batch_fn, scalar_fn, mid_x, x1, y0, y1, tol * 0.5, depth + 1, max_depth, batch_ok);
    }
    return adaptive_2d_c64(batch_fn, scalar_fn, x0, x1, y0, mid_y, tol * 0.5, depth + 1, max_depth, batch_ok) +
           adaptive_2d_c64(batch_fn, scalar_fn, x0, x1, mid_y, y1, tol * 0.5, depth + 1, max_depth, batch_ok);
}

} // namespace

// ============================================================================
// Public Adaptive Quadrature API
// ============================================================================

Result<f64> integral(Fn1 fn, f64 a, f64 b, f64 abs_tol, i32 max_depth) {
    if (fn.empty()) return {Status::InvalidCallback, 0.0};
    if (!is_finite(a) || !is_finite(b)) return {Status::InvalidBounds, 0.0};
    if (!(abs_tol > 0.0) || !is_finite(abs_tol)) return {Status::InvalidTolerance, 0.0};
    f64 sign = 1.0;
    if (b < a) {
        swap(a, b);
        sign = -1.0;
    }
    if (a == b) return {Status::Ok, 0.0};
    const i32 depth_limit = (max_depth >= 0) ? max_depth : 20;
    return {Status::Ok, sign * adaptive_1d_f64(fn, a, b, abs_tol, 0, depth_limit)};
}

Result<c64> integral_c64(Fn1C fn, f64 a, f64 b, f64 abs_tol, i32 max_depth) {
    if (fn.empty()) return {Status::InvalidCallback, c64(0.0, 0.0)};
    if (!is_finite(a) || !is_finite(b)) return {Status::InvalidBounds, c64(0.0, 0.0)};
    if (!(abs_tol > 0.0) || !is_finite(abs_tol)) return {Status::InvalidTolerance, c64(0.0, 0.0)};
    f64 sign = 1.0;
    if (b < a) {
        swap(a, b);
        sign = -1.0;
    }
    if (a == b) return {Status::Ok, c64(0.0, 0.0)};
    const i32 depth_limit = (max_depth >= 0) ? max_depth : 20;
    return {Status::Ok, sign * adaptive_1d_c64(fn, a, b, abs_tol, 0, depth_limit)};
}

Result<f64> integral2(Fn2 fn, f64 a, f64 b, f64 c, f64 d, f64 abs_tol, i32 max_depth) {
    if (fn.empty()) return {Status::InvalidCallback, 0.0};
    if (!is_finite(a) || !is_finite(b) || !is_finite(c) || !is_finite(d)) return {Status::InvalidBounds, 0.0};
    if (!(abs_tol > 0.0) || !is_finite(abs_tol)) return {Status::InvalidTolerance, 0.0};
    f64 sign = 1.0;
    if (b < a) {
        swap(a, b);
        sign = -sign;
    }
    if (d < c) {
        swap(c, d);
        sign = -sign;
    }
    if (a == b || c == d) return {Status::Ok, 0.0};
    const i32 depth_limit = (max_depth >= 0) ? max_depth : 18;
    return {Status::Ok, sign * adaptive_2d_f64(fn, a, b, c, d, abs_tol, 0, depth_limit)};
}

Result<c64> integral2_c64(Fn2C fn, f64 a, f64 b, f64 c, f64 d, f64 abs_tol, i32 max_depth) {
    return integral2_batch_c64(nullptr, fn, a, b, c, d, abs_tol, max_depth);
}

Result<c64> integral2_batch_c64(BatchFn2C batch_fn, Fn2C scalar_fn, f64 a, f64 b, f64 c, f64 d, f64 abs_tol,
                                i32 max_depth) {
    if (scalar_fn.empty() && batch_fn.empty()) return {Status::InvalidCallback, c64(0.0, 0.0)};
    if (!is_finite(a) || !is_finite(b) || !is_finite(c) || !is_finite(d)) return {Status::InvalidBounds, c64(0.0, 0.0)};
    if (!(abs_tol > 0.0) || !is_finite(abs_tol)) return {Status::InvalidTolerance, c64(0.0, 0.0)};
    f64 sign = 1.0;
    if (b < a) {
        swap(a, b);
        sign = -sign;
    }
    if (d < c) {
        swap(c, d);
        sign = -sign;
    }
    if (a == b || c == d) return {Status::Ok, c64(0.0, 0.0)};
    const i32 depth_limit = (max_depth >= 0) ? max_depth : 18;
    bool batch_ok = !batch_fn.empty();
    return {Status::Ok, sign * adaptive_2d_c64(batch_fn, scalar_fn, a, b, c, d, abs_tol, 0, depth_limit, batch_ok)};
}

Result<f64> integral3(Fn3 fn, f64 a, f64 b, f64 c, f64 d, f64 e, f64 f, f64 abs_tol, i32 max_depth_outer,
                      i32 max_depth_inner) {
    if (fn.empty()) return {Status::InvalidCallback, 0.0};
    if (!is_finite(a) || !is_finite(b) || !is_finite(c) || !is_finite(d) || !is_finite(e) || !is_finite(f)) {
        return {Status::InvalidBounds, 0.0};
    }
    if (!(abs_tol > 0.0) || !is_finite(abs_tol)) return {Status::InvalidTolerance, 0.0};

    auto outer_fn = [&](f64 x) -> f64 {
        auto inner_fn = [&](f64 y, f64 z) -> f64 { return fn(x, y, z); };
        return integral2(inner_fn, c, d, e, f, abs_tol, max_depth_inner).value;
    };
    return integral(outer_fn, a, b, abs_tol, max_depth_outer);
}

Result<c64> integral3_c64(Fn3C fn, f64 a, f64 b, f64 c, f64 d, f64 e, f64 f, f64 abs_tol, i32 max_depth_outer,
                          i32 max_depth_inner) {
    if (fn.empty()) return {Status::InvalidCallback, c64(0.0, 0.0)};
    if (!is_finite(a) || !is_finite(b) || !is_finite(c) || !is_finite(d) || !is_finite(e) || !is_finite(f)) {
        return {Status::InvalidBounds, c64(0.0, 0.0)};
    }
    if (!(abs_tol > 0.0) || !is_finite(abs_tol)) return {Status::InvalidTolerance, c64(0.0, 0.0)};

    auto outer_fn = [&](f64 x) -> c64 {
        auto inner_fn = [&](f64 y, f64 z) -> c64 { return fn(x, y, z); };
        return integral2_c64(inner_fn, c, d, e, f, abs_tol, max_depth_inner).value;
    };
    return integral_c64(outer_fn, a, b, abs_tol, max_depth_outer);
}

// ============================================================================
// Discrete Trapezoidal Integration
// ============================================================================

f64 trapz_strided(const f64* y, usize n, usize y_stride, const f64* x, usize x_stride) noexcept {
    if (n < 2 || y == nullptr) return 0.0;
    f64 sum = 0.0;
    for (usize i = 1; i < n; ++i) {
        const f64 dx = (x != nullptr) ? (x[i * x_stride] - x[(i - 1) * x_stride]) : 1.0;
        sum += 0.5 * (y[(i - 1) * y_stride] + y[i * y_stride]) * dx;
    }
    return sum;
}

c64 trapz_strided_c64(const c64* y, usize n, usize y_stride, const f64* x, usize x_stride) noexcept {
    if (n < 2 || y == nullptr) return c64(0.0, 0.0);
    c64 sum(0.0, 0.0);
    for (usize i = 1; i < n; ++i) {
        const f64 dx = (x != nullptr) ? (x[i * x_stride] - x[(i - 1) * x_stride]) : 1.0;
        sum += 0.5 * (y[(i - 1) * y_stride] + y[i * y_stride]) * dx;
    }
    return sum;
}

f64 trapz(ArrayRef<f64> y, ArrayRef<f64> x) noexcept {
    const f64* xp = (!x.empty() && x.size() >= y.size()) ? x.data() : nullptr;
    return trapz_strided(y.data(), y.size(), 1, xp, 1);
}

c64 trapz_c64(ArrayRef<c64> y, ArrayRef<f64> x) noexcept {
    const f64* xp = (!x.empty() && x.size() >= y.size()) ? x.data() : nullptr;
    return trapz_strided_c64(y.data(), y.size(), 1, xp, 1);
}

void cumtrapz_strided(const f64* y, f64* dst, usize n, usize y_stride, usize dst_stride, const f64* x,
                      usize x_stride) noexcept {
    if (n == 0 || y == nullptr || dst == nullptr) return;
    dst[0] = 0.0;
    for (usize i = 1; i < n; ++i) {
        const f64 dx = (x != nullptr) ? (x[i * x_stride] - x[(i - 1) * x_stride]) : 1.0;
        dst[i * dst_stride] = dst[(i - 1) * dst_stride] + 0.5 * (y[(i - 1) * y_stride] + y[i * y_stride]) * dx;
    }
}

void cumtrapz_strided_c64(const c64* y, c64* dst, usize n, usize y_stride, usize dst_stride, const f64* x,
                          usize x_stride) noexcept {
    if (n == 0 || y == nullptr || dst == nullptr) return;
    dst[0] = c64(0.0, 0.0);
    for (usize i = 1; i < n; ++i) {
        const f64 dx = (x != nullptr) ? (x[i * x_stride] - x[(i - 1) * x_stride]) : 1.0;
        dst[i * dst_stride] = dst[(i - 1) * dst_stride] + 0.5 * (y[(i - 1) * y_stride] + y[i * y_stride]) * dx;
    }
}

void cumtrapz(ArrayRef<f64> y, MutArrayRef<f64> dst, ArrayRef<f64> x) noexcept {
    const usize n = min(y.size(), dst.size());
    const f64* xp = (!x.empty() && x.size() >= n) ? x.data() : nullptr;
    cumtrapz_strided(y.data(), dst.data(), n, 1, 1, xp, 1);
}

void cumtrapz_c64(ArrayRef<c64> y, MutArrayRef<c64> dst, ArrayRef<f64> x) noexcept {
    const usize n = min(y.size(), dst.size());
    const f64* xp = (!x.empty() && x.size() >= n) ? x.data() : nullptr;
    cumtrapz_strided_c64(y.data(), dst.data(), n, 1, 1, xp, 1);
}

// ============================================================================
// Finite-Difference Calculus
// ============================================================================

void gradient_strided(const f64* src, f64* dst, usize n, usize stride, f64 h) noexcept {
    if (n == 0 || src == nullptr || dst == nullptr) return;
    if (n == 1) {
        dst[0] = 0.0;
        return;
    }
    const f64 inv2h = 0.5 / h;
    const f64 inv_h = 1.0 / h;
    dst[0] = (src[stride] - src[0]) * inv_h;
    dst[(n - 1) * stride] = (src[(n - 1) * stride] - src[(n - 2) * stride]) * inv_h;
    for (usize i = 1; i + 1 < n; ++i) {
        dst[i * stride] = (src[(i + 1) * stride] - src[(i - 1) * stride]) * inv2h;
    }
}

void gradient_1d(ArrayRef<f64> src, MutArrayRef<f64> dst, f64 h) noexcept {
    const usize n = min(src.size(), dst.size());
    gradient_strided(src.data(), dst.data(), n, 1, h);
}

void del2_strided(const f64* src, f64* dst, usize n, usize stride, f64 h) noexcept {
    if (n == 0 || src == nullptr || dst == nullptr) return;
    for (usize i = 0; i < n; ++i) {
        dst[i * stride] = 0.0;
    }
    if (n < 3) return;
    const f64 inv = 1.0 / (2.0 * h * h);
    for (usize i = 1; i + 1 < n; ++i) {
        dst[i * stride] = (src[(i + 1) * stride] - 2.0 * src[i * stride] + src[(i - 1) * stride]) * inv;
    }
    if (n == 3) {
        dst[0] = dst[1 * stride];
        dst[2 * stride] = dst[1 * stride];
    } else {
        dst[0] = 2.0 * dst[1 * stride] - dst[2 * stride];
        dst[(n - 1) * stride] = 2.0 * dst[(n - 2) * stride] - dst[(n - 3) * stride];
    }
}

} // namespace keel::quad
