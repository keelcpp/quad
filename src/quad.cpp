#include "keel/quad.h"

namespace keel::quad {

namespace {

constexpr i32 kMaxDepth1D = 50;
constexpr i32 kMaxDepth2D = 32;
constexpr f64 kRoundoffFloor = 4.0 * 2.2204460492503131e-16;

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
    const f64 mid = a + half;
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
    const f64 eff_tol = max(abs_tol, kRoundoffFloor * abs(p.k));
    if (!is_finite(err) || !is_finite(p.k) || err <= eff_tol || depth >= max_depth) {
        return p.k;
    }
    const f64 mid = a + 0.5 * (b - a);
    if (mid <= a || mid >= b) {
        return p.k;
    }
    return adaptive_1d_f64(fn, a, mid, abs_tol * 0.5, depth + 1, max_depth) +
           adaptive_1d_f64(fn, mid, b, abs_tol * 0.5, depth + 1, max_depth);
}

[[nodiscard]] Gk15PairC64 gk15_c64(Fn1C fn, f64 a, f64 b) {
    const f64 half = 0.5 * (b - a);
    const f64 mid = a + half;
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
    const f64 eff_tol = max(abs_tol, kRoundoffFloor * abs(p.k));
    if (!is_finite(err) || !is_finite(p.k.real()) || !is_finite(p.k.imag()) || err <= eff_tol || depth >= max_depth) {
        return p.k;
    }
    const f64 mid = a + 0.5 * (b - a);
    if (mid <= a || mid >= b) {
        return p.k;
    }
    return adaptive_1d_c64(fn, a, mid, abs_tol * 0.5, depth + 1, max_depth) +
           adaptive_1d_c64(fn, mid, b, abs_tol * 0.5, depth + 1, max_depth);
}

// ============================================================================
// 2D Adaptive Tensor-Product 15x15 Gauss-Kronrod
// ============================================================================

[[nodiscard]] f64 adaptive_2d_f64(Fn2 fn, f64 x0, f64 x1, f64 y0, f64 y1, f64 tol, i32 depth, i32 max_depth) {
    const f64 half_x = 0.5 * (x1 - x0);
    const f64 mid_x = x0 + half_x;
    const f64 half_y = 0.5 * (y1 - y0);
    const f64 mid_y = y0 + half_y;

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
    const f64 eff_tol = max(tol, kRoundoffFloor * abs(kxy));
    if (!is_finite(err) || !is_finite(kxy) || err <= eff_tol || depth >= max_depth) {
        return kxy;
    }

    const bool can_split_x = (mid_x > x0 && mid_x < x1);
    const bool can_split_y = (mid_y > y0 && mid_y < y1);
    if (!can_split_x && !can_split_y) {
        return kxy;
    }

    const bool prefer_x = (err_x > err_y * 1.25 || (err_y <= err_x * 1.25 && (x1 - x0) >= (y1 - y0)));
    if ((prefer_x && can_split_x) || !can_split_y) {
        return adaptive_2d_f64(fn, x0, mid_x, y0, y1, tol * 0.5, depth + 1, max_depth) +
               adaptive_2d_f64(fn, mid_x, x1, y0, y1, tol * 0.5, depth + 1, max_depth);
    }
    return adaptive_2d_f64(fn, x0, x1, y0, mid_y, tol * 0.5, depth + 1, max_depth) +
           adaptive_2d_f64(fn, x0, x1, mid_y, y1, tol * 0.5, depth + 1, max_depth);
}

[[nodiscard]] c64 adaptive_2d_c64(BatchFn2C batch_fn, Fn2C scalar_fn, f64 x0, f64 x1, f64 y0, f64 y1, f64 tol,
                                  i32 depth, i32 max_depth, bool& batch_ok, bool& callback_failed) {
    const f64 half_x = 0.5 * (x1 - x0);
    const f64 mid_x = x0 + half_x;
    const f64 half_y = 0.5 * (y1 - y0);
    const f64 mid_y = y0 + half_y;

    // Indexed as z[j][i] (y-major) so batch_fn can write directly without a second 225-element buffer.
    c64 z[15][15];
    bool evaluated = false;

    if (batch_ok && !batch_fn.empty()) {
        f64 xs[225];
        f64 ys[225];
        for (usize j = 0; j < 15; ++j) {
            const f64 yv = mid_y + half_y * kKronrodX[j];
            for (usize i = 0; i < 15; ++i) {
                xs[j * 15 + i] = mid_x + half_x * kKronrodX[i];
                ys[j * 15 + i] = yv;
            }
        }
        if (batch_fn(ArrayRef<f64>(xs, 225), ArrayRef<f64>(ys, 225), MutArrayRef<c64>(&z[0][0], 225))) {
            evaluated = true;
        } else {
            batch_ok = false;
        }
    }

    if (!evaluated) {
        if (scalar_fn.empty()) {
            callback_failed = true;
            return c64(0.0, 0.0);
        }
        for (usize j = 0; j < 15; ++j) {
            const f64 yv = mid_y + half_y * kKronrodX[j];
            for (usize i = 0; i < 15; ++i) {
                const f64 xv = mid_x + half_x * kKronrodX[i];
                z[j][i] = scalar_fn(xv, yv);
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
            ky += kKronrodW[j] * z[j][i];
            if ((j & 1u) == 1u) {
                gy += kGaussW[j / 2] * z[j][i];
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
    const f64 eff_tol = max(tol, kRoundoffFloor * abs(kxy));
    if (!is_finite(err) || !is_finite(kxy.real()) || !is_finite(kxy.imag()) || err <= eff_tol || depth >= max_depth) {
        return kxy;
    }

    const bool can_split_x = (mid_x > x0 && mid_x < x1);
    const bool can_split_y = (mid_y > y0 && mid_y < y1);
    if (!can_split_x && !can_split_y) {
        return kxy;
    }

    const bool prefer_x = (err_x > err_y * 1.25 || (err_y <= err_x * 1.25 && (x1 - x0) >= (y1 - y0)));
    if ((prefer_x && can_split_x) || !can_split_y) {
        const c64 left = adaptive_2d_c64(batch_fn, scalar_fn, x0, mid_x, y0, y1, tol * 0.5, depth + 1, max_depth,
                                         batch_ok, callback_failed);
        if (callback_failed) return c64(0.0, 0.0);
        const c64 right = adaptive_2d_c64(batch_fn, scalar_fn, mid_x, x1, y0, y1, tol * 0.5, depth + 1, max_depth,
                                          batch_ok, callback_failed);
        return left + right;
    }
    const c64 bot = adaptive_2d_c64(batch_fn, scalar_fn, x0, x1, y0, mid_y, tol * 0.5, depth + 1, max_depth, batch_ok,
                                    callback_failed);
    if (callback_failed) return c64(0.0, 0.0);
    const c64 top = adaptive_2d_c64(batch_fn, scalar_fn, x0, x1, mid_y, y1, tol * 0.5, depth + 1, max_depth, batch_ok,
                                    callback_failed);
    return bot + top;
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
    const i32 depth_limit = (max_depth >= 0) ? min(max_depth, kMaxDepth1D) : 20;
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
    const i32 depth_limit = (max_depth >= 0) ? min(max_depth, kMaxDepth1D) : 20;
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
    const i32 depth_limit = (max_depth >= 0) ? min(max_depth, kMaxDepth2D) : 18;
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
    const i32 depth_limit = (max_depth >= 0) ? min(max_depth, kMaxDepth2D) : 18;
    bool batch_ok = !batch_fn.empty();
    bool callback_failed = false;
    const c64 val = adaptive_2d_c64(batch_fn, scalar_fn, a, b, c, d, abs_tol, 0, depth_limit, batch_ok, callback_failed);
    if (callback_failed) return {Status::InvalidCallback, c64(0.0, 0.0)};
    return {Status::Ok, sign * val};
}

Result<f64> integral3(Fn3 fn, f64 a, f64 b, f64 c, f64 d, f64 e, f64 f, f64 abs_tol, i32 max_depth_outer,
                      i32 max_depth_inner) {
    if (fn.empty()) return {Status::InvalidCallback, 0.0};
    if (!is_finite(a) || !is_finite(b) || !is_finite(c) || !is_finite(d) || !is_finite(e) || !is_finite(f)) {
        return {Status::InvalidBounds, 0.0};
    }
    if (!(abs_tol > 0.0) || !is_finite(abs_tol)) return {Status::InvalidTolerance, 0.0};
    if (a == b || c == d || e == f) return {Status::Ok, 0.0};

    const f64 span_x = max(1.0, abs(b - a));
    const f64 inner_tol = max(1e-15, (0.25 * abs_tol) / span_x);
    auto outer_fn = [&](f64 x) -> f64 {
        auto inner_fn = [&](f64 y, f64 z) -> f64 { return fn(x, y, z); };
        return integral2(inner_fn, c, d, e, f, inner_tol, max_depth_inner).value;
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
    if (a == b || c == d || e == f) return {Status::Ok, c64(0.0, 0.0)};

    const f64 span_x = max(1.0, abs(b - a));
    const f64 inner_tol = max(1e-15, (0.25 * abs_tol) / span_x);
    auto outer_fn = [&](f64 x) -> c64 {
        auto inner_fn = [&](f64 y, f64 z) -> c64 { return fn(x, y, z); };
        return integral2_c64(inner_fn, c, d, e, f, inner_tol, max_depth_inner).value;
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
    f64 prev_y = y[0];
    f64 acc = 0.0;
    dst[0] = 0.0;
    for (usize i = 1; i < n; ++i) {
        const f64 cur_y = y[i * y_stride];
        const f64 dx = (x != nullptr) ? (x[i * x_stride] - x[(i - 1) * x_stride]) : 1.0;
        acc += 0.5 * (prev_y + cur_y) * dx;
        dst[i * dst_stride] = acc;
        prev_y = cur_y;
    }
}

void cumtrapz_strided_c64(const c64* y, c64* dst, usize n, usize y_stride, usize dst_stride, const f64* x,
                          usize x_stride) noexcept {
    if (n == 0 || y == nullptr || dst == nullptr) return;
    c64 prev_y = y[0];
    c64 acc(0.0, 0.0);
    dst[0] = c64(0.0, 0.0);
    for (usize i = 1; i < n; ++i) {
        const c64 cur_y = y[i * y_stride];
        const f64 dx = (x != nullptr) ? (x[i * x_stride] - x[(i - 1) * x_stride]) : 1.0;
        acc += 0.5 * (prev_y + cur_y) * dx;
        dst[i * dst_stride] = acc;
        prev_y = cur_y;
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
    f64 prev = src[0];
    f64 curr = src[stride];
    dst[0] = (curr - prev) * inv_h;
    for (usize i = 1; i + 1 < n; ++i) {
        const f64 next = src[(i + 1) * stride];
        dst[i * stride] = (next - prev) * inv2h;
        prev = curr;
        curr = next;
    }
    dst[(n - 1) * stride] = (curr - prev) * inv_h;
}

void gradient_1d(ArrayRef<f64> src, MutArrayRef<f64> dst, f64 h) noexcept {
    const usize n = min(src.size(), dst.size());
    gradient_strided(src.data(), dst.data(), n, 1, h);
}

void del2_strided(const f64* src, f64* dst, usize n, usize stride, f64 h) noexcept {
    if (n == 0 || src == nullptr || dst == nullptr) return;
    if (n < 3) {
        for (usize i = 0; i < n; ++i) {
            dst[i * stride] = 0.0;
        }
        return;
    }
    const f64 inv = 1.0 / (2.0 * h * h);
    f64 prev = src[0];
    f64 curr = src[stride];
    for (usize i = 1; i + 1 < n; ++i) {
        const f64 next = src[(i + 1) * stride];
        dst[i * stride] = (next - 2.0 * curr + prev) * inv;
        prev = curr;
        curr = next;
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
