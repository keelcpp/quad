#include <keel/quad.h>

#include <cmath>
#include <cstdio>

namespace {

int g_failures = 0;

void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++g_failures;
    }
}

void check_near(keel::f64 got, keel::f64 expected, keel::f64 tol, const char* msg) {
    if (std::fabs(got - expected) > tol || std::isnan(got)) {
        std::fprintf(stderr, "FAIL: %s (got %.15g, expected %.15g, tol %.3g)\n", msg, got, expected, tol);
        ++g_failures;
    }
}

void test_integral_1d() {
    auto r1 = keel::quad::integral([](keel::f64 x) { return std::sin(x); }, 0.0, keel::kPi);
    check(r1.ok(), "integral sin ok");
    check_near(r1.value, 2.0, 1e-12, "integral sin(0..pi) == 2");

    // Reversed bounds flip sign
    auto r2 = keel::quad::integral([](keel::f64 x) { return x * x; }, 2.0, 0.0);
    check(r2.ok(), "integral reversed ok");
    check_near(r2.value, -8.0 / 3.0, 1e-12, "integral x^2 reversed");

    // Complex integrand: integral_0^pi exp(i*x) dx = 2i
    auto rc = keel::quad::integral_c64([](keel::f64 x) { return keel::c64(std::cos(x), std::sin(x)); }, 0.0, keel::kPi);
    check(rc.ok(), "integral_c64 ok");
    check_near(rc.value.real(), 0.0, 1e-12, "integral_c64 real");
    check_near(rc.value.imag(), 2.0, 1e-12, "integral_c64 imag");

    // Error statuses
    check(keel::quad::integral(nullptr, 0.0, 1.0).status == keel::quad::Status::InvalidCallback, "null callback");
    check(keel::quad::integral([](keel::f64 x) { return x; }, 0.0, keel::Limits<keel::f64>::infinity()).status ==
              keel::quad::Status::InvalidBounds,
          "inf bound");
    check(keel::quad::integral([](keel::f64 x) { return x; }, 0.0, 1.0, -1e-6).status ==
              keel::quad::Status::InvalidTolerance,
          "negative tol");
}

void test_integral_2d_and_3d() {
    // integral_0^1 integral_0^2 (x*y^2) dy dx = (1/2) * (8/3) = 4/3
    auto r2 = keel::quad::integral2([](keel::f64 x, keel::f64 y) { return x * y * y; }, 0.0, 1.0, 0.0, 2.0);
    check(r2.ok(), "integral2 ok");
    check_near(r2.value, 4.0 / 3.0, 1e-12, "integral2 x*y^2");

    // Batch 2D complex path + fallback path
    int batch_calls = 0;
    auto batch_fn = [&](keel::ArrayRef<keel::f64> xs, keel::ArrayRef<keel::f64> ys,
                        keel::MutArrayRef<keel::c64> out) -> bool {
        ++batch_calls;
        for (keel::usize k = 0; k < xs.size(); ++k) {
            out[k] = keel::c64(xs[k] + ys[k], xs[k] * ys[k]);
        }
        return true;
    };
    auto scalar_fn = [](keel::f64 x, keel::f64 y) { return keel::c64(x + y, x * y); };
    auto r2b = keel::quad::integral2_batch_c64(batch_fn, scalar_fn, 0.0, 1.0, 0.0, 2.0);
    check(r2b.ok(), "integral2_batch_c64 ok");
    check(batch_calls > 0, "batch_fn called");
    // int_0^1 int_0^2 (x + y) dy dx = 1 + 2 = 3; int_0^1 int_0^2 (x*y) dy dx = (1/2)*2 = 1
    check_near(r2b.value.real(), 3.0, 1e-12, "integral2_batch_c64 real");
    check_near(r2b.value.imag(), 1.0, 1e-12, "integral2_batch_c64 imag");

    // 3D integral: int_0^1 int_0^2 int_0^3 (x * y * z) dz dy dx = (1/2)*(2)*(9/2) = 4.5
    auto r3 = keel::quad::integral3([](keel::f64 x, keel::f64 y, keel::f64 z) { return x * y * z; }, 0.0, 1.0, 0.0, 2.0,
                                    0.0, 3.0, 1e-8);
    check(r3.ok(), "integral3 ok");
    check_near(r3.value, 4.5, 1e-9, "integral3 x*y*z");
}

void test_trapz_and_cumtrapz() {
    const keel::f64 x[4] = {0.0, 1.0, 3.0, 6.0};
    const keel::f64 y[4] = {1.0, 2.0, 4.0, 7.0};
    // Exact linear y = x + 1 on [0, 6]: integral = 0.5*36 + 6 = 24
    check_near(keel::quad::trapz(y, x), 24.0, 1e-14, "trapz nonuniform");
    // Unit spacing: 0.5*(1+2) + 0.5*(2+4) + 0.5*(4+7) = 1.5 + 3 + 5.5 = 10
    check_near(keel::quad::trapz(y), 10.0, 1e-14, "trapz unit");

    keel::f64 out[4];
    keel::quad::cumtrapz(y, out, x);
    check_near(out[0], 0.0, 1e-14, "cumtrapz[0]");
    check_near(out[1], 1.5, 1e-14, "cumtrapz[1]");
    check_near(out[2], 7.5, 1e-14, "cumtrapz[2]");
    check_near(out[3], 24.0, 1e-14, "cumtrapz[3]");

    const keel::c64 yc[3] = {keel::c64(1.0, 2.0), keel::c64(3.0, 4.0), keel::c64(5.0, 6.0)};
    const keel::c64 tc = keel::quad::trapz_c64(yc);
    check_near(tc.real(), 6.0, 1e-14, "trapz_c64 real");
    check_near(tc.imag(), 8.0, 1e-14, "trapz_c64 imag");

    keel::c64 outc[3];
    keel::quad::cumtrapz_c64(yc, outc);
    check_near(outc[2].real(), 6.0, 1e-14, "cumtrapz_c64 real");
    check_near(outc[2].imag(), 8.0, 1e-14, "cumtrapz_c64 imag");
}

void test_gradient_and_del2() {
    // f(x) = x^2 at x = 0, 0.5, 1.0, 1.5, 2.0 (h = 0.5)
    const keel::f64 y[5] = {0.0, 0.25, 1.0, 2.25, 4.0};
    keel::f64 g[5];
    keel::quad::gradient_1d(y, g, 0.5);
    check_near(g[0], 0.5, 1e-14, "gradient forward boundary");
    check_near(g[1], 1.0, 1e-14, "gradient interior 1");
    check_near(g[2], 2.0, 1e-14, "gradient interior 2");
    check_near(g[3], 3.0, 1e-14, "gradient interior 3");
    check_near(g[4], 3.5, 1e-14, "gradient backward boundary");

    // del2 of x^2: second derivative is 2, divided by 2 -> 1.0 everywhere (exact for quadratics!)
    keel::f64 d2[5];
    keel::quad::del2_strided(y, d2, 5, 1, 0.5);
    for (keel::usize i = 0; i < 5; ++i) {
        check_near(d2[i], 1.0, 1e-13, "del2_strided quadratic");
    }
}

} // namespace

int main() {
    test_integral_1d();
    test_integral_2d_and_3d();
    test_trapz_and_cumtrapz();
    test_gradient_and_del2();
    if (g_failures == 0) {
        std::printf("ALL keel::quad TESTS PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "%d FAILURE(S)\n", g_failures);
    return 1;
}
