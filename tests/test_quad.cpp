#include <keel/format.h>
#include <keel/math.h>
#include <keel/quad.h>
#include <keel/string.h>

namespace {

keel::i32 g_failures = 0;

void check(bool cond, const char* msg) {
    if (!cond) {
        keel::eprintln("FAIL: %s", msg);
        ++g_failures;
    }
}

void check_near(keel::f64 got, keel::f64 expected, keel::f64 tol, const char* msg) {
    if (keel::abs(got - expected) > tol || keel::is_nan(got)) {
        keel::eprintln("FAIL: %s (got %.15g, expected %.15g, tol %.3g)", msg, got, expected, tol);
        ++g_failures;
    }
}

void test_integral_1d() {
    // Full-precision weight sum check: integral_{-1}^1 1 dx == 2 to ~1 ULP
    auto r0 = keel::quad::integral([](keel::f64) { return 1.0; }, -1.0, 1.0);
    check(r0.ok(), "integral const ok");
    check_near(r0.value, 2.0, 1e-15, "integral const 17-digit weights");

    auto r1 = keel::quad::integral([](keel::f64 x) { return keel::sin(x); }, 0.0, keel::kPi);
    check(r1.ok(), "integral sin ok");
    check_near(r1.value, 2.0, 1e-14, "integral sin(0..pi) == 2");

    // Reversed bounds flip sign
    auto r2 = keel::quad::integral([](keel::f64 x) { return x * x; }, 2.0, 0.0);
    check(r2.ok(), "integral reversed ok");
    check_near(r2.value, -8.0 / 3.0, 1e-14, "integral x^2 reversed");

    // Complex integrand: integral_0^pi exp(i*x) dx = 2i
    auto rc = keel::quad::integral_c64([](keel::f64 x) { return keel::c64(keel::cos(x), keel::sin(x)); }, 0.0, keel::kPi);
    check(rc.ok(), "integral_c64 ok");
    check_near(rc.value.real(), 0.0, 1e-14, "integral_c64 real");
    check_near(rc.value.imag(), 2.0, 1e-14, "integral_c64 imag");

    // NaN integrand terminates immediately instead of doing 2^20 splits
    keel::i32 nan_evals = 0;
    auto rnan = keel::quad::integral(
        [&](keel::f64) {
            ++nan_evals;
            return keel::kNaN;
        },
        0.0, 1.0);
    check(rnan.ok() && keel::is_nan(rnan.value) && nan_evals == 15, "NaN integrand stops immediately");

    // Error statuses
    check(keel::quad::integral(nullptr, 0.0, 1.0).status == keel::quad::Status::InvalidCallback, "null callback");
    check(keel::quad::integral([](keel::f64 x) { return x; }, 0.0, keel::kInf).status ==
              keel::quad::Status::InvalidBounds,
          "inf bound");
    check(keel::quad::integral([](keel::f64 x) { return x; }, 0.0, 1.0, -1e-6).status ==
              keel::quad::Status::InvalidTolerance,
          "negative tol");
}

void test_integral_2d_and_3d() {
    auto r2 = keel::quad::integral2([](keel::f64 x, keel::f64 y) { return x * y * y; }, 0.0, 1.0, 0.0, 2.0);
    check(r2.ok(), "integral2 ok");
    check_near(r2.value, 4.0 / 3.0, 1e-14, "integral2 x*y^2");

    // Batch 2D complex path + fallback path
    keel::i32 batch_calls = 0;
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
    check_near(r2b.value.real(), 3.0, 1e-13, "integral2_batch_c64 real");
    check_near(r2b.value.imag(), 1.0, 1e-13, "integral2_batch_c64 imag");

    // Batch returns false with empty scalar_fn must return InvalidCallback (not crash!)
    auto fail_batch = [](keel::ArrayRef<keel::f64>, keel::ArrayRef<keel::f64>, keel::MutArrayRef<keel::c64>) -> bool {
        return false;
    };
    auto r2fail = keel::quad::integral2_batch_c64(fail_batch, nullptr, 0.0, 1.0, 0.0, 2.0);
    check(r2fail.status == keel::quad::Status::InvalidCallback, "batch false + null scalar -> InvalidCallback");

    // 3D integral
    auto r3 = keel::quad::integral3([](keel::f64 x, keel::f64 y, keel::f64 z) { return x * y * z; }, 0.0, 1.0, 0.0, 2.0,
                                    0.0, 3.0, 1e-10);
    check(r3.ok(), "integral3 ok");
    check_near(r3.value, 4.5, 1e-11, "integral3 x*y*z");

    // 3D zero-width domain exits early
    auto r3z = keel::quad::integral3([](keel::f64 x, keel::f64 y, keel::f64 z) { return x + y + z; }, 0.0, 1.0, 2.0, 2.0,
                                     0.0, 3.0);
    check(r3z.ok() && r3z.value == 0.0, "integral3 zero-width");
}

void test_trapz_and_cumtrapz() {
    const keel::f64 x[4] = {0.0, 1.0, 3.0, 6.0};
    const keel::f64 y[4] = {1.0, 2.0, 4.0, 7.0};
    check_near(keel::quad::trapz(y, x), 24.0, 1e-14, "trapz nonuniform");
    check_near(keel::quad::trapz(y), 10.0, 1e-14, "trapz unit");

    keel::f64 out[4];
    keel::quad::cumtrapz(y, out, x);
    check_near(out[0], 0.0, 1e-14, "cumtrapz[0]");
    check_near(out[1], 1.5, 1e-14, "cumtrapz[1]");
    check_near(out[2], 7.5, 1e-14, "cumtrapz[2]");
    check_near(out[3], 24.0, 1e-14, "cumtrapz[3]");

    // In-place cumtrapz
    keel::f64 ip[4] = {1.0, 2.0, 4.0, 7.0};
    keel::quad::cumtrapz(ip, ip, x);
    for (keel::usize i = 0; i < 4; ++i) {
        check_near(ip[i], out[i], 1e-14, "cumtrapz in-place");
    }

    const keel::c64 yc[3] = {keel::c64(1.0, 2.0), keel::c64(3.0, 4.0), keel::c64(5.0, 6.0)};
    const keel::c64 tc = keel::quad::trapz_c64(yc);
    check_near(tc.real(), 6.0, 1e-14, "trapz_c64 real");
    check_near(tc.imag(), 8.0, 1e-14, "trapz_c64 imag");

    keel::c64 outc[3] = {keel::c64(1.0, 2.0), keel::c64(3.0, 4.0), keel::c64(5.0, 6.0)};
    keel::quad::cumtrapz_c64(outc, outc);
    check_near(outc[2].real(), 6.0, 1e-14, "cumtrapz_c64 in-place real");
    check_near(outc[2].imag(), 8.0, 1e-14, "cumtrapz_c64 in-place imag");
}

void test_gradient_and_del2() {
    const keel::f64 y[5] = {0.0, 0.25, 1.0, 2.25, 4.0};
    keel::f64 g[5];
    keel::quad::gradient_1d(y, g, 0.5);
    check_near(g[0], 0.5, 1e-14, "gradient forward boundary");
    check_near(g[1], 1.0, 1e-14, "gradient interior 1");
    check_near(g[2], 2.0, 1e-14, "gradient interior 2");
    check_near(g[3], 3.0, 1e-14, "gradient interior 3");
    check_near(g[4], 3.5, 1e-14, "gradient backward boundary");

    // In-place gradient
    keel::f64 gip[5] = {0.0, 0.25, 1.0, 2.25, 4.0};
    keel::quad::gradient_1d(gip, gip, 0.5);
    for (keel::usize i = 0; i < 5; ++i) {
        check_near(gip[i], g[i], 1e-14, "gradient in-place");
    }

    // del2 of x^2 + in-place del2
    keel::f64 d2[5];
    keel::quad::del2_strided(y, d2, 5, 1, 0.5);
    keel::f64 d2ip[5] = {0.0, 0.25, 1.0, 2.25, 4.0};
    keel::quad::del2_strided(d2ip, d2ip, 5, 1, 0.5);
    for (keel::usize i = 0; i < 5; ++i) {
        check_near(d2[i], 1.0, 1e-13, "del2_strided quadratic");
        check_near(d2ip[i], d2[i], 1e-14, "del2_strided in-place");
    }
}

void dump_f64_array(const char* name, keel::ArrayRef<keel::f64> v) {
    keel::print("%s:", name);
    for (keel::usize i = 0; i < v.size(); ++i) {
        keel::print(" %.17g", v[i]);
    }
    keel::println("");
}

void dump_reference() {
    // 1D smooth, oscillatory, peaked, weak algebraic singularity
    auto i1 = keel::quad::integral([](keel::f64 x) { return keel::exp(-x * x) * keel::cos(3.0 * x); }, -1.5, 2.5, 1e-13);
    auto i2 = keel::quad::integral([](keel::f64 x) { return 1.0 / (1.0 + 100.0 * (x - 0.3) * (x - 0.3)); }, 0.0, 1.0, 1e-12);
    auto i3 = keel::quad::integral([](keel::f64 x) { return keel::sqrt(x) * keel::log(x + 1.0); }, 0.0, 2.0, 1e-11);
    auto ic = keel::quad::integral_c64(
        [](keel::f64 x) { return keel::c64(keel::exp(-x) * keel::cos(5.0 * x), keel::exp(-x) * keel::sin(5.0 * x)); },
        0.0, 3.0, 1e-13);
    keel::println("int1_smooth: %.17g", i1.value);
    keel::println("int1_peaked: %.17g", i2.value);
    keel::println("int1_sqrtlog: %.17g", i3.value);
    keel::println("int1_c64: %.17g %.17g", ic.value.real(), ic.value.imag());

    // 2D real & complex
    auto i2d = keel::quad::integral2(
        [](keel::f64 x, keel::f64 y) { return keel::sin(x * y) * keel::exp(-0.5 * (x + y)); }, 0.0, 2.0, -1.0, 1.5,
        1e-12);
    auto i2dc = keel::quad::integral2_c64(
        [](keel::f64 x, keel::f64 y) { return keel::c64(keel::cos(x + y), keel::sin(x * y)); }, 0.0, 1.5, 0.0, 2.0,
        1e-12);
    keel::println("int2_real: %.17g", i2d.value);
    keel::println("int2_c64: %.17g %.17g", i2dc.value.real(), i2dc.value.imag());

    // 3D real & complex
    auto i3d = keel::quad::integral3(
        [](keel::f64 x, keel::f64 y, keel::f64 z) { return keel::exp(-(x + 0.5 * y + 0.25 * z)) * keel::cos(x * y * z); },
        0.0, 1.0, 0.0, 1.5, 0.0, 2.0, 1e-10);
    auto i3dc = keel::quad::integral3_c64(
        [](keel::f64 x, keel::f64 y, keel::f64 z) { return keel::c64(x * y + z, y * z - x); }, 0.0, 1.0, -1.0, 1.0, 0.5,
        2.5, 1e-10);
    keel::println("int3_real: %.17g", i3d.value);
    keel::println("int3_c64: %.17g %.17g", i3dc.value.real(), i3dc.value.imag());

    // trapz, cumtrapz, gradient, del2
    const keel::f64 tx[6] = {0.0, 0.3, 0.9, 1.4, 2.2, 3.0};
    const keel::f64 ty[6] = {1.0, 1.4, 2.1, 0.7, -0.5, 1.8};
    keel::println("trapz_nonunif: %.17g", keel::quad::trapz(ty, tx));
    keel::println("trapz_unif: %.17g", keel::quad::trapz(ty));
    keel::f64 ct[6];
    keel::quad::cumtrapz(ty, ct, tx);
    dump_f64_array("cumtrapz_nonunif", ct);

    const keel::f64 gy[7] = {0.0, 0.125, 1.0, 3.375, 8.0, 15.625, 27.0}; // (0.5*k)^3
    keel::f64 grad_out[7];
    keel::quad::gradient_1d(gy, grad_out, 0.5);
    dump_f64_array("gradient_1d", grad_out);

    keel::f64 del2_out[7];
    keel::quad::del2_strided(gy, del2_out, 7, 1, 0.5);
    dump_f64_array("del2_1d", del2_out);
}

} // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && keel::StringRef(argv[1]) == "--dump-ref") {
        dump_reference();
        return 0;
    }
    test_integral_1d();
    test_integral_2d_and_3d();
    test_trapz_and_cumtrapz();
    test_gradient_and_del2();
    if (g_failures == 0) {
        keel::println("ALL keel::quad TESTS PASSED");
        return 0;
    }
    keel::eprintln("%d FAILURE(S)", g_failures);
    return 1;
}
