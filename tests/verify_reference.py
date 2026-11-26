#!/usr/bin/env python3
"""Differential reference verification for keel::quad against NumPy, SciPy, and MATLAB."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import integrate


def parse_dump(text: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, rest = line.split(":", 1)
        vals = [float(tok) for tok in rest.strip().split()] if rest.strip() else []
        out[key.strip()] = vals
    return out


def check_close(name: str, got: np.ndarray, ref: np.ndarray, tol: float, failures: list[str]) -> None:
    got = np.asarray(got)
    ref = np.asarray(ref)
    if got.shape != ref.shape:
        failures.append(f"{name}: shape mismatch got={got.shape} ref={ref.shape}")
        return
    scale = max(1.0, float(np.max(np.abs(ref))) if ref.size else 1.0)
    err = float(np.max(np.abs(got - ref))) / scale if ref.size else 0.0
    status = "PASS" if err <= tol else "FAIL"
    print(f"  [{status}] {name:<28s} rel_err={err:.3e} (tol={tol:.1e})")
    if err > tol:
        failures.append(f"{name}: rel_err={err:.3e} > tol={tol:.1e}")


def run_matlab_checks(dump: dict[str, list[float]], failures: list[str]) -> None:
    if not shutil.which("matlab"):
        print("  [SKIP] MATLAB not found on PATH")
        return
    script = r"""
format longg
i1 = integral(@(x) exp(-x.^2).*cos(3*x), -1.5, 2.5, 'AbsTol', 1e-14, 'RelTol', 1e-14);
i2 = integral(@(x) 1./(1 + 100*(x - 0.3).^2), 0, 1, 'AbsTol', 1e-14, 'RelTol', 1e-14);
i2d = integral2(@(x, y) sin(x.*y).*exp(-0.5*(x + y)), 0, 2, -1, 1.5, 'AbsTol', 1e-13, 'RelTol', 1e-13);
i3d = integral3(@(x, y, z) exp(-(x + 0.5*y + 0.25*z)).*cos(x.*y.*z), 0, 1, 0, 1.5, 0, 2, 'AbsTol', 1e-12, 'RelTol', 1e-12);
gy = [0, 0.125, 1, 3.375, 8, 15.625, 27];
g1 = gradient(gy, 0.5);
d2 = del2(gy, 0.5);
fprintf('m_i1: %.17g\n', i1);
fprintf('m_i2: %.17g\n', i2);
fprintf('m_i2d: %.17g\n', i2d);
fprintf('m_i3d: %.17g\n', i3d);
fprintf('m_grad: %s\n', sprintf('%.17g ', g1));
fprintf('m_del2: %s\n', sprintf('%.17g ', d2));
"""
    proc = subprocess.run(["matlab", "-batch", script], capture_output=True, text=True, check=True)
    md = parse_dump(proc.stdout)
    check_close("matlab/int1_smooth", dump["int1_smooth"], md["m_i1"], 1e-13, failures)
    check_close("matlab/int1_peaked", dump["int1_peaked"], md["m_i2"], 1e-12, failures)
    check_close("matlab/int2_real", dump["int2_real"], md["m_i2d"], 1e-11, failures)
    check_close("matlab/int3_real", dump["int3_real"], md["m_i3d"], 1e-9, failures)
    check_close("matlab/gradient_1d", dump["gradient_1d"], md["m_grad"], 1e-14, failures)
    # MATLAB treats 1D vectors as 2D (1xN), dividing by 4*h^2; keel::quad::del2_strided
    # computes the 1D pass divided by 2*h^2 (exact factor of 2).
    check_close("matlab/del2_1d", 0.5 * np.asarray(dump["del2_1d"]), md["m_del2"], 1e-14, failures)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: verify_reference.py <path-to-test_quad>", file=sys.stderr)
        return 2
    exe = Path(sys.argv[1])
    proc = subprocess.run([str(exe), "--dump-ref"], capture_output=True, text=True, check=True)
    dump = parse_dump(proc.stdout)
    failures: list[str] = []

    print("Running keel::quad reference verification against NumPy/SciPy/MATLAB...")

    i1_ref, _ = integrate.quad(lambda x: np.exp(-x * x) * np.cos(3.0 * x), -1.5, 2.5, epsabs=1e-13, epsrel=1e-13)
    i2_ref, _ = integrate.quad(lambda x: 1.0 / (1.0 + 100.0 * (x - 0.3) ** 2), 0.0, 1.0, epsabs=1e-14, epsrel=1e-14)
    i3_ref, _ = integrate.quad(lambda x: np.sqrt(x) * np.log(x + 1.0), 0.0, 2.0, epsabs=1e-13, epsrel=1e-13)
    ic_re, _ = integrate.quad(lambda x: np.exp(-x) * np.cos(5.0 * x), 0.0, 3.0, epsabs=1e-14, epsrel=1e-14)
    ic_im, _ = integrate.quad(lambda x: np.exp(-x) * np.sin(5.0 * x), 0.0, 3.0, epsabs=1e-14, epsrel=1e-14)

    check_close("scipy/int1_smooth", dump["int1_smooth"], [i1_ref], 1e-13, failures)
    check_close("scipy/int1_peaked", dump["int1_peaked"], [i2_ref], 1e-12, failures)
    check_close("scipy/int1_sqrtlog", dump["int1_sqrtlog"], [i3_ref], 1e-10, failures)
    check_close("scipy/int1_c64", dump["int1_c64"], [ic_re, ic_im], 1e-13, failures)

    i2d_ref, _ = integrate.dblquad(
        lambda y, x: np.sin(x * y) * np.exp(-0.5 * (x + y)), 0.0, 2.0, -1.0, 1.5, epsabs=1e-13, epsrel=1e-13
    )
    i2dc_re, _ = integrate.dblquad(lambda y, x: np.cos(x + y), 0.0, 1.5, 0.0, 2.0, epsabs=1e-13, epsrel=1e-13)
    i2dc_im, _ = integrate.dblquad(lambda y, x: np.sin(x * y), 0.0, 1.5, 0.0, 2.0, epsabs=1e-13, epsrel=1e-13)
    check_close("scipy/int2_real", dump["int2_real"], [i2d_ref], 1e-12, failures)
    check_close("scipy/int2_c64", dump["int2_c64"], [i2dc_re, i2dc_im], 1e-12, failures)

    i3d_ref, _ = integrate.tplquad(
        lambda z, y, x: np.exp(-(x + 0.5 * y + 0.25 * z)) * np.cos(x * y * z),
        0.0, 1.0, 0.0, 1.5, 0.0, 2.0, epsabs=1e-11, epsrel=1e-11,
    )
    i3dc_re, _ = integrate.tplquad(lambda z, y, x: x * y + z, 0.0, 1.0, -1.0, 1.0, 0.5, 2.5, epsabs=1e-12, epsrel=1e-12)
    i3dc_im, _ = integrate.tplquad(lambda z, y, x: y * z - x, 0.0, 1.0, -1.0, 1.0, 0.5, 2.5, epsabs=1e-12, epsrel=1e-12)
    check_close("scipy/int3_real", dump["int3_real"], [i3d_ref], 1e-10, failures)
    check_close("scipy/int3_c64", dump["int3_c64"], [i3dc_re, i3dc_im], 1e-11, failures)

    tx = np.array([0.0, 0.3, 0.9, 1.4, 2.2, 3.0])
    ty = np.array([1.0, 1.4, 2.1, 0.7, -0.5, 1.8])
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    check_close("numpy/trapz_nonunif", dump["trapz_nonunif"], [trapz_fn(ty, tx)], 1e-14, failures)
    check_close("numpy/trapz_unif", dump["trapz_unif"], [trapz_fn(ty)], 1e-14, failures)
    check_close(
        "scipy/cumtrapz_nonunif",
        dump["cumtrapz_nonunif"],
        integrate.cumulative_trapezoid(ty, tx, initial=0.0),
        1e-14,
        failures,
    )

    gy = np.array([0.0, 0.125, 1.0, 3.375, 8.0, 15.625, 27.0])
    check_close("numpy/gradient_1d", dump["gradient_1d"], np.gradient(gy, 0.5), 1e-14, failures)

    run_matlab_checks(dump, failures)

    if failures:
        print(f"\nFAILED ({len(failures)} check(s)):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nALL keel::quad REFERENCE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
