# keel::quad

Zero-STL C++20 numerical quadrature and discrete calculus library built on [`keel`](https://github.com/keelcpp/keel).

## Features

- **15-point Gauss–Kronrod (G7/K15) tables**: `kKronrodX`, `kKronrodW`, `kGaussW`
- **Adaptive quadrature** (real `f64` and complex `c64`):
  - `integral` / `integral_c64` (1D adaptive G7/K15)
  - `integral2` / `integral2_c64` / `integral2_batch_c64` (2D tensor-product 15×15 adaptive G7/K15 with optional vectorized batch evaluation)
  - `integral3` / `integral3_c64` (3D nested adaptive G7/K15)
- **Discrete trapezoidal integration** (real & complex, contiguous & strided):
  - `trapz` / `trapz_c64` / `trapz_strided` / `trapz_strided_c64`
  - `cumtrapz` / `cumtrapz_c64` / `cumtrapz_strided` / `cumtrapz_strided_c64`
- **Finite-difference calculus**:
  - `gradient_1d` / `gradient_strided` (central interior, one-sided boundaries)
  - `del2_strided` (discrete Laplacian 1D pass with linear boundary extrapolation)
- Non-owning `keel::FunctionRef` callbacks — works cleanly with `-fno-exceptions -fno-rtti`
- Zero STL headers in public headers and implementation

## Usage

```cpp
#include <keel/quad.h>

auto r = keel::quad::integral([](keel::f64 x) { return x * x; }, 0.0, 1.0);
// r.ok() == true, r.value ~= 1.0 / 3.0
```

## Build & Test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

`-fno-exceptions -fno-rtti` build:

```bash
cmake -S . -B build-noeh -DCMAKE_BUILD_TYPE=Release -DKEEL_QUAD_NO_EXCEPTIONS_RTTI=ON
cmake --build build-noeh -j
ctest --test-dir build-noeh --output-on-failure
```
