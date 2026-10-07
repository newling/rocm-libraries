# ROCm Host Numerics

ROCm Host Numerics is a shared CPU-only component for generating test inputs,
computing numerical reference results, and comparing them with observed results.
It is being introduced incrementally, starting with build, packaging, and test
infrastructure. The numerical implementation and consumer integrations are
demonstrated in the [prototype PR #10553](https://github.com/ROCm/rocm-libraries/pull/10553).

## Intended end state

hipBLASLt, TensileLite, and rocRoller's GEMM paths currently contain overlapping
input generation, datatype conversion, reference computation, and comparison
code. The goal is for all three to use host-numerics for this shared CPU work,
with the same APIs available to other rocm-libraries components as they adopt
it. New datatype support and numerical fixes can then be implemented and tested
in one place.

The shared API will use a NumPy-like tensor model that keeps datatype, shape,
layout, and storage together. It will cover ordinary and packed low-precision
types, deterministic generation, reference operations such as GEMM and its
epilogues, and numerical comparison. Consumers will use the C++ API; Python
bindings will expose the same numerical behavior for independent tests using
NumPy, `ml_dtypes`, encoded values, and Python integer arithmetic. The component
will build and test without HIP, a GPU toolchain, or GPU hardware.

host-numerics will subsume `mxDataGenerator`. Its CPU generation and datatype
support will become part of the shared numerical APIs. CPU transforms that
construct physical AMD GPU layouts, such as scale swizzling and pre-tiling,
will live behind a separate API within host-numerics, keeping architecture
details out of generic tensors and numerical operations. Once all callers and
build dependencies have migrated, the old `mxDataGenerator` component will be
removed.

Each consuming product will translate its descriptors and buffers through a
small private adapter. Products will continue to own GPU allocation, execution,
transfers, tolerance selection, and reporting. The rocRoller migration covers
its GEMM validation paths; its general kernel-generation functionality stays
with rocRoller.

## Incremental landing

The prototype in #10553 serves as an end-to-end integration reference while
the implementation lands in `develop` through small, independently reviewable
PRs. The sequence is:

1. **Build and package foundations.** Establish the CMake targets and installed
   package, Python module, C++ and Python smoke tests, installed-package tests,
   dedicated CPU CI, and ownership. This is the scope of the
   [first PR, #12208](https://github.com/ROCm/rocm-libraries/pull/12208).
2. **Tensor and datatype core.** Introduce the tensor model, storage and
   ownership rules, scalar formats, and conversions, with C++ tests and
   independent Python coverage.
3. **Numerical operations.** Add deterministic input generation, reference
   arithmetic, and comparison in focused changes. Each change brings tests for
   its numerical behavior, including independent expected results, and the
   standalone suite gains sanitizer coverage.
4. **Consumer migrations and removal.** Move hipBLASLt, TensileLite, and
   rocRoller's GEMM paths onto the shared APIs incrementally. Remove duplicate
   implementations as their callers migrate, and retire `mxDataGenerator`
   once its remaining responsibilities and build dependencies have moved.

The first three stages develop and test host-numerics independently of the
consumers. They keep it outside the default monorepo build and provide fast
feedback through the standalone CPU CI job. The target is for the component
build and tests to finish in under two minutes, excluding runner queue and
setup time.

Consumer migrations must preserve existing product tests, supported numerical
behavior, and tolerances. Independent numerical tests should establish the
expected results before the shared implementation becomes a product's sole
reference. Compare representative input-generation and reference-computation
timings, as well as their effect on complete test and benchmark runs, before
removing an old accelerated path.

The initial package implements stage 1: the C++ version header and Python
`__version__` attribute exercise the build and installation paths. Tensor types,
numerical operations, and adoption by consumers follow in later PRs.

## Build and test

The standalone build requires CMake 3.25.2 or newer, a C++20 compiler, and Ninja.
The Python module also needs Python 3.9 or newer with development headers and
nanobind. From the repository root, the following creates a virtual environment
and build directory outside the source tree:

```shell
host_numerics_venv="$PWD/../venvs/host-numerics"
host_numerics_build="$PWD/../builds/host-numerics"
python3 -m venv "$host_numerics_venv"
"$host_numerics_venv/bin/python" -m pip install nanobind==3.0.1

cmake -S shared/host-numerics -B "$host_numerics_build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$host_numerics_venv/bin/python" \
  -DHOST_NUMERICS_BUILD_TESTING=ON \
  -DHOST_NUMERICS_BUILD_PYTHON=ON
cmake --build "$host_numerics_build" --parallel 4
ctest --test-dir "$host_numerics_build" --output-on-failure
```

For a C++-only build, set `HOST_NUMERICS_BUILD_PYTHON=OFF`; Python and nanobind
are then unnecessary.

Install into a local prefix with:

```shell
cmake --install "$host_numerics_build" --prefix "$host_numerics_build/install"
```

The installed `ROCHostNumerics` CMake package exports
`roc::host-numerics-core` and `roc::host-numerics`. Consumers can add the install
prefix to `CMAKE_PREFIX_PATH` and use:

```cmake
find_package(ROCHostNumerics CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE roc::host-numerics)
```

The Python package is named `roc_host_numerics`.
