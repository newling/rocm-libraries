# Fused-A2A configuration tests

The shape and collective-name checks run in both `hipblaslt-test` and the
standalone `hipblaslt-a2a-config-test` executable. They do not require fusion
to be enabled. The standalone executable links only Google Test and can run
without HIP, GPU hardware, or generated device libraries.

To build just these checks, use a C++ compiler and an installed Google Test:

```sh
cmake -S projects/hipblaslt/clients/tests/a2a_config -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_PREFIX_PATH="$GTEST_INSTALL_DIR"
cmake --build "$BUILD_DIR" --target hipblaslt-a2a-config-test -j2
ctest --test-dir "$BUILD_DIR" --output-on-failure
```

This target checks host configuration only. The multi-process GPU sweep
remains in `fused_a2a_multiprocess_gtest.cpp` and requires a fusion-enabled
hipBLASLt build and suitable GPUs.
