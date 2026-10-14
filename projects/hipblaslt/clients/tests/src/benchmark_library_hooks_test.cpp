// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

namespace
{
    int libraryHookCalls = 0;
}

// Simulate a library hook rejecting the timer call, without requiring a GPU.
#define HIPBLASLT_BENCHMARK_CHECK_HIP(expr) \
    do                                      \
    {                                       \
        ++libraryHookCalls;                 \
        return;                             \
    } while(0)
#include <hipblaslt_benchmark/candidate_search.hpp>

int benchmarkLibraryHookCallsForTest()
{
    libraryHookCalls   = 0;
    hipEvent_t  event  = nullptr;
    hipStream_t stream = nullptr;
    double      time   = 0;
    // Force an emitted function body so the test also detects linker coalescing
    // in optimized builds, where an ordinary call could be inlined.
    auto volatile span = &hipblaslt_bench::pre_gpu_time;
    span(true, event, time, stream);
    return libraryHookCalls;
}
