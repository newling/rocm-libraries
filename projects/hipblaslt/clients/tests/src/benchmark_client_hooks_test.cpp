// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

namespace
{
    int clientHookCalls = 0;
}

// A different hook in another translation unit must retain its own definition.
#define HIPBLASLT_BENCHMARK_CHECK_HIP(expr) \
    do                                      \
    {                                       \
        ++clientHookCalls;                  \
        return;                             \
    } while(0)
#include <hipblaslt_benchmark/candidate_search.hpp>

int benchmarkClientHookCallsForTest()
{
    clientHookCalls    = 0;
    hipEvent_t  event  = nullptr;
    hipStream_t stream = nullptr;
    double      time   = 0;
    auto volatile span = &hipblaslt_bench::pre_gpu_time;
    span(true, event, time, stream);
    return clientHookCalls;
}
