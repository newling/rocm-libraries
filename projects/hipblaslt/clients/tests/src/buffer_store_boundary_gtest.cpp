// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace
{
    constexpr int64_t  kBelowM         = 8388599;
    constexpr int64_t  kPastM          = 8388607;
    constexpr int64_t  kN              = 256;
    constexpr int64_t  kK              = 48;
    constexpr size_t   kWorkspaceBytes = 128ull << 20;
    constexpr size_t   kTailElements   = 4096;
    constexpr uint16_t kBf16One        = 0x3f80;
    constexpr uint16_t kBf16NaN        = 0x7fc0;
    constexpr uint16_t kBf16FortyEight = 0x4240;

    struct Resources
    {
        hipblasLtHandle_t       handle = nullptr;
        hipblasLtMatmulDesc_t   desc   = nullptr;
        hipblasLtMatrixLayout_t a = nullptr, b = nullptr, d = nullptr;
        hipStream_t             stream    = nullptr;
        void*                   deviceA   = nullptr;
        void*                   deviceB   = nullptr;
        void*                   deviceD   = nullptr;
        void*                   workspace = nullptr;

        ~Resources()
        {
            if(stream)
                (void)hipStreamSynchronize(stream);
            if(deviceA)
                (void)hipFree(deviceA);
            if(deviceB)
                (void)hipFree(deviceB);
            if(deviceD)
                (void)hipFree(deviceD);
            if(workspace)
                (void)hipFree(workspace);
            if(a)
                hipblasLtMatrixLayoutDestroy(a);
            if(b)
                hipblasLtMatrixLayoutDestroy(b);
            if(d)
                hipblasLtMatrixLayoutDestroy(d);
            if(desc)
                hipblasLtMatmulDescDestroy(desc);
            if(handle)
                hipblasLtDestroy(handle);
            if(stream)
                (void)hipStreamDestroy(stream);
        }
    };

    void checkShape(Resources&                                           r,
                    int64_t                                              m,
                    const std::vector<hipblasLtMatmulHeuristicResult_t>& algorithms,
                    std::set<int>&                                       executed)
    {
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&r.a, HIP_R_16BF, m, kK, m), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&r.d, HIP_R_16BF, m, kN, m), HIPBLAS_STATUS_SUCCESS);

        const float alpha = 1.0f;
        const float beta  = 0.0f;
        auto*       tail  = static_cast<uint16_t*>(r.deviceD) + size_t(m) * kN - kTailElements;
        std::array<uint16_t, kTailElements> output;

        for(auto candidate : algorithms)
        {
            size_t     workspaceBytes = 0;
            const auto status         = hipblaslt_ext::matmulIsAlgoSupported(r.handle,
                                                                             r.desc,
                                                                             &alpha,
                                                                             r.a,
                                                                             r.b,
                                                                             &beta,
                                                                             r.d,
                                                                             r.d,
                                                                             candidate.algo,
                                                                             workspaceBytes);
            if(status == HIPBLAS_STATUS_INVALID_VALUE || status == HIPBLAS_STATUS_NOT_SUPPORTED)
                continue;
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
            if(workspaceBytes > kWorkspaceBytes)
                continue;

            const int index = hipblaslt_ext::getIndexFromAlgo(candidate.algo);
            SCOPED_TRACE(::testing::Message()
                         << "M=" << m << ", algorithm=" << index << ", "
                         << hipblaslt_ext::getSolutionNameFromAlgo(r.handle, candidate.algo));

            // Poison before EVERY launch: a previous working solution must not
            // hide a later solution's missing stores. A/B contain exact ones,
            // so the expected bf16 output is exactly K, with no CPU GEMM.
            ASSERT_EQ(hipMemsetD16Async(tail, kBf16NaN, kTailElements, r.stream), hipSuccess);
            ASSERT_EQ(hipblasLtMatmul(r.handle,
                                      r.desc,
                                      &alpha,
                                      r.deviceA,
                                      r.a,
                                      r.deviceB,
                                      r.b,
                                      &beta,
                                      r.deviceD,
                                      r.d,
                                      r.deviceD,
                                      r.d,
                                      &candidate.algo,
                                      r.workspace,
                                      workspaceBytes,
                                      r.stream),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipMemcpyAsync(
                          output.data(), tail, sizeof(output), hipMemcpyDeviceToHost, r.stream),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(r.stream), hipSuccess);
            const auto wrong = std::find_if(output.begin(), output.end(), [](uint16_t value) {
                return value != kBf16FortyEight;
            });
            if(wrong != output.end())
                FAIL() << "Wrong or unwritten output at row "
                       << m - kTailElements + std::distance(output.begin(), wrong)
                       << " of the last column: expected bf16 bits 0x" << std::hex
                       << kBf16FortyEight << ", observed 0x" << *wrong;
            executed.insert(index);
        }

        ASSERT_FALSE(executed.empty()) << "No supported solution executed for M=" << m;
        ASSERT_EQ(hipblasLtMatrixLayoutDestroy(r.a), HIPBLAS_STATUS_SUCCESS);
        r.a = nullptr;
        ASSERT_EQ(hipblasLtMatrixLayoutDestroy(r.d), HIPBLAS_STATUS_SUCCESS);
        r.d = nullptr;
    }
}

// About 4.9 GiB of device memory, reused for both shapes and every solution.
// The suite name puts this in the shared nightly/comprehensive test tier.
TEST(nightly_BufferStoreBoundary, Bf16OutputIsWritten_ROCM31016)
{
    int device = 0;
    ASSERT_EQ(hipGetDevice(&device), hipSuccess);
    hipDeviceProp_t properties{};
    ASSERT_EQ(hipGetDeviceProperties(&properties, device), hipSuccess);
    const std::string architecture(properties.gcnArchName);
    if(architecture.find("gfx942") != 0 && architecture.find("gfx950") != 0)
        GTEST_SKIP() << "Requires the gfx942/gfx950 bf16 tuned libraries";

    Resources r;
    ASSERT_EQ(hipStreamCreate(&r.stream), hipSuccess);
    ASSERT_EQ(hipblasLtCreate(&r.handle), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtMatmulDescCreate(&r.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
              HIPBLAS_STATUS_SUCCESS);
    const hipblasOperation_t op = HIPBLAS_OP_N;
    ASSERT_EQ(
        hipblasLtMatmulDescSetAttribute(r.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op)),
        HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        hipblasLtMatmulDescSetAttribute(r.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op, sizeof(op)),
        HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtMatrixLayoutCreate(&r.b, HIP_R_16BF, kK, kN, kK), HIPBLAS_STATUS_SUCCESS);

    // Enumeration plus the normal support check avoids depending on one
    // heuristic pick or an algorithm index that changes when kernels are tuned.
    std::vector<hipblasLtMatmulHeuristicResult_t> algorithms;
    ASSERT_EQ(hipblaslt_ext::getAllAlgos(r.handle,
                                         hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                                         op,
                                         op,
                                         HIP_R_16BF,
                                         HIP_R_16BF,
                                         HIP_R_16BF,
                                         HIP_R_16BF,
                                         HIPBLAS_COMPUTE_32F,
                                         algorithms),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_FALSE(algorithms.empty());
    std::sort(algorithms.begin(), algorithms.end(), [](auto lhs, auto rhs) {
        return hipblaslt_ext::getIndexFromAlgo(lhs.algo)
               < hipblaslt_ext::getIndexFromAlgo(rhs.algo);
    });

    for(auto allocation : {std::make_pair(&r.deviceA, size_t(kPastM) * kK * sizeof(uint16_t)),
                           std::make_pair(&r.deviceB, size_t(kK) * kN * sizeof(uint16_t)),
                           std::make_pair(&r.deviceD, size_t(kPastM) * kN * sizeof(uint16_t)),
                           std::make_pair(&r.workspace, kWorkspaceBytes)})
    {
        const auto status = hipMalloc(allocation.first, allocation.second);
        if(status == hipErrorOutOfMemory)
            GTEST_SKIP() << "Insufficient device memory for the store-boundary regression";
        ASSERT_EQ(status, hipSuccess);
    }
    ASSERT_EQ(hipMemsetD16Async(r.deviceA, kBf16One, size_t(kPastM) * kK, r.stream), hipSuccess);
    ASSERT_EQ(hipMemsetD16Async(r.deviceB, kBf16One, size_t(kK) * kN, r.stream), hipSuccess);

    // For 256-column tiles these extents bracket 0xfffff000. The larger
    // shape's final partial tile was left unwritten by the affected kernels.
    // The control also ensures tuning has retained solutions at this boundary.
    std::set<int> below, past;
    ASSERT_NO_FATAL_FAILURE(checkShape(r, kBelowM, algorithms, below));
    RecordProperty("below_boundary_solutions", static_cast<int>(below.size()));
    ASSERT_NO_FATAL_FAILURE(checkShape(r, kPastM, algorithms, past));
    RecordProperty("past_boundary_solutions", static_cast<int>(past.size()));
    EXPECT_TRUE(std::any_of(below.begin(), below.end(), [&](int index) {
        return past.count(index) == 0;
    })) << "No boundary-sensitive solution was exercised; check the tuned library coverage";
}
