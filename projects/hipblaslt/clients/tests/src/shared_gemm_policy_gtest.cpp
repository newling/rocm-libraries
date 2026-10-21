// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt_benchmark/candidate_policy.hpp>
#include <hipblaslt_benchmark/gemm_rotation.hpp>

#include <array>
#include <vector>

extern "C" size_t hipblaslt_tuning_candidates_for_test(
    void*, void*, void*, void*, void*, void*, int, size_t, int, size_t, int*, size_t);

TEST(SharedGemmPolicy_pre_checkin, ShortRankingFillsAfterFilteringAndCountsBaseline)
{
    auto ranked = [] { return std::vector<int>{8, 8, 9}; };
    auto all    = [] { return std::vector<int>{1, 2, 8, 9, 10}; };
    // 9 is unavailable at replay; 1 exceeds the workspace limit.
    auto accept = [](int index) { return index != 9 && index != 1; };
    EXPECT_EQ(hipblaslt_bench::select_candidate_indexes({false, 3, 10}, ranked, all, accept),
              (std::vector<int>{10, 8, 2}));
    EXPECT_EQ(hipblaslt_bench::select_candidate_indexes({true, 0, -1}, ranked, all, accept),
              (std::vector<int>{2, 8, 10}));
    EXPECT_EQ(hipblaslt_bench::select_candidate_indexes(
                  {false, 1, 10},
                  []() -> std::vector<int> {
                      ADD_FAILURE() << "ranking after full limit";
                      return {};
                  },
                  []() -> std::vector<int> {
                      ADD_FAILURE() << "enumeration after full limit";
                      return {};
                  },
                  accept),
              (std::vector<int>{10}));
}

TEST(SharedGemmPolicy_pre_checkin, DistinctAndAliasedCRotation)
{
    constexpr size_t tensor = 1024 * 1024 * 2;
    for(bool inPlace : {false, true})
    {
        const auto spans = hipblaslt_bench::gemm_rotation_spans(
            {tensor, tensor, tensor, tensor, 0, 128}, inPlace, false);
        const auto plan = hipblaslt_bench::plan_gemm_rotation(
            spans, true, false, 0, 1000, 1000, size_t(512) << 20);
        EXPECT_EQ(spans.c, inPlace ? 0u : tensor);
        EXPECT_EQ(plan.block_count, inPlace ? 86 : 64);
        EXPECT_EQ(hipblaslt_bench::plan_gemm_rotation(
                      spans, false, false, 0, 1000, 1000, size_t(512) << 20)
                      .block_count,
                  1);
    }
    EXPECT_EQ(hipblaslt_bench::tensor_span_bytes(2, 64, 32, 72, 1, 0), (72u * 31 + 64) * 2);
}

TEST(SharedGemmPolicy_pre_checkin, PublicHeuristicAndRuntimeUseSameCandidates)
{
    int devices = 0;
    if(hipGetDeviceCount(&devices) != hipSuccess || devices == 0)
        GTEST_SKIP() << "No GPU available";
    hipblasLtHandle_t handle = nullptr;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    hipblasLtMatmulDesc_t                  desc = nullptr;
    hipblasLtMatmulPreference_t            pref = nullptr;
    std::array<hipblasLtMatrixLayout_t, 4> layouts{};
    struct Resources
    {
        hipblasLtHandle_t&                      handle;
        hipblasLtMatmulDesc_t&                  desc;
        hipblasLtMatmulPreference_t&            pref;
        std::array<hipblasLtMatrixLayout_t, 4>& layouts;
        ~Resources()
        {
            if(pref)
                hipblasLtMatmulPreferenceDestroy(pref);
            for(auto layout : layouts)
                if(layout)
                    hipblasLtMatrixLayoutDestroy(layout);
            if(desc)
                hipblasLtMatmulDescDestroy(desc);
            hipblasLtDestroy(handle);
        }
    } resources{handle, desc, pref, layouts};
    ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layouts[0], HIP_R_16F, 1024, 1024, 1024),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layouts[1], HIP_R_16F, 1024, 512, 1024),
              HIPBLAS_STATUS_SUCCESS);
    for(int i : {2, 3})
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layouts[i], HIP_R_16F, 1024, 512, 1024),
                  HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
    const size_t workspace = size_t(32) << 20;
    ASSERT_EQ(hipblasLtMatmulPreferenceSetAttribute(
                  pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace)),
              HIPBLAS_STATUS_SUCCESS);
    std::vector<hipblasLtMatmulHeuristicResult_t> publicResults(16);
    int                                           count = 0;
    ASSERT_EQ(hipblasLtMatmulAlgoGetHeuristic(handle,
                                              desc,
                                              layouts[0],
                                              layouts[1],
                                              layouts[2],
                                              layouts[3],
                                              pref,
                                              16,
                                              publicResults.data(),
                                              &count),
              HIPBLAS_STATUS_SUCCESS);
    std::vector<int> expected;
    for(int i = 0; i < count; ++i)
        expected.push_back(hipblaslt_ext::getIndexFromAlgo(publicResults[i].algo));
    std::vector<int> runtime(16);
    const auto       runtimeCount = hipblaslt_tuning_candidates_for_test(handle,
                                                                   desc,
                                                                   layouts[0],
                                                                   layouts[1],
                                                                   layouts[2],
                                                                   layouts[3],
                                                                   0,
                                                                   16,
                                                                   -1,
                                                                   workspace,
                                                                   runtime.data(),
                                                                   runtime.size());
    runtime.resize(runtimeCount);
    EXPECT_EQ(runtime, expected);
    std::vector<hipblasLtMatmulHeuristicResult_t> all;
    ASSERT_EQ(hipblaslt_ext::getAllAlgos(handle,
                                         hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                                         HIPBLAS_OP_N,
                                         HIPBLAS_OP_N,
                                         HIP_R_16F,
                                         HIP_R_16F,
                                         HIP_R_16F,
                                         HIP_R_16F,
                                         HIPBLAS_COMPUTE_32F,
                                         all),
              HIPBLAS_STATUS_SUCCESS);
    std::vector<int> exhaustive;
    float            alpha = 1.0f, beta = 1.0f;
    for(auto& result : all)
    {
        size_t required                 = 0;
        result.algo.max_workspace_bytes = workspace;
        if(hipblaslt_ext::matmulIsAlgoSupported(handle,
                                                desc,
                                                &alpha,
                                                layouts[0],
                                                layouts[1],
                                                &beta,
                                                layouts[2],
                                                layouts[3],
                                                result.algo,
                                                required)
               == HIPBLAS_STATUS_SUCCESS
           && required <= workspace)
            exhaustive.push_back(hipblaslt_ext::getIndexFromAlgo(result.algo));
        if(exhaustive.size() == 16)
            break;
    }
    runtime.resize(16);
    const auto exhaustiveCount = hipblaslt_tuning_candidates_for_test(handle,
                                                                      desc,
                                                                      layouts[0],
                                                                      layouts[1],
                                                                      layouts[2],
                                                                      layouts[3],
                                                                      1,
                                                                      16,
                                                                      -1,
                                                                      workspace,
                                                                      runtime.data(),
                                                                      runtime.size());
    runtime.resize(exhaustiveCount);
    EXPECT_EQ(runtime, exhaustive);
    if(!expected.empty())
    {
        // A caller-selected baseline moves to the front and still consumes one slot.
        const int  baseline     = expected.back();
        const auto withBaseline = hipblaslt_tuning_candidates_for_test(handle,
                                                                       desc,
                                                                       layouts[0],
                                                                       layouts[1],
                                                                       layouts[2],
                                                                       layouts[3],
                                                                       0,
                                                                       1,
                                                                       baseline,
                                                                       workspace,
                                                                       runtime.data(),
                                                                       1);
        EXPECT_EQ(withBaseline, 1u);
        EXPECT_EQ(runtime[0], baseline);
    }
    if(expected.empty())
        GTEST_SKIP() << "No fp16 NN algorithm available";
}
