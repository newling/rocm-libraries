// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt_benchmark/candidate_policy.hpp>
#include <hipblaslt_benchmark/gemm_rotation.hpp>
#include <hipblaslt_benchmark/gemm_tuning.hpp>

#include <array>
#include <cstring>
#include <stdexcept>
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

TEST(SharedGemmPolicy_pre_checkin, EpilogueRotationAndInputEligibility)
{
    using namespace hipblaslt_bench;
    const GemmRotationTensor input{2, 64, 32, 72, 2, 4096};
    const GemmRotationTensor output{2, 64, 32, 80, 2, 4096};
    const GemmRotationTensor bias{4, 32, 1, 32, 2, 80};
    const auto               forward
        = gemm_rotation_layout(input, input, output, output, output, bias, false, false);
    ASSERT_TRUE(forward.usable);
    EXPECT_TRUE(forward.rotatable);
    EXPECT_EQ(forward.spans.c, forward.spans.d);
    EXPECT_EQ(forward.spans.e, forward.spans.d);
    EXPECT_EQ(forward.spans.bias, 0u);
    const auto gradient
        = gemm_rotation_layout(input, input, output, output, output, bias, false, true);
    EXPECT_EQ(gradient.spans.bias, (80u + 32) * 4);
    EXPECT_GT(gemm_rotation_bytes(gradient.spans), gemm_rotation_bytes(forward.spans));
    auto broadcast         = input;
    broadcast.batch_stride = 0;
    EXPECT_FALSE(
        gemm_rotation_layout(broadcast, input, output, output, {}, {}, false, false).rotatable);
    EXPECT_FALSE(
        gemm_rotation_layout(input, input, output, output, broadcast, bias, false, true).usable);
    EXPECT_TRUE(gemm_tuning_eligible(false, 1));
    EXPECT_TRUE(gemm_tuning_eligible(true, 0));
    EXPECT_FALSE(gemm_tuning_eligible(true, 1));
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

TEST(SharedGemmPolicy_pre_checkin, SessionRotatesMeasuresAndPreservesReplay)
{
    using namespace hipblaslt_bench;
    int devices = 0;
    if(hipGetDeviceCount(&devices) != hipSuccess || devices == 0)
        GTEST_SKIP() << "No GPU available";
    constexpr int    dimension      = 128;
    constexpr size_t elements       = dimension * dimension;
    constexpr size_t bytes          = elements * sizeof(uint16_t);
    constexpr size_t workspaceBytes = size_t(32) << 20;
    struct Resources
    {
        hipblasLtHandle_t       handle    = nullptr;
        hipblasLtMatmulDesc_t   desc      = nullptr;
        hipblasLtMatrixLayout_t layout    = nullptr;
        hipStream_t             stream    = nullptr;
        void*                   tensors   = nullptr;
        void*                   workspace = nullptr;
        ~Resources()
        {
            if(stream)
                hipStreamSynchronize(stream);
            if(workspace)
                hipFree(workspace);
            if(tensors)
                hipFree(tensors);
            if(layout)
                hipblasLtMatrixLayoutDestroy(layout);
            if(desc)
                hipblasLtMatmulDescDestroy(desc);
            if(handle)
                hipblasLtDestroy(handle);
            if(stream)
                hipStreamDestroy(stream);
        }
    } r;
    ASSERT_EQ(hipblasLtCreate(&r.handle), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtMatmulDescCreate(&r.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtMatrixLayoutCreate(&r.layout, HIP_R_16F, dimension, dimension, dimension),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipStreamCreate(&r.stream), hipSuccess);
    ASSERT_EQ(hipMalloc(&r.tensors, bytes * 8), hipSuccess);
    ASSERT_EQ(hipMalloc(&r.workspace, workspaceBytes), hipSuccess);
    auto pointer = [&](int tensor) { return static_cast<char*>(r.tensors) + tensor * bytes; };
    std::vector<GemmTuningBlock> blocks;
    for(int b = 0; b < 2; ++b)
    {
        blocks.push_back(
            {pointer(4 * b), pointer(4 * b + 1), pointer(4 * b + 2), pointer(4 * b + 3)});
        for(int tensor = 0; tensor < 3; ++tensor)
        {
            // Distinct products expose a session that repeatedly launches only block zero.
            const uint16_t value = tensor == 2 ? 0x4200 : tensor == 0 && b == 1 ? 0x4000 : 0x3c00;
            std::vector<uint16_t> input(elements, value);
            ASSERT_EQ(
                hipMemcpy(pointer(4 * b + tensor), input.data(), bytes, hipMemcpyHostToDevice),
                hipSuccess);
        }
    }
    const float alpha = 1, beta = 1;
    auto        run = [&](const GemmTuningConfig&    config,
                   const GemmTuningCallbacks& callbacks,
                   GemmTuningResult&          result) {
        run_gemm_tuning(r.handle,
                        r.desc,
                        r.layout,
                        r.layout,
                        r.layout,
                        r.layout,
                        &alpha,
                        &beta,
                        blocks,
                        r.workspace,
                        workspaceBytes,
                        r.stream,
                        config,
                        callbacks,
                        result);
    };
    auto verify = [&] {
        for(int b = 0; b < 2; ++b)
        {
            std::vector<uint16_t> output(elements);
            ASSERT_EQ(hipMemcpy(output.data(), blocks[b].d, bytes, hipMemcpyDeviceToHost),
                      hipSuccess);
            const _Float16 expected = static_cast<_Float16>(dimension * (b + 1) + 3);
            uint16_t       bits;
            std::memcpy(&bits, &expected, sizeof(bits));
            for(auto value : output)
                ASSERT_EQ(value, bits) << "rotation block " << b;
        }
    };
    for(bool exhaustive : {false, true})
    {
        for(bool gpuTimer : {false, true})
        {
            GemmTuningConfig config;
            config.candidates           = {exhaustive, 2, -1};
            config.search.cold_calls    = 2;
            config.timing.iters         = 2;
            config.timing.use_gpu_timer = gpuTimer;
            std::vector<hipblasLtMatmulHeuristicResult_t> expected;
            ASSERT_TRUE(gemm_tuning_candidates(r.handle,
                                               r.desc,
                                               r.layout,
                                               r.layout,
                                               r.layout,
                                               r.layout,
                                               &alpha,
                                               &beta,
                                               blocks[0],
                                               workspaceBytes,
                                               config.candidates,
                                               expected));
            if(expected.size() < 2)
                GTEST_SKIP() << "Two supported fp16 NN algorithms required";
            GemmTuningCallbacks callbacks;
            std::vector<size_t> visited;
            callbacks.prepared = [&](size_t c) { visited.push_back(c); };
            GemmTuningResult result;
            run(config, callbacks, result);
            ASSERT_FALSE(result.setup_failed);
            ASSERT_FALSE(result.enumeration_failed);
            ASSERT_FALSE(result.search.aborted);
            EXPECT_EQ(result.attempted, 2u);
            EXPECT_EQ(result.measured, 2u);
            EXPECT_EQ(visited, (std::vector<size_t>{0, 1}));
            ASSERT_LT(result.search.best, 2u);
            const size_t best
                = result.measurements[1].median_us < result.measurements[0].median_us ? 1 : 0;
            EXPECT_EQ(result.search.best, best);
            for(size_t i = 0; i < expected.size(); ++i)
            {
                EXPECT_EQ(result.indexes[i], hipblaslt_ext::getIndexFromAlgo(expected[i].algo));
                EXPECT_EQ(result.workspace[i], expected[i].workspaceSize);
                EXPECT_EQ(result.measurements[i].hot_iters, 2);
            }
            verify();

            callbacks.should_stop = [&] { return result.measured == 1; };
            run(config, callbacks, result);
            EXPECT_TRUE(result.search.stopped);
            EXPECT_EQ(result.measured, 1u);
            EXPECT_EQ(result.search.best, 0u);
            callbacks.should_stop = {};
            callbacks.measured    = [](size_t c, TimingResult&) { return c == 1; };
            run(config, callbacks, result);
            EXPECT_EQ(result.measured, 2u);
            EXPECT_EQ(result.search.best, 1u);

            config.supplied_candidates = true;
            config.indexes             = {-1, hipblaslt_ext::getIndexFromAlgo(expected[0].algo)};
            std::vector<size_t> failures;
            callbacks.failed = [&](size_t c) { failures.push_back(c); };
            run(config, callbacks, result);
            EXPECT_EQ(failures, (std::vector<size_t>{0}));
            EXPECT_EQ(result.attempted, 2u);
            EXPECT_EQ(result.measured, 1u);
            EXPECT_EQ(result.search.best, 1u);
            verify();

            config.indexes                 = {hipblaslt_ext::getIndexFromAlgo(expected[0].algo)};
            config.timing.adaptive         = true;
            config.timing.warmup_time      = 0.1f;
            config.timing.sample_time      = 0.1f;
            config.timing.measure_time     = 0.2f;
            config.timing.max_measure_time = 1.0f;
            config.timing.min_iters        = 2;
            config.timing.max_iters        = 128;
            callbacks                      = {};
            run(config, callbacks, result);
            ASSERT_EQ(result.measured, 1u);
            EXPECT_EQ(result.search.best, 0u);
            EXPECT_TRUE(result.search.best_timing.adaptive);
            EXPECT_GE(result.search.best_timing.hot_iters, 2);
            verify();

            bool abort             = false;
            callbacks.cold         = [&](size_t, int32_t) { abort = true; };
            callbacks.should_abort = [&] { return abort; };
            run(config, callbacks, result);
            EXPECT_TRUE(result.search.aborted);
            EXPECT_EQ(result.measured, 0u);
            EXPECT_EQ(hipStreamQuery(r.stream), hipSuccess);
            callbacks = {};

            callbacks.cold
                = [&](size_t, int32_t) { throw std::runtime_error("consumer callback failed"); };
            EXPECT_THROW(run(config, callbacks, result), std::runtime_error);
            EXPECT_EQ(hipStreamQuery(r.stream), hipSuccess);
            verify();

            // A session must leave the descriptor usable by the public launch path.
            ASSERT_EQ(hipMemset(blocks[0].d, 0, bytes), hipSuccess);
            ASSERT_EQ(hipblasLtMatmul(r.handle,
                                      r.desc,
                                      &alpha,
                                      blocks[0].a,
                                      r.layout,
                                      blocks[0].b,
                                      r.layout,
                                      &beta,
                                      blocks[0].c,
                                      r.layout,
                                      blocks[0].d,
                                      r.layout,
                                      &expected[0].algo,
                                      r.workspace,
                                      workspaceBytes,
                                      r.stream),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipStreamSynchronize(r.stream), hipSuccess);
            verify();
        }
    }
    GemmTuningConfig variants;
    variants.candidates        = {false, 2, -1};
    variants.variants          = {{0, 0}, {1, 0}, {2, 1}};
    variants.search.cold_calls = 2;
    variants.timing.iters      = 2;
    GemmTuningResult variantResult;
    run(variants, {}, variantResult);
    ASSERT_FALSE(variantResult.enumeration_failed);
    ASSERT_FALSE(variantResult.setup_failed);
    EXPECT_EQ(variantResult.measured, variantResult.indexes.size());
    verify();
    std::vector<hipblasLtMatmulHeuristicResult_t> variantCandidates;
    std::vector<size_t>                           variantIndexes;
    ASSERT_TRUE(gemm_tuning_candidates(r.handle,
                                       r.desc,
                                       r.layout,
                                       r.layout,
                                       r.layout,
                                       r.layout,
                                       &alpha,
                                       &beta,
                                       blocks[0],
                                       workspaceBytes,
                                       variants.candidates,
                                       variantCandidates,
                                       variants.variants,
                                       &variantIndexes));
    ASSERT_EQ(variantCandidates.size(), variantResult.indexes.size());
    for(size_t i = 0; i < variantCandidates.size(); ++i)
    {
        EXPECT_EQ(variantResult.indexes[i],
                  hipblaslt_ext::getIndexFromAlgo(variantCandidates[i].algo));
        EXPECT_EQ(variantResult.parameters[i].split_k,
                  variants.variants[variantIndexes[i]].split_k);
        EXPECT_EQ(variantResult.parameters[i].wgm, variants.variants[variantIndexes[i]].wgm);
        EXPECT_EQ(variantResult.workspace[i], variantCandidates[i].workspaceSize);
    }
    auto savedC = blocks[0].c;
    blocks[0].c = blocks[0].d;
    run(variants, {}, variantResult);
    EXPECT_TRUE(variantResult.ineligible);
    EXPECT_EQ(variantResult.attempted, 0u);
    blocks[0].c = savedC;
    GemmTuningConfig malformed;
    malformed.supplied_candidates = true;
    malformed.indexes             = {0, 1};
    malformed.parameters          = {{0, 0}};
    run(malformed, {}, variantResult);
    EXPECT_TRUE(variantResult.setup_failed);

    GemmTuningConfig empty;
    empty.supplied_candidates = true;
    GemmTuningResult result;
    run(empty, {}, result);
    EXPECT_TRUE(result.enumeration_failed);
    EXPECT_EQ(result.attempted, 0u);
    empty.supplied_candidates = false;
    empty.candidates          = {false, 2, -1};
    GemmTuningCallbacks canceled;
    canceled.enumerated = [](const std::vector<int>&) { return false; };
    run(empty, canceled, result);
    EXPECT_TRUE(result.enumeration_failed);
    EXPECT_EQ(result.attempted, 0u);
}

TEST(SharedGemmPolicy_pre_checkin, SessionUsesEveryBlocksEpilogueAndScalePointers)
{
    using namespace hipblaslt_bench;
    int devices = 0;
    if(hipGetDeviceCount(&devices) != hipSuccess || devices == 0)
        GTEST_SKIP() << "No GPU available";
    constexpr size_t dimension      = 128;
    constexpr size_t elements       = dimension * dimension;
    constexpr size_t bytes          = elements * sizeof(uint16_t);
    constexpr size_t workspaceBytes = size_t(32) << 20;
    struct Resources
    {
        hipblasLtHandle_t                    handle = nullptr;
        hipblasLtMatrixLayout_t              layout = nullptr;
        std::array<hipblasLtMatmulDesc_t, 2> descriptors{};
        hipStream_t                          stream    = nullptr;
        void*                                tensors   = nullptr;
        void*                                vectors   = nullptr;
        void*                                workspace = nullptr;
        ~Resources()
        {
            if(stream)
                static_cast<void>(hipStreamSynchronize(stream));
            for(auto descriptor : descriptors)
                if(descriptor)
                    hipblasLtMatmulDescDestroy(descriptor);
            if(layout)
                hipblasLtMatrixLayoutDestroy(layout);
            if(handle)
                hipblasLtDestroy(handle);
            for(void* allocation : {tensors, vectors, workspace})
                if(allocation)
                    static_cast<void>(hipFree(allocation));
            if(stream)
                static_cast<void>(hipStreamDestroy(stream));
        }
    } r;
    ASSERT_EQ(hipblasLtCreate(&r.handle), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtMatrixLayoutCreate(&r.layout, HIP_R_16F, dimension, dimension, dimension),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipStreamCreate(&r.stream), hipSuccess);
    ASSERT_EQ(hipMalloc(&r.tensors, bytes * 10), hipSuccess);
    ASSERT_EQ(hipMalloc(&r.vectors, dimension * sizeof(float) * 4 + 4 * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&r.workspace, workspaceBytes), hipSuccess);
    auto tensor = [&](int i) { return static_cast<char*>(r.tensors) + i * bytes; };
    auto vector = [&](int i) { return static_cast<float*>(r.vectors) + i * dimension; };
    auto scales = vector(4);
    const std::array<float, 4> scaleValues{0.5f, 1, 1, 1};
    ASSERT_EQ(hipMemcpy(scales, scaleValues.data(), sizeof(scaleValues), hipMemcpyHostToDevice),
              hipSuccess);
    std::vector<GemmTuningBlock> blocks;
    for(int b = 0; b < 2; ++b)
    {
        ASSERT_EQ(hipblasLtMatmulDescCreate(&r.descriptors[b], HIPBLAS_COMPUTE_32F, HIP_R_32F),
                  HIPBLAS_STATUS_SUCCESS);
        auto op = HIPBLAS_OP_T;
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                      r.descriptors[b], HIPBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op)),
                  HIPBLAS_STATUS_SUCCESS);
        blocks.push_back({tensor(5 * b),
                          tensor(5 * b + 1),
                          tensor(5 * b + 2),
                          tensor(5 * b + 3),
                          r.descriptors[b],
                          vector(2 * b + 1)});
        for(int i = 0; i < 3; ++i)
        {
            std::vector<uint16_t> values(elements, i == 2 ? 0x4200 : 0x2c00); // C=3, A=B=1/16
            ASSERT_EQ(hipMemcpy(tensor(5 * b + i), values.data(), bytes, hipMemcpyHostToDevice),
                      hipSuccess);
        }
        for(int i = 0; i < 2; ++i)
        {
            std::vector<float> values(dimension, i == 0 ? float(b + 1) / 4 : float(2 * (b + 1)));
            ASSERT_EQ(hipMemcpy(vector(2 * b + i),
                                values.data(),
                                dimension * sizeof(float),
                                hipMemcpyHostToDevice),
                      hipSuccess);
        }
    }
    const float alpha = 1, beta = 1;
    auto        set = [&](int b, hipblasLtMatmulDescAttributes_t attr, const auto& value) {
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(r.descriptors[b], attr, &value, sizeof(value)),
                  HIPBLAS_STATUS_SUCCESS);
    };
    for(bool auxiliary : {false, true})
    {
        for(int b = 0; b < 2; ++b)
        {
            set(b,
                HIPBLASLT_MATMUL_DESC_EPILOGUE,
                auxiliary ? HIPBLASLT_EPILOGUE_GELU_AUX_BIAS : HIPBLASLT_EPILOGUE_BIAS);
            set(b, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, HIP_R_32F);
            void* bias = vector(2 * b);
            set(b, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, bias);
            for(int i = 0; i < 4; ++i)
            {
                void* pointer = auxiliary ? nullptr : scales + i;
                const std::array<hipblasLtMatmulDescAttributes_t, 4> attributes{
                    HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                    HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                    HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER,
                    HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER};
                set(b, attributes[i], pointer);
            }
            set(b,
                HIPBLASLT_MATMUL_DESC_POINTER_MODE,
                auxiliary ? HIPBLASLT_POINTER_MODE_HOST
                          : HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST);
            blocks[b].alpha = auxiliary ? &alpha : vector(2 * b + 1);
            void* e         = auxiliary ? tensor(5 * b + 4) : nullptr;
            set(b, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, e);
            const int64_t ld = dimension;
            set(b, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, ld);
            set(b, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE, HIP_R_16F);
        }
        GemmTuningConfig config;
        config.candidates        = {false, 1, -1};
        config.search.cold_calls = 2;
        config.timing.iters      = 2;
        std::vector<hipblasLtMatmulHeuristicResult_t> candidates;
        ASSERT_TRUE(gemm_tuning_candidates(r.handle,
                                           r.descriptors[0],
                                           r.layout,
                                           r.layout,
                                           r.layout,
                                           r.layout,
                                           blocks[0].alpha,
                                           &beta,
                                           blocks[0],
                                           workspaceBytes,
                                           config.candidates,
                                           candidates));
        if(candidates.empty())
            GTEST_SKIP() << "No fp16 TN algorithm for "
                         << (auxiliary ? "GELU auxiliary" : "scaling and bias");
        std::array<std::vector<uint16_t>, 2> expectedD, expectedE;
        for(int b = 0; b < 2; ++b)
        {
            ASSERT_EQ(hipblasLtMatmul(r.handle,
                                      r.descriptors[b],
                                      blocks[b].alpha,
                                      blocks[b].a,
                                      r.layout,
                                      blocks[b].b,
                                      r.layout,
                                      &beta,
                                      blocks[b].c,
                                      r.layout,
                                      blocks[b].d,
                                      r.layout,
                                      &candidates[0].algo,
                                      r.workspace,
                                      workspaceBytes,
                                      r.stream),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipStreamSynchronize(r.stream), hipSuccess);
            expectedD[b].resize(elements);
            ASSERT_EQ(hipMemcpy(expectedD[b].data(), blocks[b].d, bytes, hipMemcpyDeviceToHost),
                      hipSuccess);
            if(auxiliary)
            {
                expectedE[b].resize(elements);
                ASSERT_EQ(
                    hipMemcpy(expectedE[b].data(), tensor(5 * b + 4), bytes, hipMemcpyDeviceToHost),
                    hipSuccess);
            }
        }
        EXPECT_NE(expectedD[0], expectedD[1]);
        ASSERT_EQ(hipMemset(blocks[0].d, 0, bytes), hipSuccess);
        ASSERT_EQ(hipMemset(blocks[1].d, 0, bytes), hipSuccess);
        if(auxiliary)
            for(int b = 0; b < 2; ++b)
                ASSERT_EQ(hipMemset(tensor(5 * b + 4), 0, bytes), hipSuccess);
        GemmTuningResult result;
        run_gemm_tuning(r.handle,
                        r.descriptors[0],
                        r.layout,
                        r.layout,
                        r.layout,
                        r.layout,
                        blocks[0].alpha,
                        &beta,
                        blocks,
                        r.workspace,
                        workspaceBytes,
                        r.stream,
                        config,
                        {},
                        result);
        ASSERT_FALSE(result.setup_failed);
        ASSERT_FALSE(result.enumeration_failed);
        ASSERT_EQ(result.measured, 1u);
        for(int b = 0; b < 2; ++b)
        {
            std::vector<uint16_t> actual(elements);
            ASSERT_EQ(hipMemcpy(actual.data(), blocks[b].d, bytes, hipMemcpyDeviceToHost),
                      hipSuccess);
            EXPECT_EQ(actual, expectedD[b]) << "rotation block " << b;
            if(auxiliary)
            {
                ASSERT_EQ(hipMemcpy(actual.data(), tensor(5 * b + 4), bytes, hipMemcpyDeviceToHost),
                          hipSuccess);
                EXPECT_EQ(actual, expectedE[b]) << "auxiliary block " << b;
            }
            else
            {
                const _Float16 expected = b == 0 ? 3.75f : 4.5f;
                uint16_t       bits;
                std::memcpy(&bits, &expected, sizeof(bits));
                for(auto value : actual)
                    ASSERT_EQ(value, bits);
            }
        }
    }
}
