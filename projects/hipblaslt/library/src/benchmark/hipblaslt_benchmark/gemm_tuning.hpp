// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "candidate_policy.hpp"
#include "candidate_search.hpp"
#include <functional>
#include <hipblaslt/hipblaslt.h>
#include <vector>

namespace hipblaslt_bench
{
    struct GemmTuningDefaults
    {
        static constexpr int  cold_iters         = 1000;
        static constexpr int  hot_iters          = 1000;
        static constexpr int  rotating_megabytes = 512;
        static constexpr bool flush              = true;
        static constexpr bool gpu_timer          = true;
    };

    struct GemmTuningParameters
    {
        uint16_t split_k = 0;
        int16_t  wgm     = 0;
    };

    inline bool gemm_tuning_eligible(bool c_equals_d, double beta)
    {
        // Repeated launches must not overwrite an input that the next launch reads.
        return !c_equals_d || beta == 0;
    }

    struct GemmTuningConfig
    {
        CandidatePolicy candidates;
        TimingConfig    timing;
        SearchConfig    search;
        // Explicit indexes retain caller order; enumeration otherwise uses candidates.
        bool             supplied_candidates = false;
        std::vector<int> indexes;
        // One parameter entry per supplied index; empty selects kernel defaults.
        std::vector<GemmTuningParameters> parameters;
        // Enumeration expands each supported index across these caller-selected variants.
        std::vector<GemmTuningParameters> variants;
    };

    struct GemmTuningCallbacks
    {
        // Returning false cancels enumeration before allocating timing resources.
        std::function<bool(const std::vector<int>&)> enumerated;
        std::function<void(double)>                  calibrated;
        std::function<void(size_t)>                  failed;
        std::function<void(size_t)>                  prepared;
        std::function<void(size_t, int32_t)>         cold;
        std::function<void(size_t, double, double)>  skipped;
        std::function<void(size_t)>                  measuring;
        std::function<bool(size_t, TimingResult&)>   measured;
        std::function<bool()>                        should_stop;
        std::function<bool()>                        should_abort;
    };

    struct GemmTuningResult
    {
        SearchResult                      search;
        std::vector<int>                  indexes;
        std::vector<GemmTuningParameters> parameters;
        std::vector<size_t>               workspace;
        std::vector<TimingResult>         measurements;
        size_t                            attempted = 0;
        size_t measured           = 0; // includes measurements vetoed by the caller
        bool   setup_failed       = false;
        bool   enumeration_failed = false;
        bool   ineligible         = false;
    };

    // Blocks differ only in their tensor pointers. Ownership
    // stays with the caller until the session has drained its stream.
    struct GemmTuningBlock
    {
        const void*           a;
        const void*           b;
        const void*           c;
        void*                 d;
        hipblasLtMatmulDesc_t descriptor = nullptr;
        const void*           alpha      = nullptr;
    };

    // Private build-tree entry point; not part of the installed hipBLASLt API.
    HIPBLASLT_EXPORT bool
        gemm_tuning_candidates(hipblasLtHandle_t                              handle,
                               hipblasLtMatmulDesc_t                          descriptor,
                               hipblasLtMatrixLayout_t                        a,
                               hipblasLtMatrixLayout_t                        b,
                               hipblasLtMatrixLayout_t                        c,
                               hipblasLtMatrixLayout_t                        d,
                               const void*                                    alpha,
                               const void*                                    beta,
                               const GemmTuningBlock&                         block,
                               size_t                                         workspace_bytes,
                               const CandidatePolicy&                         policy,
                               std::vector<hipblasLtMatmulHeuristicResult_t>& results,
                               const std::vector<GemmTuningParameters>&       variants = {},
                               std::vector<size_t>*    parameter_indexes               = nullptr,
                               const std::vector<int>* supplied_indexes                = nullptr);

    // Single GEMM blocks share layouts; optional descriptors and alpha pointers
    // supply each block's epilogue and scaling buffers. All inputs are
    // validated by the caller, and the descriptor must select the Tensile backend.
    // Successful measurements may be vetoed by the measured callback. Failed
    // candidates are reported and skipped; callbacks may stop or abort the search.
    HIPBLASLT_EXPORT void run_gemm_tuning(hipblasLtHandle_t                   handle,
                                          hipblasLtMatmulDesc_t               descriptor,
                                          hipblasLtMatrixLayout_t             a,
                                          hipblasLtMatrixLayout_t             b,
                                          hipblasLtMatrixLayout_t             c,
                                          hipblasLtMatrixLayout_t             d,
                                          const void*                         alpha,
                                          const void*                         beta,
                                          const std::vector<GemmTuningBlock>& blocks,
                                          void*                               workspace,
                                          size_t                              workspace_bytes,
                                          hipStream_t                         stream,
                                          const GemmTuningConfig&             config,
                                          const GemmTuningCallbacks&          callbacks,
                                          GemmTuningResult&                   result);
}
