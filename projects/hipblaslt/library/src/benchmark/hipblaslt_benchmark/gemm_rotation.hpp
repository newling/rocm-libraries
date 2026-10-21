// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "benchmark_stats.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace hipblaslt_bench
{
    // A rounded-up stride is an upper bound for output storage, but cannot be
    // copied from a broadcast input. expanded tells input owners to decline rotation.
    inline size_t tensor_span_bytes(size_t elementSize,
                                    size_t rows,
                                    size_t cols,
                                    size_t colStride,
                                    size_t batchCount,
                                    size_t batchStride,
                                    bool*  expanded = nullptr)
    {
        if(expanded)
            *expanded = false;

        if(elementSize == 0 || rows == 0 || cols == 0)
            return 0;

        const size_t batches = std::max<size_t>(batchCount, 1);

        // A column stride below the row count would overlap columns; treat as
        // dense rather than under-allocating.
        const size_t effectiveColStride = std::max(colStride, rows);
        if(expanded && effectiveColStride != colStride)
            *expanded = true;

        // Every guard precedes the multiplication it protects. Checking after
        // the fact reads an already-wrapped value and lets a too-small buffer
        // through.
        if(cols > 1 && effectiveColStride > (SIZE_MAX - rows) / (cols - 1))
            return 0;

        const size_t perBatch = effectiveColStride * (cols - 1) + rows;
        const size_t effectiveBatchStride
            = (batches > 1) ? std::max(batchStride, perBatch) : perBatch;
        if(expanded && batches > 1 && effectiveBatchStride != batchStride)
            *expanded = true;

        if(batches > 1 && effectiveBatchStride > (SIZE_MAX - perBatch) / (batches - 1))
            return 0;

        const size_t elements = effectiveBatchStride * (batches - 1) + perBatch;
        if(elements > (SIZE_MAX / elementSize))
            return 0;

        return elements * elementSize;
    }

    struct GemmRotationSpans
    {
        size_t a = 0, b = 0, c = 0, d = 0, e = 0, bias = 0;
    };

    struct GemmRotationTensor
    {
        size_t element_bytes = 0, rows = 0, cols = 0, ld = 0, batches = 1, batch_stride = 0;
        bool   swizzled = false;
    };

    struct GemmRotationLayout
    {
        GemmRotationSpans spans;
        bool              usable    = false;
        bool              rotatable = false;
    };

    inline size_t rotation_stride(size_t bytes)
    {
        return bytes > SIZE_MAX - 255 ? SIZE_MAX : (bytes + 255) / 256 * 256;
    }

    // C participates even at beta zero. An aliased C/D uses one span covering
    // both tensors. Small read-only bias and scale vectors stay cache resident;
    // gradient bias is an output and participates with D and E.
    inline GemmRotationSpans
        gemm_rotation_spans(GemmRotationSpans spans, bool c_equals_d, bool gradient_bias)
    {
        if(c_equals_d)
        {
            spans.d = std::max(spans.c, spans.d);
            spans.c = 0;
        }
        if(!gradient_bias)
            spans.bias = 0;
        return spans;
    }

    inline GemmRotationLayout gemm_rotation_layout(const GemmRotationTensor& a,
                                                   const GemmRotationTensor& b,
                                                   const GemmRotationTensor& c,
                                                   const GemmRotationTensor& d,
                                                   const GemmRotationTensor& e,
                                                   const GemmRotationTensor& bias,
                                                   bool                      c_equals_d,
                                                   bool                      gradient)
    {
        auto span = [](const GemmRotationTensor& t, bool* expanded = nullptr) {
            return tensor_span_bytes(
                t.element_bytes, t.rows, t.cols, t.ld, t.batches, t.batch_stride, expanded);
        };
        bool expandedA = false, expandedB = false, expandedC = false, expandedE = false;
        GemmRotationSpans spans{span(a, &expandedA),
                                span(b, &expandedB),
                                span(c, &expandedC),
                                span(d),
                                span(e, &expandedE),
                                span(bias)};
        const bool        usable = spans.d != 0 && (!e.element_bytes || spans.e != 0)
                            && (!gradient || !bias.element_bytes || spans.bias != 0)
                            && (!c_equals_d || (spans.c != 0 && !expandedC))
                            && !(gradient && expandedE);
        const bool rotatable = spans.a != 0 && spans.b != 0 && !expandedA && !expandedB
                               && !expandedC && !(gradient && expandedE) && !a.swizzled
                               && !b.swizzled;
        return {gemm_rotation_spans(spans, c_equals_d, gradient), usable, rotatable};
    }

    inline size_t gemm_rotation_bytes(const GemmRotationSpans& spans)
    {
        size_t bytes = 0;
        for(size_t span : {spans.a, spans.b, spans.c, spans.d, spans.e, spans.bias})
        {
            const size_t stride = rotation_stride(span);
            if(bytes > SIZE_MAX - stride)
                return SIZE_MAX;
            bytes += stride;
        }
        return bytes;
    }

    inline RotatingBufferPlan plan_gemm_rotation(const GemmRotationSpans& spans,
                                                 bool                     rotatable,
                                                 bool                     adaptive,
                                                 int32_t                  max_iters,
                                                 int32_t                  cold_iters,
                                                 int32_t                  hot_iters,
                                                 int64_t                  target_bytes)
    {
        const auto bytes = gemm_rotation_bytes(spans);
        return compute_rotating_buffer_plan(
            adaptive,
            max_iters,
            cold_iters,
            hot_iters,
            rotatable && bytes != SIZE_MAX ? target_bytes : 0,
            static_cast<int64_t>(std::min<size_t>(bytes, INT64_MAX)));
    }
}
