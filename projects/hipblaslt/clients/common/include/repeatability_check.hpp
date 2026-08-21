/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace hipblaslt_bench
{
    enum class RepeatabilityCheckMode
    {
        exact,
        nonfinite,
        exact_and_nonfinite,
    };

    inline const char* repeatability_mode_name(RepeatabilityCheckMode mode)
    {
        switch(mode)
        {
        case RepeatabilityCheckMode::exact:
            return "exact";
        case RepeatabilityCheckMode::nonfinite:
            return "nonfinite";
        case RepeatabilityCheckMode::exact_and_nonfinite:
            return "both";
        }
        return "unknown";
    }

    struct RepeatabilityCheckConfig
    {
        bool                   enabled                  = false;
        RepeatabilityCheckMode mode                     = RepeatabilityCheckMode::exact;
        uint64_t               scan_every               = 1;
        uint64_t               mutate_call              = 0;
        size_t                 mutate_element           = 0;
        bool                   expect_mismatch          = false;
        uint64_t               mutate_nonfinite_call    = 0;
        size_t                 mutate_nonfinite_element = 0;
        bool                   expect_nonfinite         = false;

        bool exact_enabled() const
        {
            return mode == RepeatabilityCheckMode::exact
                   || mode == RepeatabilityCheckMode::exact_and_nonfinite;
        }

        bool nonfinite_enabled() const
        {
            return mode == RepeatabilityCheckMode::nonfinite
                   || mode == RepeatabilityCheckMode::exact_and_nonfinite;
        }
    };

    struct RepeatabilityMismatch
    {
        unsigned long long call_ordinal = 0;
        unsigned long long byte_offset  = 0;
        unsigned int       expected     = 0;
        unsigned int       actual       = 0;
        unsigned long long nonfinite_call_ordinal = 0;
        unsigned long long nonfinite_element      = 0;
        unsigned int       nonfinite_bits         = 0;
    };

    inline bool repeatability_env_bool(const char* name, bool default_value = false)
    {
        const char* value = std::getenv(name);
        if(!value)
            return default_value;
        if(std::string(value) == "0")
            return false;
        if(std::string(value) == "1")
            return true;
        throw std::invalid_argument(std::string(name) + " must be 0 or 1");
    }

    template <typename T>
    inline T repeatability_env_unsigned(const char* name, T default_value = 0)
    {
        const char* value = std::getenv(name);
        if(!value)
            return default_value;

        std::string text(value);
        if(text.empty() || text.front() == '-')
            throw std::invalid_argument(std::string(name) + " must be an unsigned integer");
        size_t      parsed = 0;
        unsigned long long result;
        try
        {
            result = std::stoull(text, &parsed, 10);
        }
        catch(const std::exception&)
        {
            throw std::invalid_argument(std::string(name) + " must be an unsigned integer");
        }
        if(parsed != text.size() || result > std::numeric_limits<T>::max())
            throw std::invalid_argument(std::string(name) + " must be an unsigned integer");
        return static_cast<T>(result);
    }

    inline RepeatabilityCheckConfig repeatability_check_config_from_environment()
    {
        RepeatabilityCheckConfig config;
        config.enabled
            = repeatability_env_bool("HIPBLASLT_BENCH_REPEATABILITY_CHECK", false);
        if(const char* mode = std::getenv("HIPBLASLT_BENCH_REPEATABILITY_MODE"))
        {
            std::string text(mode);
            if(text == "exact")
                config.mode = RepeatabilityCheckMode::exact;
            else if(text == "nonfinite")
                config.mode = RepeatabilityCheckMode::nonfinite;
            else if(text == "both")
                config.mode = RepeatabilityCheckMode::exact_and_nonfinite;
            else
                throw std::invalid_argument(
                    "HIPBLASLT_BENCH_REPEATABILITY_MODE must be exact, nonfinite, or both");
        }
        config.scan_every = repeatability_env_unsigned<uint64_t>(
            "HIPBLASLT_BENCH_REPEATABILITY_SCAN_EVERY", 1);
        config.mutate_call
            = repeatability_env_unsigned<uint64_t>("HIPBLASLT_BENCH_REPEATABILITY_MUTATE_CALL");
        config.mutate_element
            = repeatability_env_unsigned<size_t>("HIPBLASLT_BENCH_REPEATABILITY_MUTATE_ELEMENT");
        config.expect_mismatch = repeatability_env_bool(
            "HIPBLASLT_BENCH_REPEATABILITY_EXPECT_MISMATCH", false);
        config.mutate_nonfinite_call = repeatability_env_unsigned<uint64_t>(
            "HIPBLASLT_BENCH_REPEATABILITY_MUTATE_NONFINITE_CALL");
        config.mutate_nonfinite_element = repeatability_env_unsigned<size_t>(
            "HIPBLASLT_BENCH_REPEATABILITY_MUTATE_NONFINITE_ELEMENT");
        config.expect_nonfinite = repeatability_env_bool(
            "HIPBLASLT_BENCH_REPEATABILITY_EXPECT_NONFINITE", false);

        if(!config.enabled
           && (std::getenv("HIPBLASLT_BENCH_REPEATABILITY_MODE") != nullptr
               || std::getenv("HIPBLASLT_BENCH_REPEATABILITY_SCAN_EVERY") != nullptr
               || config.mutate_call != 0 || config.mutate_element != 0 || config.expect_mismatch
               || config.mutate_nonfinite_call != 0
               || config.mutate_nonfinite_element != 0 || config.expect_nonfinite))
            throw std::invalid_argument(
                "HIPBLASLT_BENCH_REPEATABILITY_CHECK=1 is required for repeatability controls");
        if(config.scan_every == 0)
            throw std::invalid_argument(
                "HIPBLASLT_BENCH_REPEATABILITY_SCAN_EVERY must exceed 0");
        if(!config.exact_enabled()
           && (config.mutate_call != 0 || config.mutate_element != 0 || config.expect_mismatch))
            throw std::invalid_argument(
                "exact mismatch controls require exact or both repeatability mode");
        if(!config.nonfinite_enabled()
           && (config.mutate_nonfinite_call != 0 || config.mutate_nonfinite_element != 0
               || config.expect_nonfinite))
            throw std::invalid_argument(
                "non-finite controls require nonfinite or both repeatability mode");
        if(config.mutate_call == 1)
            throw std::invalid_argument(
                "HIPBLASLT_BENCH_REPEATABILITY_MUTATE_CALL must exceed 1 because call 1 "
                "establishes the golden output");
        if(config.mutate_element != 0 && config.mutate_call == 0)
            throw std::invalid_argument(
                "HIPBLASLT_BENCH_REPEATABILITY_MUTATE_ELEMENT requires a mutation call");
        if(config.mutate_nonfinite_element != 0 && config.mutate_nonfinite_call == 0)
            throw std::invalid_argument(
                "HIPBLASLT_BENCH_REPEATABILITY_MUTATE_NONFINITE_ELEMENT requires a mutation "
                "call");
        auto is_observed = [&](uint64_t call) {
            return call == 1 || call % config.scan_every == 0;
        };
        if(config.mutate_call != 0 && !is_observed(config.mutate_call))
            throw std::invalid_argument(
                "exact mutation call must be selected by HIPBLASLT_BENCH_REPEATABILITY_SCAN_EVERY");
        if(config.mutate_nonfinite_call != 0 && !is_observed(config.mutate_nonfinite_call))
            throw std::invalid_argument(
                "non-finite mutation call must be selected by "
                "HIPBLASLT_BENCH_REPEATABILITY_SCAN_EVERY");
        return config;
    }

    static __global__ void repeatability_mutate_byte(unsigned char* output, size_t byte_offset)
    {
        if(blockIdx.x == 0 && threadIdx.x == 0)
            output[byte_offset] ^= 1;
    }

    static __global__ void repeatability_mutate_bf16_nonfinite(uint16_t* output,
                                                                size_t    element_offset)
    {
        if(blockIdx.x == 0 && threadIdx.x == 0)
            output[element_offset] = 0x7fc1;
    }

    static __global__ void repeatability_scan_bf16_nonfinite(
        const uint16_t*       output,
        size_t                elements,
        unsigned long long    call_ordinal,
        RepeatabilityMismatch* mismatch)
    {
        size_t index  = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
        for(; index < elements; index += stride)
        {
            if(mismatch->nonfinite_call_ordinal != 0)
                return;
            uint16_t bits = output[index];
            if((bits & 0x7f80) != 0x7f80)
                continue;
            if(atomicCAS(&mismatch->nonfinite_call_ordinal, 0ULL, call_ordinal) == 0ULL)
            {
                mismatch->nonfinite_element = index;
                mismatch->nonfinite_bits    = bits;
            }
            return;
        }
    }

    static __global__ void repeatability_compare_bytes(const unsigned char* golden,
                                                       const unsigned char* output,
                                                       size_t               bytes,
                                                       unsigned long long   call_ordinal,
                                                       RepeatabilityMismatch* mismatch)
    {
        size_t index  = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
        for(; index < bytes; index += stride)
        {
            if(mismatch->call_ordinal != 0)
                return;
            if(golden[index] == output[index])
                continue;
            if(atomicCAS(&mismatch->call_ordinal, 0ULL, call_ordinal) == 0ULL)
            {
                mismatch->byte_offset = index;
                mismatch->expected    = golden[index];
                mismatch->actual      = output[index];
            }
            return;
        }
    }

    class RepeatabilityChecker
    {
    public:
        RepeatabilityChecker(RepeatabilityCheckConfig config,
                             size_t                   element_count,
                             size_t                   element_bytes,
                             bool                     output_is_bf16)
            : m_config(config)
            , m_element_count(element_count)
            , m_element_bytes(element_bytes)
            , m_output_is_bf16(output_is_bf16)
        {
            if(!m_config.enabled)
                return;
            if(m_element_count == 0 || m_element_bytes == 0
               || m_element_count > std::numeric_limits<size_t>::max() / m_element_bytes)
                throw std::invalid_argument("invalid repeatability-check output size");
            m_bytes = m_element_count * m_element_bytes;
            if(m_config.mutate_call != 0 && m_config.mutate_element >= m_element_count)
                throw std::invalid_argument(
                    "HIPBLASLT_BENCH_REPEATABILITY_MUTATE_ELEMENT is outside D");
            if(m_config.nonfinite_enabled() && !m_output_is_bf16)
                throw std::invalid_argument(
                    "non-finite repeatability mode currently requires BF16 D");
            if(m_config.mutate_nonfinite_call != 0
               && m_config.mutate_nonfinite_element >= m_element_count)
                throw std::invalid_argument(
                    "HIPBLASLT_BENCH_REPEATABILITY_MUTATE_NONFINITE_ELEMENT is outside D");

            hipError_t status = hipSuccess;
            if(m_config.exact_enabled())
            {
                status = hipMalloc(&m_golden, m_bytes);
                if(status != hipSuccess)
                    throw std::runtime_error(
                        std::string("repeatability golden allocation failed: ")
                        + hipGetErrorString(status));
            }
            status = hipMalloc(&m_mismatch, sizeof(RepeatabilityMismatch));
            if(status != hipSuccess)
            {
                (void)hipFree(m_golden);
                m_golden = nullptr;
                throw std::runtime_error(std::string("repeatability record allocation failed: ")
                                         + hipGetErrorString(status));
            }
        }

        ~RepeatabilityChecker()
        {
            if(m_mismatch)
                (void)hipFree(m_mismatch);
            if(m_golden)
                (void)hipFree(m_golden);
        }

        RepeatabilityChecker(const RepeatabilityChecker&)            = delete;
        RepeatabilityChecker& operator=(const RepeatabilityChecker&) = delete;

        bool enabled() const
        {
            return m_config.enabled;
        }

        const RepeatabilityCheckConfig& config() const
        {
            return m_config;
        }

        size_t element_bytes() const
        {
            return m_element_bytes;
        }

        bool observes(uint64_t call_ordinal) const
        {
            return call_ordinal == 1 || call_ordinal % m_config.scan_every == 0;
        }

        uint64_t observation_count(uint64_t total_calls) const
        {
            if(total_calls == 0)
                return 0;
            uint64_t count = total_calls / m_config.scan_every;
            if(m_config.scan_every != 1)
                ++count;
            return count;
        }

        hipError_t reset(hipStream_t stream)
        {
            if(!enabled())
                return hipSuccess;
            return hipMemsetAsync(m_mismatch, 0, sizeof(RepeatabilityMismatch), stream);
        }

        hipError_t observe(void* output, uint64_t call_ordinal, hipStream_t stream)
        {
            if(!enabled())
                return hipSuccess;
            if(m_config.exact_enabled() && call_ordinal == 1)
            {
                if(hipError_t status = hipMemcpyAsync(
                       m_golden, output, m_bytes, hipMemcpyDeviceToDevice, stream);
                   status != hipSuccess)
                    return status;
            }

            if(m_config.mutate_call == call_ordinal)
            {
                size_t byte_offset = m_config.mutate_element * m_element_bytes;
                hipLaunchKernelGGL(repeatability_mutate_byte,
                                   dim3(1),
                                   dim3(1),
                                   0,
                                   stream,
                                   static_cast<unsigned char*>(output),
                                   byte_offset);
                if(hipError_t status = hipPeekAtLastError(); status != hipSuccess)
                    return status;
            }

            if(m_config.mutate_nonfinite_call == call_ordinal)
            {
                hipLaunchKernelGGL(repeatability_mutate_bf16_nonfinite,
                                   dim3(1),
                                   dim3(1),
                                   0,
                                   stream,
                                   static_cast<uint16_t*>(output),
                                   m_config.mutate_nonfinite_element);
                if(hipError_t status = hipPeekAtLastError(); status != hipSuccess)
                    return status;
            }

            if(!observes(call_ordinal))
                return hipSuccess;

            constexpr unsigned int threads = 256;
            constexpr size_t max_blocks = 4096;
            if(m_config.nonfinite_enabled())
            {
                size_t blocks_size
                    = std::min(max_blocks, (m_element_count + threads - 1) / threads);
                hipLaunchKernelGGL(repeatability_scan_bf16_nonfinite,
                                   dim3(static_cast<unsigned int>(blocks_size)),
                                   dim3(threads),
                                   0,
                                   stream,
                                   static_cast<const uint16_t*>(output),
                                   m_element_count,
                                   static_cast<unsigned long long>(call_ordinal),
                                   m_mismatch);
                if(hipError_t status = hipPeekAtLastError(); status != hipSuccess)
                    return status;
            }
            if(m_config.exact_enabled() && call_ordinal > 1)
            {
                size_t blocks_size = std::min(max_blocks, (m_bytes + threads - 1) / threads);
                hipLaunchKernelGGL(repeatability_compare_bytes,
                                   dim3(static_cast<unsigned int>(blocks_size)),
                                   dim3(threads),
                                   0,
                                   stream,
                                   static_cast<const unsigned char*>(m_golden),
                                   static_cast<const unsigned char*>(output),
                                   m_bytes,
                                   static_cast<unsigned long long>(call_ordinal),
                                   m_mismatch);
            }
            return hipPeekAtLastError();
        }

        hipError_t finish(RepeatabilityMismatch& result, hipStream_t stream)
        {
            if(!enabled())
                return hipSuccess;
            hipError_t status = hipMemcpyAsync(&result,
                                               m_mismatch,
                                               sizeof(RepeatabilityMismatch),
                                               hipMemcpyDeviceToHost,
                                               stream);
            return status == hipSuccess ? hipStreamSynchronize(stream) : status;
        }

    private:
        RepeatabilityCheckConfig m_config;
        size_t                   m_element_count = 0;
        size_t                   m_element_bytes = 0;
        size_t                   m_bytes         = 0;
        bool                     m_output_is_bf16 = false;
        void*                    m_golden        = nullptr;
        RepeatabilityMismatch*   m_mismatch      = nullptr;
    };
}
