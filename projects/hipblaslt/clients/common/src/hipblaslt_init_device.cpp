// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipblaslt/host_numerics/hipblaslt_init.hpp>
#include "hipblaslt_test.hpp"

#include <hip/hip_runtime.h>

#include <cstring>
#include <optional>
#include <stdexcept>

namespace
{
    struct IntegerExactPatternState
    {
        IntegerExactPattern pattern    = IntegerExactPattern::standard;
        size_t              K          = 0;
        bool                a_k_is_row = false;
    };
    IntegerExactPatternState& integer_exact_pattern_state()
    {
        static thread_local IntegerExactPatternState state;
        return state;
    }
}

bool parse_integer_exact_pattern(const char* name, IntegerExactPattern& pattern)
{
    if(!name || !*name || !strcmp(name, "standard"))
        pattern = IntegerExactPattern::standard;
    else if(!strcmp(name, "ternary"))
        pattern = IntegerExactPattern::ternary;
    else if(!strcmp(name, "sparse_k"))
        pattern = IntegerExactPattern::sparse_k;
    else
        return false;
    return true;
}

void set_integer_exact_pattern_state(IntegerExactPattern pattern, size_t K, bool a_k_is_row)
{
    integer_exact_pattern_state() = {pattern, K, a_k_is_row};
}

namespace
{
    hipblaslt::host_numerics::MatrixRole matrixRole(ABC_dims role)
    {
        using hipblaslt::host_numerics::MatrixRole;

        switch(role)
        {
        case ABC_dims::A:
            return MatrixRole::A;
        case ABC_dims::B:
            return MatrixRole::B;
        case ABC_dims::C:
            return MatrixRole::C;
        }
        throw std::invalid_argument("Unsupported hipBLASLt matrix role.");
    }
} // namespace

void hipblaslt_init_device(
    ABC_dims                                                   abc,
    hipblaslt_initialization                                   init,
    bool                                                       is_nan,
    void*                                                      destination,
    size_t                                                     rows,
    size_t                                                     columns,
    size_t                                                     leadingDimension,
    hipDataType                                                type,
    size_t                                                     batchStride,
    size_t                                                     batchCount,
    bool                                                       positiveOnly,
    std::optional<hipblaslt::host_numerics::OneSpecialValue> oneSpecialValue)
{
    using hipblaslt::host_numerics::MatrixInitialization;
    using roc::host_numerics::Tensor;

    MatrixInitialization initialization;
    initialization.role             = matrixRole(abc);
    initialization.initialization   = init;
    initialization.forceNaN         = is_nan;
    initialization.type             = type;
    initialization.rows             = rows;
    initialization.columns          = columns;
    initialization.leadingDimension = leadingDimension;
    initialization.batchStride      = batchStride;
    initialization.batchCount       = batchCount;
    initialization.oneSpecialValue  = oneSpecialValue;
    initialization.positiveOnly     = positiveOnly;

    Tensor     matrix  = hipblaslt::host_numerics::generateMatrix(initialization);
    const auto storage = matrix.rawEncodedBackingStorage();
    if(!storage.empty())
    {
        if(destination == nullptr)
            throw std::invalid_argument("hipBLASLt device initialization destination is null.");
        CHECK_HIP_ERROR(
            hipMemcpy(destination, storage.data(), storage.size(), hipMemcpyHostToDevice));
    }
}
