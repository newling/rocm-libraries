// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace roc::hostnumerics {
// Scalar types define the runtime numeric vocabulary shared by every HostNumerics component.
// This header describes type identity and encoding metadata.
enum class ScalarCategory : uint8_t {
    Boolean,
    SignedInteger,
    UnsignedInteger,
    FloatingPoint,
    Complex,
    Scale,
};

// Dense table index for every supported scalar encoding. Count is one-past-last
// and exists only to size the metadata table. Metadata lookup rejects Count.
enum class ScalarType : uint16_t {
    Boolean,
    UInt8,
    Int8,
    UInt16,
    Int16,
    UInt32,
    Int32,
    UInt64,
    Int64,
    Float16,
    BFloat16,
    Float32,
    Float64,
    ComplexFloat32,
    ComplexFloat64,
    Float8E4M3,
    Float8E5M2,
    Float8E4M3Fnuz,
    Float8E5M2Fnuz,
    Float6E2M3,
    Float6E3M2,
    Float4E2M1,
    Int4,
    E8M0,
    E8M0Zero,
    E5M3,
    E4M3,
    Count,
};

struct ScalarTypeInfo {
    // Short human-readable spelling used in diagnostics and bindings.
    std::string_view name;
    // Broad conversion and comparison behavior of the scalar.
    ScalarCategory category;
    // Encoded bits occupied by one logical scalar, including both complex components and any
    // packed or reserved payload bits.
    uint16_t storageBits;
    // Encoded exponent bits per real component; zero for non-floating categories.
    uint8_t exponentBits;
    // Explicitly stored fraction bits per real component, excluding any implicit leading bit.
    uint8_t mantissaBits;
    // Bias subtracted from an encoded exponent; zero when exponentBits is zero.
    int16_t exponentBias;
    // Whether the encoding has at least one representation for NaN.
    bool supportsNaN;
    // Whether the encoding has representations for positive and negative infinity.
    bool supportsInfinity;

    bool isPacked() const {
        return storageBits % 8 != 0;
    }
};

inline constexpr size_t scalarTypeCount = static_cast<size_t>(ScalarType::Count);

inline constexpr bool isConcreteScalarType(ScalarType type) {
    return static_cast<size_t>(type) < scalarTypeCount;
}

// Columns follow ScalarTypeInfo's fields: name, category, storage bits, exponent bits,
// mantissa bits, exponent bias, NaN support, and infinity support.
inline constexpr std::array<ScalarTypeInfo, scalarTypeCount> scalarTypeInfos{{
    {"bool", ScalarCategory::Boolean, 8, 0, 0, 0, false, false},
    {"u8", ScalarCategory::UnsignedInteger, 8, 0, 0, 0, false, false},
    {"i8", ScalarCategory::SignedInteger, 8, 0, 0, 0, false, false},
    {"u16", ScalarCategory::UnsignedInteger, 16, 0, 0, 0, false, false},
    {"i16", ScalarCategory::SignedInteger, 16, 0, 0, 0, false, false},
    {"u32", ScalarCategory::UnsignedInteger, 32, 0, 0, 0, false, false},
    {"i32", ScalarCategory::SignedInteger, 32, 0, 0, 0, false, false},
    {"u64", ScalarCategory::UnsignedInteger, 64, 0, 0, 0, false, false},
    {"i64", ScalarCategory::SignedInteger, 64, 0, 0, 0, false, false},
    {"f16", ScalarCategory::FloatingPoint, 16, 5, 10, 15, true, true},
    {"bf16", ScalarCategory::FloatingPoint, 16, 8, 7, 127, true, true},
    {"f32", ScalarCategory::FloatingPoint, 32, 8, 23, 127, true, true},
    {"f64", ScalarCategory::FloatingPoint, 64, 11, 52, 1023, true, true},
    {"c64", ScalarCategory::Complex, 64, 8, 23, 127, true, true},
    {"c128", ScalarCategory::Complex, 128, 11, 52, 1023, true, true},
    {"f8e4m3", ScalarCategory::FloatingPoint, 8, 4, 3, 7, true, false},
    {"f8e5m2", ScalarCategory::FloatingPoint, 8, 5, 2, 15, true, true},
    {"f8e4m3fnuz", ScalarCategory::FloatingPoint, 8, 4, 3, 8, true, false},
    {"f8e5m2fnuz", ScalarCategory::FloatingPoint, 8, 5, 2, 16, true, false},
    {"f6e2m3", ScalarCategory::FloatingPoint, 6, 2, 3, 1, false, false},
    {"f6e3m2", ScalarCategory::FloatingPoint, 6, 3, 2, 3, false, false},
    {"f4e2m1", ScalarCategory::FloatingPoint, 4, 2, 1, 1, false, false},
    {"i4", ScalarCategory::SignedInteger, 4, 0, 0, 0, false, false},
    {"e8m0", ScalarCategory::Scale, 8, 8, 0, 127, true, false},
    {"e8m0_zero", ScalarCategory::Scale, 8, 8, 0, 127, true, false},
    {"e5m3", ScalarCategory::Scale, 8, 5, 3, 15, true, false},
    {"e4m3", ScalarCategory::Scale, 8, 4, 3, 7, true, false},
}};

inline constexpr const ScalarTypeInfo& scalarTypeInfo(ScalarType type) {
    if (!isConcreteScalarType(type)) throw std::invalid_argument("Invalid ScalarType.");
    return scalarTypeInfos[static_cast<size_t>(type)];
}

inline constexpr std::string_view scalarTypeName(ScalarType type) {
    return scalarTypeInfo(type).name;
}
}  // namespace roc::hostnumerics
