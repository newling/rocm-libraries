// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

namespace roc::hostnumerics {
enum class ScalarCategory : uint8_t {
    Boolean,
    SignedInteger,
    UnsignedInteger,
    FloatingPoint,
    Complex,
    Scale,
    Count,
};

// Dense table index for every supported scalar encoding. Count is one-past-last
// and exists only to size the metadata table. Metadata lookup rejects Count.
enum class ScalarType : uint16_t {
    Boolean,
    Int4,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float4E2M1,
    Float6E2M3,
    Float6E3M2,
    Float8E4M3,
    Float8E5M2,
    Float8E4M3Fnuz,
    Float8E5M2Fnuz,
    Float16,
    BFloat16,
    Float32,
    Float64,
    ComplexFloat32,
    ComplexFloat64,
    // Unsigned float: four exponent bits and three fraction bits in an eight-bit slot.
    // The top bit is ignored when decoding and zero when encoding; 0x7f is NaN.
    // Float8E4M3 instead uses the top bit as a sign bit.
    E4M3,
    // Unsigned float: five exponent bits and three fraction bits; 0xff is NaN.
    E5M3,
    // Unsigned powers of two: byte b represents 2^(b-127), except 0xff is NaN.
    E8M0,
    // E8M0 with byte 0x00 representing zero instead of 2^-127.
    E8M0Zero,
    Count,
};

enum class IntegerRounding : uint8_t {
    TowardZero,
    NearestEven,
};

enum class IntegerOverflow : uint8_t {
    Reject,
    Saturate,
    // Reduce modulo 2^N, then interpret the N-bit result as two's complement when signed.
    ModuloWrap,
};

enum class BFloat16Rounding : uint8_t {
    NearestEven,
    Truncate,
};

struct ScalarConversionOptions {
    // Option-bearing conversions reject overflow by default. No-options
    // Tensor and conversion APIs use the documented deterministic implicit policy.
    IntegerRounding integerRounding = IntegerRounding::TowardZero;
    IntegerOverflow integerOverflow = IntegerOverflow::Reject;
    BFloat16Rounding bfloat16Rounding = BFloat16Rounding::NearestEven;
};

struct ScalarTypeInfo {
    std::string_view name;
    ScalarCategory category;
    // Bits per element in tensor storage: 4 for Int4, 6 for Float6, 8 for Boolean
    // and E4M3, and 64 for ComplexFloat32 (two 32-bit components).
    uint16_t storageBits;
    // Exponent and fraction widths are per real component; zero for integers and Boolean.
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

// Reject Count and unnamed values produced by casts from integers.
inline constexpr bool isConcreteScalarType(ScalarType type) {
    return static_cast<size_t>(type) < scalarTypeCount;
}

// Exponent and fraction widths are per real component. NaN/Inf columns describe support.
// clang-format off
inline constexpr std::array<ScalarTypeInfo, scalarTypeCount> scalarTypeInfos{{
    // Name          Category                          Bits Exp Frac Bias NaN    Inf
    {"bool",         ScalarCategory::Boolean,            8,  0,   0,   0, false, false},
    {"i4",           ScalarCategory::SignedInteger,      4,  0,   0,   0, false, false},
    {"i8",           ScalarCategory::SignedInteger,      8,  0,   0,   0, false, false},
    {"i16",          ScalarCategory::SignedInteger,     16,  0,   0,   0, false, false},
    {"i32",          ScalarCategory::SignedInteger,     32,  0,   0,   0, false, false},
    {"i64",          ScalarCategory::SignedInteger,     64,  0,   0,   0, false, false},
    {"u8",           ScalarCategory::UnsignedInteger,    8,  0,   0,   0, false, false},
    {"u16",          ScalarCategory::UnsignedInteger,   16,  0,   0,   0, false, false},
    {"u32",          ScalarCategory::UnsignedInteger,   32,  0,   0,   0, false, false},
    {"u64",          ScalarCategory::UnsignedInteger,   64,  0,   0,   0, false, false},
    {"f4e2m1",       ScalarCategory::FloatingPoint,      4,  2,   1,   1, false, false},
    {"f6e2m3",       ScalarCategory::FloatingPoint,      6,  2,   3,   1, false, false},
    {"f6e3m2",       ScalarCategory::FloatingPoint,      6,  3,   2,   3, false, false},
    {"f8e4m3",       ScalarCategory::FloatingPoint,      8,  4,   3,   7, true,  false},
    {"f8e5m2",       ScalarCategory::FloatingPoint,      8,  5,   2,  15, true,  true },
    {"f8e4m3fnuz",   ScalarCategory::FloatingPoint,      8,  4,   3,   8, true,  false},
    {"f8e5m2fnuz",   ScalarCategory::FloatingPoint,      8,  5,   2,  16, true,  false},
    {"f16",          ScalarCategory::FloatingPoint,     16,  5,  10,  15, true,  true },
    {"bf16",         ScalarCategory::FloatingPoint,     16,  8,   7, 127, true,  true },
    {"f32",          ScalarCategory::FloatingPoint,     32,  8,  23, 127, true,  true },
    {"f64",          ScalarCategory::FloatingPoint,     64, 11,  52, 1023, true,  true },
    {"c64",          ScalarCategory::Complex,           64,  8,  23, 127, true,  true },
    {"c128",         ScalarCategory::Complex,          128, 11,  52, 1023, true,  true },
    {"e4m3",         ScalarCategory::Scale,              8,  4,   3,   7, true,  false},
    {"e5m3",         ScalarCategory::Scale,              8,  5,   3,  15, true,  false},
    {"e8m0",         ScalarCategory::Scale,              8,  8,   0, 127, true,  false},
    {"e8m0_zero",    ScalarCategory::Scale,              8,  8,   0, 127, true,  false},
}};
// clang-format on

// Invalid identifiers throw std::invalid_argument, including in release builds.
inline constexpr const ScalarTypeInfo& scalarTypeInfo(ScalarType type) {
    if (!isConcreteScalarType(type)) throw std::invalid_argument("Invalid ScalarType.");
    return scalarTypeInfos[static_cast<size_t>(type)];
}

inline constexpr std::string_view scalarTypeName(ScalarType type) {
    return scalarTypeInfo(type).name;
}

inline constexpr size_t scalarElementGroupSize(ScalarType type) {
    const size_t bits = scalarTypeInfo(type).storageBits;
    return 8 / std::gcd(bits, size_t{8});
}

template <typename T>
struct NativeScalarType;

template <>
struct NativeScalarType<bool> {
    static constexpr ScalarType value = ScalarType::Boolean;
};
template <>
struct NativeScalarType<uint8_t> {
    static constexpr ScalarType value = ScalarType::UInt8;
};
template <>
struct NativeScalarType<int8_t> {
    static constexpr ScalarType value = ScalarType::Int8;
};
template <>
struct NativeScalarType<uint16_t> {
    static constexpr ScalarType value = ScalarType::UInt16;
};
template <>
struct NativeScalarType<int16_t> {
    static constexpr ScalarType value = ScalarType::Int16;
};
template <>
struct NativeScalarType<uint32_t> {
    static constexpr ScalarType value = ScalarType::UInt32;
};
template <>
struct NativeScalarType<int32_t> {
    static constexpr ScalarType value = ScalarType::Int32;
};
template <>
struct NativeScalarType<uint64_t> {
    static constexpr ScalarType value = ScalarType::UInt64;
};
template <>
struct NativeScalarType<int64_t> {
    static constexpr ScalarType value = ScalarType::Int64;
};
template <>
struct NativeScalarType<float> {
    static constexpr ScalarType value = ScalarType::Float32;
};
template <>
struct NativeScalarType<double> {
    static constexpr ScalarType value = ScalarType::Float64;
};
template <>
struct NativeScalarType<std::complex<float>> {
    static constexpr ScalarType value = ScalarType::ComplexFloat32;
};
template <>
struct NativeScalarType<std::complex<double>> {
    static constexpr ScalarType value = ScalarType::ComplexFloat64;
};

template <typename T>
inline constexpr ScalarType nativeScalarType = NativeScalarType<std::remove_cvref_t<T>>::value;

template <typename T>
concept NativeScalar = requires { NativeScalarType<std::remove_cvref_t<T>>::value; };

namespace detail {
template <ScalarType TypeValue, typename StorageType>
struct ScalarTag {
    using Storage = StorageType;
    static constexpr ScalarType type = TypeValue;
    static constexpr uint16_t storageBits = scalarTypeInfo(TypeValue).storageBits;
};

using BooleanTag = ScalarTag<ScalarType::Boolean, uint8_t>;
using UInt8Tag = ScalarTag<ScalarType::UInt8, uint8_t>;
using Int8Tag = ScalarTag<ScalarType::Int8, int8_t>;
using UInt16Tag = ScalarTag<ScalarType::UInt16, uint16_t>;
using Int16Tag = ScalarTag<ScalarType::Int16, int16_t>;
using UInt32Tag = ScalarTag<ScalarType::UInt32, uint32_t>;
using Int32Tag = ScalarTag<ScalarType::Int32, int32_t>;
using UInt64Tag = ScalarTag<ScalarType::UInt64, uint64_t>;
using Int64Tag = ScalarTag<ScalarType::Int64, int64_t>;
using Float16Tag = ScalarTag<ScalarType::Float16, uint16_t>;
using BFloat16Tag = ScalarTag<ScalarType::BFloat16, uint16_t>;
using Float32Tag = ScalarTag<ScalarType::Float32, float>;
using Float64Tag = ScalarTag<ScalarType::Float64, double>;
using ComplexFloat32Tag = ScalarTag<ScalarType::ComplexFloat32, std::complex<float>>;
using ComplexFloat64Tag = ScalarTag<ScalarType::ComplexFloat64, std::complex<double>>;
using Float8E4M3Tag = ScalarTag<ScalarType::Float8E4M3, uint8_t>;
using Float8E5M2Tag = ScalarTag<ScalarType::Float8E5M2, uint8_t>;
using Float8E4M3FnuzTag = ScalarTag<ScalarType::Float8E4M3Fnuz, uint8_t>;
using Float8E5M2FnuzTag = ScalarTag<ScalarType::Float8E5M2Fnuz, uint8_t>;
using Float6E2M3Tag = ScalarTag<ScalarType::Float6E2M3, void>;
using Float6E3M2Tag = ScalarTag<ScalarType::Float6E3M2, void>;
using Float4E2M1Tag = ScalarTag<ScalarType::Float4E2M1, void>;
using Int4Tag = ScalarTag<ScalarType::Int4, void>;
using E8M0Tag = ScalarTag<ScalarType::E8M0, uint8_t>;
using E8M0ZeroTag = ScalarTag<ScalarType::E8M0Zero, uint8_t>;
using E5M3Tag = ScalarTag<ScalarType::E5M3, uint8_t>;
using E4M3Tag = ScalarTag<ScalarType::E4M3, uint8_t>;
}  // namespace detail

// Runtime dispatch maps every concrete ScalarType to a compile-time tag carrying its enum value,
// encoded width, and native storage type when one C++ object represents one scalar.
template <typename Visitor, typename... Args>
decltype(auto) visitScalarType(ScalarType type, Visitor&& visitor, Args&&... args) {
    switch (type) {
        case ScalarType::Boolean:
            return std::forward<Visitor>(visitor).template operator()<detail::BooleanTag>(
                std::forward<Args>(args)...);
        case ScalarType::UInt8:
            return std::forward<Visitor>(visitor).template operator()<detail::UInt8Tag>(
                std::forward<Args>(args)...);
        case ScalarType::Int8:
            return std::forward<Visitor>(visitor).template operator()<detail::Int8Tag>(
                std::forward<Args>(args)...);
        case ScalarType::UInt16:
            return std::forward<Visitor>(visitor).template operator()<detail::UInt16Tag>(
                std::forward<Args>(args)...);
        case ScalarType::Int16:
            return std::forward<Visitor>(visitor).template operator()<detail::Int16Tag>(
                std::forward<Args>(args)...);
        case ScalarType::UInt32:
            return std::forward<Visitor>(visitor).template operator()<detail::UInt32Tag>(
                std::forward<Args>(args)...);
        case ScalarType::Int32:
            return std::forward<Visitor>(visitor).template operator()<detail::Int32Tag>(
                std::forward<Args>(args)...);
        case ScalarType::UInt64:
            return std::forward<Visitor>(visitor).template operator()<detail::UInt64Tag>(
                std::forward<Args>(args)...);
        case ScalarType::Int64:
            return std::forward<Visitor>(visitor).template operator()<detail::Int64Tag>(
                std::forward<Args>(args)...);
        case ScalarType::Float16:
            return std::forward<Visitor>(visitor).template operator()<detail::Float16Tag>(
                std::forward<Args>(args)...);
        case ScalarType::BFloat16:
            return std::forward<Visitor>(visitor).template operator()<detail::BFloat16Tag>(
                std::forward<Args>(args)...);
        case ScalarType::Float32:
            return std::forward<Visitor>(visitor).template operator()<detail::Float32Tag>(
                std::forward<Args>(args)...);
        case ScalarType::Float64:
            return std::forward<Visitor>(visitor).template operator()<detail::Float64Tag>(
                std::forward<Args>(args)...);
        case ScalarType::ComplexFloat32:
            return std::forward<Visitor>(visitor).template operator()<detail::ComplexFloat32Tag>(
                std::forward<Args>(args)...);
        case ScalarType::ComplexFloat64:
            return std::forward<Visitor>(visitor).template operator()<detail::ComplexFloat64Tag>(
                std::forward<Args>(args)...);
        case ScalarType::Float8E4M3:
            return std::forward<Visitor>(visitor).template operator()<detail::Float8E4M3Tag>(
                std::forward<Args>(args)...);
        case ScalarType::Float8E5M2:
            return std::forward<Visitor>(visitor).template operator()<detail::Float8E5M2Tag>(
                std::forward<Args>(args)...);
        case ScalarType::Float8E4M3Fnuz:
            return std::forward<Visitor>(visitor).template operator()<detail::Float8E4M3FnuzTag>(
                std::forward<Args>(args)...);
        case ScalarType::Float8E5M2Fnuz:
            return std::forward<Visitor>(visitor).template operator()<detail::Float8E5M2FnuzTag>(
                std::forward<Args>(args)...);
        case ScalarType::Float6E2M3:
            return std::forward<Visitor>(visitor).template operator()<detail::Float6E2M3Tag>(
                std::forward<Args>(args)...);
        case ScalarType::Float6E3M2:
            return std::forward<Visitor>(visitor).template operator()<detail::Float6E3M2Tag>(
                std::forward<Args>(args)...);
        case ScalarType::Float4E2M1:
            return std::forward<Visitor>(visitor).template operator()<detail::Float4E2M1Tag>(
                std::forward<Args>(args)...);
        case ScalarType::Int4:
            return std::forward<Visitor>(visitor).template operator()<detail::Int4Tag>(
                std::forward<Args>(args)...);
        case ScalarType::E8M0:
            return std::forward<Visitor>(visitor).template operator()<detail::E8M0Tag>(
                std::forward<Args>(args)...);
        case ScalarType::E8M0Zero:
            return std::forward<Visitor>(visitor).template operator()<detail::E8M0ZeroTag>(
                std::forward<Args>(args)...);
        case ScalarType::E5M3:
            return std::forward<Visitor>(visitor).template operator()<detail::E5M3Tag>(
                std::forward<Args>(args)...);
        case ScalarType::E4M3:
            return std::forward<Visitor>(visitor).template operator()<detail::E4M3Tag>(
                std::forward<Args>(args)...);
        case ScalarType::Count:
            break;
    }
    throw std::invalid_argument("Invalid ScalarType.");
}
}  // namespace roc::hostnumerics
