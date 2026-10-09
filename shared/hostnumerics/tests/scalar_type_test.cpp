// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <initializer_list>
#include <roc/hostnumerics/scalar_type.hpp>
#include <stdexcept>

int main() {
    using namespace roc::hostnumerics;
    const auto require = [](bool condition, const char* message) {
        if (!condition) throw std::runtime_error(message);
    };

    // IEEE binary16 has a five-bit exponent, ten fraction bits, and bias 15.
    const auto& half = scalarTypeInfo(ScalarType::Float16);
    require(half.name == "f16" && half.category == ScalarCategory::FloatingPoint &&
                half.storageBits == 16 && half.exponentBits == 5 && half.mantissaBits == 10 &&
                half.exponentBias == 15 && half.supportsNaN && half.supportsInfinity &&
                !half.isPacked(),
            "Incorrect binary16 metadata.");
    const auto& complex = scalarTypeInfo(ScalarType::ComplexFloat32);
    require(complex.category == ScalarCategory::Complex && complex.storageBits == 64 &&
                complex.exponentBits == 8 && complex.mantissaBits == 23,
            "Complex metadata must count both components' storage.");
    const auto& packed = scalarTypeInfo(ScalarType::Float6E3M2);
    require(packed.storageBits == 6 && packed.isPacked() && !packed.supportsNaN &&
                !packed.supportsInfinity,
            "Incorrect finite six-bit encoding metadata.");
    require(scalarTypeInfo(ScalarType::Int4).category == ScalarCategory::SignedInteger &&
                scalarTypeInfo(ScalarType::E8M0).category == ScalarCategory::Scale,
            "Incorrect integer or scale category.");

    for (size_t index = 0; index < scalarTypeCount; ++index)
        require(!scalarTypeName(static_cast<ScalarType>(index)).empty(),
                "A concrete scalar type has no metadata.");
    for (ScalarType invalid : {ScalarType::Count, static_cast<ScalarType>(0xffff)}) {
        require(!isConcreteScalarType(invalid), "Invalid scalar type classified as concrete.");
        bool rejected = false;
        try {
            (void)scalarTypeInfo(invalid);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "Metadata lookup accepted an invalid scalar type.");
    }
}
