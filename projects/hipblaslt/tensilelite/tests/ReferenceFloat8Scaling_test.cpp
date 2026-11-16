// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Reference.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/DataTypes.hpp>

#include <array>
#include <string>

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    template <typename F8>
    void checkScaling(bool               halfA,
                      std::string const& mode,
                      bool               narrow     = true,
                      bool               unitScales = false)
    {
        constexpr auto f8      = TypeInfo<F8>::Enum;
        auto           problem = ContractionProblemGemm::GEMM_Strides(false,
                                                            false,
                                                            halfA ? rocisa::DataType::Half : f8,
                                                            halfA ? f8 : rocisa::DataType::Half,
                                                            f8,
                                                            f8,
                                                            2,
                                                            2,
                                                            1,
                                                            1,
                                                            2,
                                                            2,
                                                            1,
                                                            2,
                                                            2,
                                                            4,
                                                            2,
                                                            4,
                                                            0.0);
        problem.setComputeInputTypeA(narrow ? f8 : rocisa::DataType::Half);
        problem.setComputeInputTypeB(narrow ? f8 : rocisa::DataType::Half);
        problem.setAlphaType(rocisa::DataType::Float);
        problem.setBetaType(rocisa::DataType::Float);
        problem.setUseScaleAB(mode);
        problem.setScaleA(rocisa::DataType::Float, 2);
        problem.setScaleB(rocisa::DataType::Float, 2);

        std::array<Half, 2>  h = {Half(0.9f), Half(-0.9f)};
        std::array<F8, 2>    f = {F8(1.0f), F8(-1.0f)};
        std::array<F8, 4>    c = {}, d = {};
        std::array<float, 2> narrowedScale = {3.0f, -2.0f};
        std::array<float, 2> otherScale    = {1.0f, 0.5f};
        ContractionInputs    inputs(halfA ? static_cast<void const*>(h.data()) : f.data(),
                                 halfA ? static_cast<void const*>(f.data()) : h.data(),
                                 c.data(),
                                 d.data(),
                                 1.0f,
                                 0.0f);
        if(unitScales)
        {
            narrowedScale.fill(1.0f);
            otherScale.fill(1.0f);
        }
        inputs.scaleA = halfA ? narrowedScale.data() : otherScale.data();
        inputs.scaleB = halfA ? otherScale.data() : narrowedScale.data();
        SolveGemmCPU(problem, inputs, /*elementsToValidate=*/-1, /*tryFastPath=*/false);

        // Half(0.9) narrows to 0.875 in both F8 encodings. The two vectors
        // multiply the dot product afterward. Ties 2.625 and -1.3125 round
        // to even F8 values 2.5 and -1.25. These literals are independent of
        // the reference's conversion and scale-selection implementation.
        std::array<float, 4> expected;
        if(mode.empty() || unitScales)
            expected = {0.875f, -0.875f, -0.875f, 0.875f};
        else if(mode == "Scalar")
            // Scalar narrowing scales Half first: 0.9 * 3 -> 2.75.
            // Half MAC gives the same final F8 result for these inputs.
            expected = {2.75f, -2.75f, -2.75f, 2.75f};
        else if(narrow)
            expected = halfA ? std::array<float, 4>{2.5f, 1.75f, -1.25f, -0.875f}
                             : std::array<float, 4>{2.5f, -1.25f, 1.75f, -0.875f};
        else
            // With Half MAC operands, 0.9 * 3 is rounded only when writing D.
            expected = halfA ? std::array<float, 4>{2.75f, 1.75f, -1.375f, -0.875f}
                             : std::array<float, 4>{2.75f, -1.375f, 1.75f, -0.875f};
        for(size_t i = 0; i < d.size(); ++i)
            EXPECT_EQ(float(d[i]), expected[i]) << "D[" << i << "]";
    }
}

TEST(ReferenceFloat8Scaling, BothVectorsApplyAfterHalfToFloat8Narrowing)
{
    for(bool halfA : {true, false})
    {
        SCOPED_TRACE(halfA ? "Half A" : "Half B");
        {
            SCOPED_TRACE("OCP");
            checkScaling<Float8>(halfA, "Vector");
        }
        {
            SCOPED_TRACE("fnuz");
            checkScaling<Float8_fnuz>(halfA, "Vector");
        }
    }
}

TEST(ReferenceFloat8Scaling, PreservesScalarNoScaleAndHalfMacResults)
{
    for(bool halfA : {true, false})
        for(auto mode : {"", "Scalar", "Vector"})
            for(bool narrow : {true, false})
            {
                SCOPED_TRACE(::testing::Message()
                             << "HalfA=" << halfA << " " << mode << " narrow=" << narrow);
                {
                    SCOPED_TRACE("OCP");
                    checkScaling<Float8>(halfA, mode, narrow);
                }
                {
                    SCOPED_TRACE("fnuz");
                    checkScaling<Float8_fnuz>(halfA, mode, narrow);
                }
            }
}

TEST(ReferenceFloat8Scaling, UnitScaleBuffersPreserveNarrowedDotProduct)
{
    for(bool halfA : {true, false})
        for(auto mode : {"Scalar", "Vector"})
        {
            SCOPED_TRACE(::testing::Message() << "HalfA=" << halfA << " " << mode);
            {
                SCOPED_TRACE("OCP");
                checkScaling<Float8>(halfA, mode, true, true);
            }
            {
                SCOPED_TRACE("fnuz");
                checkScaling<Float8_fnuz>(halfA, mode, true, true);
            }
        }
}
