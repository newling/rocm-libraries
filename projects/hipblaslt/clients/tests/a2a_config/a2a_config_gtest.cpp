// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "a2a_test_config.hpp"

#include <gtest/gtest.h>

using hipblaslt_bench::Collective;

TEST(FusedA2ACollective_smoke, NamesRoundTrip)
{
    EXPECT_EQ(hipblaslt_bench::parse_collective(nullptr), Collective::GemmA2A);
    EXPECT_EQ(hipblaslt_bench::parse_collective(""), Collective::GemmA2A);
    EXPECT_EQ(
        hipblaslt_bench::parse_collective(hipblaslt_bench::collective_name(Collective::GemmA2A)),
        Collective::GemmA2A);
    EXPECT_EQ(hipblaslt_bench::parse_collective("bias"), Collective::Unknown);
}

// One rule has to give a legal fused-A2A problem at every world, with the same
// shard per rank and the same local tail.
TEST(FusedA2AWorldShape_smoke, EveryWorldFrom2To8IsTileDivisible)
{
    for(uint32_t world : hipblaslt_bench::kFusedA2AWorlds)
    {
        const auto    shape = hipblaslt_bench::fused_a2a_shape_for_world(world);
        const int64_t tile  = hipblaslt_bench::kFusedA2AMaxMacroTile0;
        EXPECT_EQ(shape.extent % (tile * int64_t(world)), 0) << "world " << world;
        EXPECT_EQ(shape.features % tile, 0) << "world " << world;
        EXPECT_EQ(shape.extent / int64_t(world), hipblaslt_bench::kFusedA2AShardFeatures)
            << "world " << world;
        EXPECT_EQ(shape.features - shape.extent, hipblaslt_bench::kFusedA2ALocalFeatures)
            << "world " << world;
    }

    const auto two = hipblaslt_bench::fused_a2a_shape_for_world(2);
    EXPECT_EQ(two.features, 4096);
    EXPECT_EQ(two.extent, 2048);
}
