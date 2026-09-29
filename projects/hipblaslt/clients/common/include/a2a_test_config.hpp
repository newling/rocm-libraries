// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Pure configuration shared by the fused-A2A rank harness and its host tests.
// This header deliberately has no HIP, SDMA, or process-launch dependencies.

#include <cstdint>
#include <cstring>

namespace hipblaslt_bench
{
    // One GPU per rank; the parent skips worlds larger than the visible device count.
    constexpr uint32_t kFusedA2AWorlds[] = {2, 3, 4, 5, 6, 7, 8};

    // Largest MacroTile0 the fused path admits; the other is 128.
    constexpr int64_t kFusedA2AMaxMacroTile0 = 256;

    // Every world moves the same per-rank shard and keeps the same local tail, so
    // only M grows with the world. Both are whole tiles for either admitted
    // MacroTile0, which makes FusedA2ATileDivisible hold for every world. At 2
    // ranks this is the 4096 x 256 x 1024 problem with a 2048 extent.
    constexpr int64_t kFusedA2AShardFeatures = 1024;
    constexpr int64_t kFusedA2ALocalFeatures = 2048;
    constexpr int64_t kFusedA2ATokens        = 256;
    constexpr int64_t kFusedA2AContractBound = 1024;

    struct FusedA2AShape
    {
        int64_t features;
        int64_t extent;
    };

    inline FusedA2AShape fused_a2a_shape_for_world(uint32_t world)
    {
        const int64_t extent = kFusedA2AShardFeatures * int64_t(world);
        return {extent + kFusedA2ALocalFeatures, extent};
    }

    // Expected to gain A2AGemm collective.
    enum class Collective
    {
        GemmA2A,
        Unknown,
    };

    constexpr const char* kCollectiveGemmA2A = "gemm-a2a";

    // An unset name selects GemmA2A, the only collective before A2A_COLLECTIVE.
    inline Collective parse_collective(const char* name)
    {
        if(name == nullptr || name[0] == '\0' || std::strcmp(name, kCollectiveGemmA2A) == 0)
            return Collective::GemmA2A;
        return Collective::Unknown;
    }

    inline const char* collective_name(Collective c)
    {
        switch(c)
        {
        case Collective::GemmA2A:
            return kCollectiveGemmA2A;
        case Collective::Unknown:
            break;
        }
        return "";
    }

} // namespace hipblaslt_bench
