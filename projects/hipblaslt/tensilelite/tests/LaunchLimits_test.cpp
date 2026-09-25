// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Tensile/ContractionTaskPredicates.hpp>

#include <limits>
#include <sstream>

using namespace TensileLite;

namespace
{
    class LaunchLimitsTest : public ::testing::Test
    {
    protected:
        ContractionSolution solution;
        AMDGPU              gpu{AMDGPU::Processor::gfx950, 256, "launch-limits-test"};
        Predicates::Contraction::LaunchLimits predicate;

        void SetUp() override
        {
            solution.sizeMapping.streamK               = 3;
            solution.sizeMapping.workGroupSize         = {256, 1, 1};
            solution.sizeMapping.threadTile            = {1, 1, 1};
            solution.sizeMapping.macroTile             = {16, 16, 1};
            solution.sizeMapping.depthU                = 32;
            solution.sizeMapping.CUOccupancy           = 1;
            solution.sizeMapping.globalSplitU          = 1;
            solution.sizeMapping.workspaceSizePerElemC = 4;
            solution.sizeMapping.clusterDim            = {1, 1, 1};
            solution.sizeMapping.workGroupMapping      = 1;
            solution.sizeMapping.workGroupMappingXCC   = 1;
            solution.internalArgsSupport.version       = 1;
            gpu.skDynamicGrid                          = 0;
        }

        ContractionProblemGemm problemForTiles(size_t tilesX, size_t tilesY = 1, size_t batch = 1)
        {
            size_t m = tilesX * 16;
            return ContractionProblemGemm::GEMM(
                false, false, m, tilesY * 16, 48, m, 48, m, 0.0, false, batch);
        }

        void
            expectGrid(ContractionProblemGemm const& problem, TensileLite::dim3 expected, bool fits)
        {
            TensileLite::dim3 threads, groups;
            solution.calculateLaunchGrid(threads, groups, problem, gpu);
            EXPECT_EQ(groups, expected);
            Task task(gpu, problem, solution);
            EXPECT_EQ(predicate(task), fits);
            std::ostringstream diagnostic;
            EXPECT_EQ(predicate.debugEval(task, diagnostic), fits);
        }
    };
}

TEST_F(LaunchLimitsTest, WorkspaceFallbackUsesActualWorkgroupSize)
{
    // At 64/128 threads, grids above the 256-thread limit remain selectable.
    // At 512 threads, the limit is lower.
    for(size_t threads : {64, 128, 256, 512})
    {
        SCOPED_TRACE(threads);
        solution.sizeMapping.workGroupSize = {threads, 1, 1};
        size_t limit                       = std::numeric_limits<uint32_t>::max() / threads;
        auto   safe                        = problemForTiles(limit);
        safe.setWorkspaceSize(0);
        expectGrid(safe, {limit, 1, 1}, true);

        // Odd tile count: the CU-sized grid needs partial-tile workspace.
        auto   large     = problemForTiles(limit + 2);
        size_t workspace = solution.partialTileSize(256);
        large.setWorkspaceSize(workspace);
        expectGrid(large, {256, 1, 1}, true);
        large.setWorkspaceSize(workspace - 1);
        expectGrid(large, {limit + 2, 1, 1}, false);

        // The workspace filter accepts the zero-workspace fallback. LaunchLimits
        // must separately reject its unrepresentable grid.
        EXPECT_EQ(solution.requiredWorkspaceSize(large, gpu), 0u);
        EXPECT_TRUE(Predicates::Contraction::WorkspaceCheck{}(Task(gpu, large, solution)));
    }
}

TEST_F(LaunchLimitsTest, DynamicAndHybridUseWorkspaceFallback)
{
    for(int mode : {4, 5})
    {
        SCOPED_TRACE(mode);
        solution.sizeMapping.streamK = mode;
        auto problem                 = problemForTiles((size_t{1} << 24) + 1);
        problem.setParams().setStreamKTileSchedulingMode(1);
        problem.setWorkspaceSize(solution.partialTileSize(256));
        expectGrid(problem, {256, 1, 1}, true);
        problem.setWorkspaceSize(0);
        expectGrid(problem, {(size_t{1} << 24) + 1, 1, 1}, false);
    }
}

TEST_F(LaunchLimitsTest, SelectionMatchesDispatchedWorkspaceFallback)
{
    auto              problem = problemForTiles((size_t{1} << 24) + 1);
    float             storage = 0;
    ContractionInputs inputs;
    inputs.a = inputs.b = inputs.c = inputs.d = inputs.ws = &storage;
    inputs.alpha                                          = 1.0f;
    inputs.beta                                           = 0.0f;

    // solve() only builds host-side invocations here; nothing is allocated or
    // dispatched to a GPU, and the placeholder matrix pointers are not read.
    for(size_t workspace : {solution.partialTileSize(256), size_t{0}})
    {
        SCOPED_TRACE(workspace);
        problem.setWorkspaceSize(workspace);
        auto calls = solution.solve(problem, inputs, gpu);
        ASSERT_EQ(calls.size(), 1u);
        size_t expectedGroups = workspace ? 256 : (size_t{1} << 24) + 1;
        EXPECT_EQ(calls[0].numWorkGroups, TensileLite::dim3(expectedGroups, 1, 1));
        EXPECT_EQ(calls[0].numWorkItems, TensileLite::dim3(expectedGroups * 256, 1, 1));
        expectGrid(problem, calls[0].numWorkGroups, workspace != 0);
    }
}

TEST_F(LaunchLimitsTest, AtomicFixedGridIsCheckedEvenForFewTiles)
{
    // Atomic kernels do not use the flags buffer, whose capacity otherwise
    // clamps an oversized fixed grid before it reaches the launch.
    solution.sizeMapping.streamKAtomic         = 1;
    solution.sizeMapping.workspaceSizePerElemC = 0;
    gpu.skFixedGrid                            = size_t{1} << 24;
    auto problem                               = problemForTiles(100);
    problem.setWorkspaceSize(solution.partialTileSize(gpu.skFixedGrid));
    expectGrid(problem, {size_t{1} << 24, 1, 1}, false);
}

TEST_F(LaunchLimitsTest, SmallFallbackAndBatchedFallback)
{
    auto small = problemForTiles(257);
    small.setWorkspaceSize(0);
    expectGrid(small, {257, 1, 1}, true);

    // Neither matrix alone exceeds the limit; the batch count makes the
    // one-workgroup-per-tile fallback too large.
    auto batched = problemForTiles((size_t{1} << 23) + 1, 1, 3);
    batched.setWorkspaceSize(0);
    expectGrid(batched, {3 * ((size_t{1} << 23) + 1), 1, 1}, false);
}

TEST_F(LaunchLimitsTest, NonStreamKUsesGsuAndPhysicalAxes)
{
    solution.sizeMapping.streamK = 0;
    size_t limit                 = std::numeric_limits<uint32_t>::max() / 256;
    auto   boundary              = problemForTiles(limit);
    expectGrid(boundary, {limit, 1, 1}, true);
    auto overflow = problemForTiles(limit + 1);
    expectGrid(overflow, {limit + 1, 1, 1}, false);

    auto split = problemForTiles((size_t{1} << 23));
    split.setParams().setGSU(2);
    expectGrid(split, {size_t{1} << 24, 1, 1}, false);

    auto twoDimensional                  = problemForTiles(limit, 2);
    solution.internalArgsSupport.version = 0;
    expectGrid(twoDimensional, {limit, 2, 1}, true);
    solution.internalArgsSupport.version = 1;
    expectGrid(twoDimensional, {limit * 2, 1, 1}, false);
}

TEST_F(LaunchLimitsTest, ClusterPaddingIsIncluded)
{
    solution.sizeMapping.streamKForceDPOnly = 1;
    solution.sizeMapping.clusterDim         = {2, 1, 1};
    size_t limit                            = std::numeric_limits<uint32_t>::max() / 256;
    auto   safe                             = problemForTiles(limit - 1, 2);
    expectGrid(safe, {limit - 1, 2, 1}, true);
    auto padded = problemForTiles(limit);
    expectGrid(padded, {limit + 1, 1, 1}, false);
}

TEST_F(LaunchLimitsTest, CustomGridUsesMetadataAndBatchExpansion)
{
    solution.customKernel.name          = "launch-limits-test";
    solution.customKernel.macrotile     = {16, 16, 32};
    solution.customKernel.threads       = {128, 2, 1};
    solution.customKernel.workspaceType = CustomWorkspaceType::StreamK;
    solution.customKernel.grid
        = {CustomGridSize::StreamKWithBatch, CustomGridSize::One, CustomGridSize::One};
    auto problem = problemForTiles(257, 1, 3);
    problem.setWorkspaceSize(0);
    // Preserves generateCustomCall's existing expansion of sk.grid by batch.
    expectGrid(problem, {257 * 3 * 3, 1, 1}, true);

    solution.sizeMapping.streamK        = 0;
    solution.customKernel.workspaceType = CustomWorkspaceType::None;
    solution.customKernel.grid
        = {CustomGridSize::TilesX, CustomGridSize::TilesY, CustomGridSize::Batch};
    auto large = problemForTiles((size_t{1} << 24), 2);
    expectGrid(large, {size_t{1} << 24, 2, 1}, true);
    solution.customKernel.args = {{CustomArgType::uint32, CustomArgSemantic::NumWorkGroups}};
    expectGrid(large, {size_t{1} << 25, 1, 1}, false);
}
