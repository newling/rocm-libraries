/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/

#include "UserDrivenTuningParser.hpp"

#include <gtest/gtest.h>

namespace
{
    class TuningCacheMap : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            TensileLite::OverrideMap::getMap().resetForTest();
        }

        void TearDown() override
        {
            TensileLite::OverrideMap::getMap().resetForTest();
        }
    };

    TensileLite::TunedEntry entry(int index, bool complete)
    {
        TensileLite::TunedEntry value;
        value.solutionIndex = index;
        value.kernelName    = "kernel_" + std::to_string(index);
        value.solutionName  = "solution_" + std::to_string(index);
        value.complete      = complete;
        return value;
    }

    TEST_F(TuningCacheMap, CurrentEntriesPreferCompleteThenNewest)
    {
        TensileLite::ProblemOverride key;
        auto&                        map = TensileLite::OverrideMap::getMap();

        map.add(key, entry(1, false));
        map.add(key, entry(2, true));
        map.add(key, entry(3, false));
        map.add(key, entry(4, true));

        const auto found = map.find(key);
        ASSERT_EQ(found.size(), 4u);
        EXPECT_EQ(found[0].solutionIndex, 4);
        EXPECT_EQ(found[1].solutionIndex, 2);
        EXPECT_EQ(found[2].solutionIndex, 3);
        EXPECT_EQ(found[3].solutionIndex, 1);
    }

    TEST_F(TuningCacheMap, LegacyEntriesKeepFileOrder)
    {
        TensileLite::ProblemOverride key;
        auto&                        map = TensileLite::OverrideMap::getMap();

        map.addLegacyIfAbsent(key, entry(1, true));
        map.addLegacyIfAbsent(key, entry(2, true));

        const auto found = map.findLegacy(key);
        ASSERT_EQ(found.size(), 2u);
        EXPECT_EQ(found[0].solutionIndex, 1);
        EXPECT_EQ(found[1].solutionIndex, 2);
    }

    TEST_F(TuningCacheMap, SelectionFieldsArePartOfCurrentKey)
    {
        TensileLite::ProblemOverride key;
        key.operationA            = 1;
        key.colStrideE            = 64;
        key.batchStrideE          = 4096;
        key.uniformSummationOrder = 1;

        auto& map = TensileLite::OverrideMap::getMap();
        map.add(key, entry(1, true));

        auto conjugate       = key;
        conjugate.operationA = 2;
        EXPECT_TRUE(map.find(conjugate).empty());

        auto differentLde = key;
        differentLde.colStrideE++;
        EXPECT_TRUE(map.find(differentLde).empty());

        auto differentStrideE = key;
        differentStrideE.batchStrideE++;
        EXPECT_TRUE(map.find(differentStrideE).empty());

        auto differentSummation                  = key;
        differentSummation.uniformSummationOrder = 0;
        EXPECT_TRUE(map.find(differentSummation).empty());
    }
} // namespace
