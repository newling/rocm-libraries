// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "UserDrivenTuningParser.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <iostream>

// Only the logging sink is stubbed; parsing, deduplication and matching are real.
std::ostream* get_logger_os()
{
    return &std::cerr;
}
uint32_t get_logger_layer_mode()
{
    return 0;
}
const char* rocblaslt_layer_mode2string(rocblaslt_layer_mode)
{
    return "";
}
std::string prefix(const char*, const char*)
{
    return {};
}

namespace
{
    using namespace TensileLite;
    const std::string fingerprint = "sha256-v1:" + std::string(64, 'a');
    const std::string changed     = "sha256-v1:" + std::string(64, 'b');

    TEST(SolutionFingerprint, OnlyKnownWellFormedIdentitiesCanMatch)
    {
        EXPECT_TRUE(solutionFingerprintMatches(fingerprint, fingerprint));
        EXPECT_FALSE(solutionFingerprintMatches(fingerprint, changed));
        EXPECT_FALSE(solutionFingerprintMatches(fingerprint, ""));
        for(const auto& invalid : {std::string{},
                                   std::string("sha256-v2:") + std::string(64, 'a'),
                                   std::string("sha256-v1:") + std::string(64, 'g'),
                                   fingerprint + "a",
                                   fingerprint.substr(1)})
        {
            EXPECT_FALSE(validSolutionFingerprint(invalid));
            EXPECT_FALSE(solutionFingerprintMatches(invalid, invalid));
        }
    }

    class TuningFingerprint : public ::testing::Test
    {
    protected:
        std::string     path;
        ProblemOverride key{false,
                            false,
                            rocisa::DataType::Half,
                            rocisa::DataType::Half,
                            rocisa::DataType::Float,
                            rocisa::DataType::Half,
                            1024,
                            512,
                            1024,
                            1};

        void SetUp() override
        {
            path = std::string("fingerprint-")
                   + ::testing::UnitTest::GetInstance()->current_test_info()->name() + ".csv";
            OverrideMap::getMap().resetForTest();
            std::ofstream(path) << "Git Version: different-build\n";
        }

        void TearDown() override
        {
            OverrideMap::getMap().resetForTest();
            std::remove(path.c_str());
        }

        void append(const std::optional<std::string>& value, bool name = true, bool torn = false)
        {
            std::ofstream file(path, std::ios::app);
            file << "transA,transB,batch_count,m,n,k,a_type,b_type,c_type,compute_type,solution_"
                    "index";
            if(name)
                file << ",kernel_name";
            if(value)
                file << ",solution_fingerprint";
            file << "\nN,N,1,1024,512,1024,f16_r,f16_r,f16_r,f32_r,123";
            if(name)
                file << ",same_kernel_name";
            if(value && !torn)
                file << ',' << *value;
            file << '\n';
        }

        std::vector<TunedEntry> read()
        {
            getContractionProblemsFromFile(path);
            return OverrideMap::getMap().find(key);
        }
    };

    TEST_F(TuningFingerprint, StaleFingerprintDoesNotHideLaterValidEntryAtSameIndex)
    {
        append(changed);
        append(fingerprint);
        append(fingerprint);
        auto entries = read();
        ASSERT_EQ(entries.size(), 2u);
        EXPECT_FALSE(solutionFingerprintMatches(*entries[0].fingerprint, fingerprint));
        EXPECT_TRUE(solutionFingerprintMatches(*entries[1].fingerprint, fingerprint));
    }

    TEST_F(TuningFingerprint, FingerprintWorksWithoutNamesOrMatchingBuildStamp)
    {
        append(fingerprint, false);
        auto entries = read();
        ASSERT_EQ(entries.size(), 1u);
        EXPECT_EQ(entries[0].fingerprint, fingerprint);
    }

    TEST_F(TuningFingerprint, EmptyUnknownAndTruncatedFingerprintsDoNotDowngrade)
    {
        append("");
        append("sha256-v2:" + std::string(64, 'a'));
        append(fingerprint, true, true);
        EXPECT_TRUE(read().empty());
    }

    TEST_F(TuningFingerprint, LegacyNamedRowsKeepTheirExistingBehavior)
    {
        append(std::nullopt);
        append(std::nullopt, false);
        auto entries = read();
        ASSERT_EQ(entries.size(), 1u);
        EXPECT_FALSE(entries[0].fingerprint);
        EXPECT_EQ(entries[0].kernelName, "same_kernel_name");
    }
}
