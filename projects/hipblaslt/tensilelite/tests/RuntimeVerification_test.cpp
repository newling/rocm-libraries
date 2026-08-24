// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <Tensile/RuntimeVerification.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace TensileLite
{
    namespace
    {
        class ScopedEnvironment
        {
        public:
            ScopedEnvironment(char const* name, char const* value)
                : m_name(name)
            {
                auto previous = std::getenv(name);
                if(previous != nullptr)
                {
                    m_hadPrevious = true;
                    m_previous    = previous;
                }
                if(value != nullptr)
                    setenv(name, value, 1);
                else
                    unsetenv(name);
            }

            ~ScopedEnvironment()
            {
                if(m_hadPrevious)
                    setenv(m_name.c_str(), m_previous.c_str(), 1);
                else
                    unsetenv(m_name.c_str());
            }

        private:
            std::string m_name;
            std::string m_previous;
            bool        m_hadPrevious = false;
        };

        RuntimeVerificationRecord sampleRecord()
        {
            RuntimeVerificationRecord record;
            record.runId                   = "run";
            record.launchId                = "launch";
            record.roctxRange              = "range";
            record.ordinal                 = 7;
            record.solutionIndex           = 357191;
            record.solutionName            = "solution";
            record.kernelName              = "kernel";
            record.streamK                 = 5;
            record.macroTileX              = 256;
            record.macroTileY              = 256;
            record.depthU                  = 128;
            record.occupancy               = 1;
            record.requestedTileScheduling = 2;
            record.forceModeEnvironment    = -1;
            record.effectiveMode           = "sk4_dynamic";
            record.modeReason              = "api-auto-heuristic";
            record.gridSource              = "dynamic-sk4";
            record.reductionInitial        = "tree";
            record.reductionFinal          = "tree";
            record.tiles                   = 128;
            record.gridWorkgroups          = 64;
            record.workspaceProvided       = 4096;
            record.workspaceIdeal          = 2048;
            record.requestedGSU            = 0;
            record.autoGSU                 = 1;
            record.effectiveGSU            = 1;
            record.workgroupX              = 256;
            record.workgroupY              = 1;
            record.workgroupZ              = 1;
            record.gridThreadsX            = 16384;
            record.gridThreadsY            = 1;
            record.gridThreadsZ            = 1;
            record.dispatchKernelNames     = {"kernel", "conversion"};
            return record;
        }

        TEST(RuntimeVerification, SerializeContainsResolvedFieldsAndEscapes)
        {
            auto record       = sampleRecord();
            record.kernelName = "kernel\n\"quoted\"";
            auto json         = serializeRuntimeVerificationRecord(record);
            EXPECT_NE(json.find("\"record_type\":\"tensilelite_runtime\""), std::string::npos);
            EXPECT_NE(json.find("\"effective_mode\":\"sk4_dynamic\""), std::string::npos);
            EXPECT_NE(json.find("\"grid_workgroups\":64"), std::string::npos);
            EXPECT_NE(json.find("\"fallback_to_tree_dp\":false"), std::string::npos);
            EXPECT_NE(json.find("kernel\\n\\\"quoted\\\""), std::string::npos);
        }

        TEST(RuntimeVerification, DisabledEmitterDoesNotCreateFile)
        {
            ScopedEnvironment path("HIPBLASLT_RUNTIME_VERIFY_PATH", nullptr);
            EXPECT_FALSE(runtimeVerificationEnabled());
            EXPECT_NO_THROW(emitRuntimeVerificationRecord(sampleRecord()));
        }

        TEST(RuntimeVerification, ExplicitContextWritesOneJsonLine)
        {
            auto path = std::filesystem::temp_directory_path()
                        / "hipblaslt-runtime-verification-test.jsonl";
            std::filesystem::remove(path);
            ScopedEnvironment outputPath("HIPBLASLT_RUNTIME_VERIFY_PATH", path.c_str());
            ScopedEnvironment runId("HIPBLASLT_RUNTIME_VERIFY_RUN_ID", "unit-run");
            setRuntimeVerificationContext("unit-launch", "unit-range", 42);

            auto record = sampleRecord();
            record.launchId.clear();
            record.roctxRange.clear();
            emitRuntimeVerificationRecord(record);
            clearRuntimeVerificationContext();

            std::ifstream input(path);
            ASSERT_TRUE(input);
            std::stringstream buffer;
            buffer << input.rdbuf();
            auto text = buffer.str();
            EXPECT_NE(text.find("\"run_id\":\"unit-run\""), std::string::npos);
            EXPECT_NE(text.find("\"launch_id\":\"unit-launch\""), std::string::npos);
            EXPECT_NE(text.find("\"roctx_range\":\"unit-range\""), std::string::npos);
            EXPECT_NE(text.find("\"ordinal\":42"), std::string::npos);
            EXPECT_EQ(std::count(text.begin(), text.end(), '\n'), 1);
            std::filesystem::remove(path);
        }

        TEST(RuntimeVerification, EnabledWithoutRunIdFailsClosed)
        {
            auto path = std::filesystem::temp_directory_path()
                        / "hipblaslt-runtime-verification-missing-run.jsonl";
            std::filesystem::remove(path);
            ScopedEnvironment outputPath("HIPBLASLT_RUNTIME_VERIFY_PATH", path.c_str());
            ScopedEnvironment runId("HIPBLASLT_RUNTIME_VERIFY_RUN_ID", nullptr);
            EXPECT_THROW(emitRuntimeVerificationRecord(sampleRecord()), std::runtime_error);
            EXPECT_FALSE(std::filesystem::exists(path));
        }

        TEST(RuntimeVerification, UnwritableOutputFailsClosed)
        {
            auto path = std::filesystem::temp_directory_path()
                        / "hipblaslt-runtime-verification-missing-directory" / "record.jsonl";
            std::filesystem::remove_all(path.parent_path());
            ScopedEnvironment outputPath("HIPBLASLT_RUNTIME_VERIFY_PATH", path.c_str());
            ScopedEnvironment runId("HIPBLASLT_RUNTIME_VERIFY_RUN_ID", "unit-run");
            EXPECT_THROW(emitRuntimeVerificationRecord(sampleRecord()), std::runtime_error);
            EXPECT_FALSE(std::filesystem::exists(path));
        }

        TEST(RuntimeVerification, NonemptyStaleOutputFailsClosed)
        {
            auto path = std::filesystem::temp_directory_path()
                        / "hipblaslt-runtime-verification-stale.jsonl";
            {
                std::ofstream stale(path);
                stale << "{\"stale\":true}\n";
            }
            ScopedEnvironment outputPath("HIPBLASLT_RUNTIME_VERIFY_PATH", path.c_str());
            ScopedEnvironment runId("HIPBLASLT_RUNTIME_VERIFY_RUN_ID", "unit-run");
            EXPECT_THROW(emitRuntimeVerificationRecord(sampleRecord()), std::runtime_error);
            std::filesystem::remove(path);
        }

    }
}
