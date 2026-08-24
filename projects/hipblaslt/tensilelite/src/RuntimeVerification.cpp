// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <Tensile/RuntimeVerification.hpp>

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace TensileLite
{
    namespace
    {
        struct LaunchContext
        {
            std::string launchId;
            std::string roctxRange;
            uint64_t    ordinal = 0;
        };

        thread_local LaunchContext runtimeContext;
        std::atomic<uint64_t>      runtimeOrdinal{0};
        std::mutex                 runtimeFileMutex;
        std::set<std::string>      initializedRuntimePaths;

        std::string jsonString(std::string const& value)
        {
            std::ostringstream stream;
            stream << '"';
            for(unsigned char character : value)
            {
                switch(character)
                {
                case '"':
                    stream << "\\\"";
                    break;
                case '\\':
                    stream << "\\\\";
                    break;
                case '\b':
                    stream << "\\b";
                    break;
                case '\f':
                    stream << "\\f";
                    break;
                case '\n':
                    stream << "\\n";
                    break;
                case '\r':
                    stream << "\\r";
                    break;
                case '\t':
                    stream << "\\t";
                    break;
                default:
                    if(character < 0x20)
                    {
                        stream << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                               << static_cast<unsigned>(character) << std::dec;
                    }
                    else
                    {
                        stream << static_cast<char>(character);
                    }
                }
            }
            stream << '"';
            return stream.str();
        }

        char const* reductionOrUnknown(std::string const& value)
        {
            return value.empty() ? "unknown" : value.c_str();
        }
    }

    bool runtimeVerificationEnabled()
    {
        auto path = std::getenv("HIPBLASLT_RUNTIME_VERIFY_PATH");
        return path != nullptr && path[0] != '\0';
    }

    void setRuntimeVerificationContext(std::string launchId,
                                       std::string roctxRange,
                                       uint64_t    ordinal)
    {
        runtimeContext.launchId   = std::move(launchId);
        runtimeContext.roctxRange = std::move(roctxRange);
        runtimeContext.ordinal    = ordinal;
    }

    void clearRuntimeVerificationContext()
    {
        runtimeContext = {};
    }

    std::string serializeRuntimeVerificationRecord(RuntimeVerificationRecord const& record)
    {
        std::ostringstream stream;
        stream << "{\"schema_version\":1"
               << ",\"record_type\":\"tensilelite_runtime\""
               << ",\"run_id\":" << jsonString(record.runId)
               << ",\"launch_id\":" << jsonString(record.launchId)
               << ",\"ordinal\":" << record.ordinal
               << ",\"roctx_range\":" << jsonString(record.roctxRange) << ",\"algorithm\":{"
               << "\"backend\":\"tensilelite\""
               << ",\"selected_solution_index\":" << record.solutionIndex
               << ",\"metadata_solution_name\":" << jsonString(record.solutionName)
               << ",\"metadata_kernel_symbol\":" << jsonString(record.kernelName)
               << ",\"expected_workgroup\":{\"x\":" << record.workgroupX
               << ",\"y\":" << record.workgroupY << ",\"z\":" << record.workgroupZ << "}}"
               << ",\"streamk\":{\"metadata\":{"
               << "\"stream_k\":" << record.streamK
               << ",\"stream_k_atomic\":" << record.streamKAtomic
               << ",\"stream_k_force_dp_only\":" << record.streamKForceDPOnly << ",\"macro_tile\":["
               << record.macroTileX << ',' << record.macroTileY << ',' << record.depthU << ']'
               << ",\"depth_u\":" << record.depthU << ",\"occupancy\":" << record.occupancy
               << ",\"global_split_u\":" << record.metadataGSU
               << ",\"global_accumulation\":" << record.globalAccumulation << "}"
               << ",\"launch\":{"
               << "\"requested_tile_scheduling\":" << record.requestedTileScheduling
               << ",\"force_mode_env\":" << record.forceModeEnvironment
               << ",\"effective_mode\":" << jsonString(record.effectiveMode)
               << ",\"mode_reason\":" << jsonString(record.modeReason)
               << ",\"sm_count_target\":{\"effective\":" << record.effectiveSmCountTarget << "}"
               << ",\"hardware\":{\"compute_units\":" << record.hardwareComputeUnits
               << ",\"num_xcd\":" << record.hardwareNumXcd
               << ",\"baked_queue_count\":" << record.bakedQueueCount << "}"
               << ",\"environment\":{\"dynamic_grid_mode\":" << record.dynamicGridMode
               << ",\"max_cus\":" << record.maxCUs << ",\"fixed_grid\":" << record.fixedGrid
               << ",\"grid_multiplier\":" << record.gridMultiplier
               << ",\"full_tiles\":" << record.fullTiles
               << ",\"override_tiles\":" << record.overrideTiles
               << ",\"override_split\":" << record.overrideSplit << "}"
               << ",\"tiles\":" << record.tiles
               << ",\"grid_source\":" << jsonString(record.gridSource)
               << ",\"grid_workgroups\":" << record.gridWorkgroups << ",\"reduction_initial\":"
               << jsonString(reductionOrUnknown(record.reductionInitial))
               << ",\"reduction_final\":" << jsonString(reductionOrUnknown(record.reductionFinal))
               << ",\"workspace\":{\"provided_bytes\":" << record.workspaceProvided
               << ",\"ideal_bytes\":" << record.workspaceIdeal
               << ",\"dynamic_queue_bytes\":" << record.dynamicQueueBytes
               << ",\"fallback_to_tree_dp\":" << (record.workspaceFallback ? "true" : "false")
               << "}"
               << ",\"gsu\":{\"metadata\":" << record.metadataGSU
               << ",\"requested\":" << record.requestedGSU << ",\"auto\":" << record.autoGSU
               << ",\"effective\":" << record.effectiveGSU << "}}}"
               << ",\"expected_grid_threads\":{\"x\":" << record.gridThreadsX
               << ",\"y\":" << record.gridThreadsY << ",\"z\":" << record.gridThreadsZ << "}"
               << ",\"dispatch_kernel_names\":[";
        for(size_t index = 0; index < record.dispatchKernelNames.size(); ++index)
        {
            if(index > 0)
                stream << ',';
            stream << jsonString(record.dispatchKernelNames[index]);
        }
        stream << "]}";
        return stream.str();
    }

    void emitRuntimeVerificationRecord(RuntimeVerificationRecord record)
    {
        auto path = std::getenv("HIPBLASLT_RUNTIME_VERIFY_PATH");
        if(path == nullptr || path[0] == '\0')
            return;
        auto runId = std::getenv("HIPBLASLT_RUNTIME_VERIFY_RUN_ID");
        if(runId == nullptr || runId[0] == '\0')
            throw std::runtime_error(
                "HIPBLASLT_RUNTIME_VERIFY_RUN_ID is required when runtime verification is enabled");

        record.runId   = runId;
        record.ordinal = runtimeContext.ordinal != 0
                             ? runtimeContext.ordinal
                             : runtimeOrdinal.fetch_add(1, std::memory_order_relaxed) + 1;
        if(!runtimeContext.launchId.empty())
            record.launchId = runtimeContext.launchId;
        if(!runtimeContext.roctxRange.empty())
            record.roctxRange = runtimeContext.roctxRange;
        if(record.launchId.empty())
        {
            auto               prefix = std::getenv("HIPBLASLT_RUNTIME_VERIFY_PREFIX");
            std::ostringstream generated;
            generated << (prefix && prefix[0] ? prefix : "launch") << '-' << std::setw(6)
                      << std::setfill('0') << record.ordinal;
            record.launchId = generated.str();
        }
        if(record.roctxRange.empty())
            record.roctxRange = record.launchId;

        auto                        serialized = serializeRuntimeVerificationRecord(record);
        std::lock_guard<std::mutex> lock(runtimeFileMutex);
        bool                        firstWrite = initializedRuntimePaths.insert(path).second;
        if(firstWrite)
        {
            std::ifstream existing(path, std::ios::binary | std::ios::ate);
            if(existing && existing.tellg() > 0)
            {
                initializedRuntimePaths.erase(path);
                throw std::runtime_error(
                    std::string("refusing to append to nonempty HIPBLASLT_RUNTIME_VERIFY_PATH: ")
                    + path);
            }
        }
        std::ofstream output(path, std::ios::out | std::ios::app);
        if(!output)
        {
            if(firstWrite)
                initializedRuntimePaths.erase(path);
            throw std::runtime_error(std::string("cannot open HIPBLASLT_RUNTIME_VERIFY_PATH: ")
                                     + path);
        }
        output << serialized << '\n';
        output.flush();
        if(!output)
        {
            if(firstWrite)
                initializedRuntimePaths.erase(path);
            throw std::runtime_error(std::string("cannot write HIPBLASLT_RUNTIME_VERIFY_PATH: ")
                                     + path);
        }
    }
}
