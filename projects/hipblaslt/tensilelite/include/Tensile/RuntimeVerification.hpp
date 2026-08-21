// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <tensilelitehost/export.h>

// tensilelite-host is normally a hidden-visibility static archive embedded in
// libhipblaslt. Its generated export macro is therefore empty, but these hook
// functions must remain callable by an external verification harness so it
// updates the same runtime state used by ContractionSolution.
#if defined(__GNUC__) && !defined(_WIN32)
#define TENSILELITE_RUNTIME_VERIFICATION_EXPORT __attribute__((visibility("default")))
#else
#define TENSILELITE_RUNTIME_VERIFICATION_EXPORT TENSILELITEHOST_EXPORT
#endif

namespace TensileLite
{
    struct RuntimeVerificationRecord
    {
        std::string runId;
        std::string launchId;
        std::string roctxRange;
        uint64_t    ordinal = 0;

        int         solutionIndex = -1;
        std::string solutionName;
        std::string kernelName;

        int      streamK            = 0;
        int      streamKAtomic      = 0;
        int      streamKForceDPOnly = 0;
        uint32_t macroTileX         = 0;
        uint32_t macroTileY         = 0;
        uint32_t depthU             = 0;
        int      occupancy          = 0;
        int32_t  metadataGSU        = 0;
        int      globalAccumulation = 0;

        int         requestedTileScheduling = 0;
        int         forceModeEnvironment    = -1;
        int         effectiveSmCountTarget  = 0;
        int         hardwareComputeUnits    = 0;
        int         hardwareNumXcd          = 0;
        int         bakedQueueCount         = 0;
        int         dynamicGridMode         = 0;
        int         maxCUs                  = 0;
        int         fixedGrid               = 0;
        int         gridMultiplier          = 1;
        int         fullTiles               = 1;
        int         overrideTiles           = -1;
        int         overrideSplit           = -1;
        std::string effectiveMode;
        std::string modeReason;
        std::string gridSource;
        std::string reductionInitial;
        std::string reductionFinal;

        uint64_t tiles             = 0;
        uint64_t gridWorkgroups    = 0;
        uint64_t workspaceProvided = 0;
        uint64_t workspaceIdeal    = 0;
        uint64_t dynamicQueueBytes = 0;
        bool     workspaceFallback = false;
        int32_t  requestedGSU      = 0;
        uint32_t autoGSU           = 0;
        uint32_t effectiveGSU      = 0;

        uint32_t workgroupX   = 0;
        uint32_t workgroupY   = 0;
        uint32_t workgroupZ   = 0;
        uint64_t gridThreadsX = 0;
        uint64_t gridThreadsY = 0;
        uint64_t gridThreadsZ = 0;

        std::vector<std::string> dispatchKernelNames;
    };

    TENSILELITE_RUNTIME_VERIFICATION_EXPORT bool runtimeVerificationEnabled();

    // An explicit context lets a harness use the same unique launch/range
    // identity for its ROCTx range and this post-resolution record.
    TENSILELITE_RUNTIME_VERIFICATION_EXPORT void
        setRuntimeVerificationContext(std::string launchId,
                                      std::string roctxRange,
                                      uint64_t    ordinal = 0);
    TENSILELITE_RUNTIME_VERIFICATION_EXPORT void clearRuntimeVerificationContext();

    TENSILELITE_RUNTIME_VERIFICATION_EXPORT std::string
        serializeRuntimeVerificationRecord(RuntimeVerificationRecord const& record);
    TENSILELITE_RUNTIME_VERIFICATION_EXPORT void
        emitRuntimeVerificationRecord(RuntimeVerificationRecord record);
}

// Stable, optional entry points for external diagnostic harnesses. Historical
// payloads do not provide these symbols, so callers must resolve them with
// dlsym and require them only for an explicitly instrumented run.
extern "C"
{
    TENSILELITE_RUNTIME_VERIFICATION_EXPORT void
        hipblaslt_tensilelite_runtime_verification_set_context(char const* launchId,
                                                               char const* roctxRange,
                                                               uint64_t    ordinal);
    TENSILELITE_RUNTIME_VERIFICATION_EXPORT void
        hipblaslt_tensilelite_runtime_verification_clear_context();
}

#undef TENSILELITE_RUNTIME_VERIFICATION_EXPORT
