// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "Arch.h"
#include "Profiler.h"
#include "RunConfig.h"
#include "Util.h"
#include "Wave.h"
#include "Workgroup.h"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>

namespace raceemulator {

struct ParsedAsm;

/// Main entry point for the race emulator. Parses AMD GPU assembly, manages
/// kernel arguments and architecture state, and delegates execution to
/// Workgroup. Decorates exceptions with assembly context for diagnostics.
///
/// TODO(newling) handle case of multiple kernels in single assembly file.
class Emulator {

public:
  /// Construct from a pre-parsed assembly (legacy path for tests with .set
  /// symbols or simplified metadata format). Will be removed.
  Emulator(std::unique_ptr<ParsedAsm> parsedAsm, std::shared_ptr<Architecture> arch);

  /// Construct from a code object ELF. Internally disassembles the .text
  /// section and parses metadata from the kernel descriptor and msgpack
  /// .note section. Requires llvm-objdump on PATH or LLVM_BIN_DIR env var.
  /// Optionally accepts original source for diagnostic source mapping.
  Emulator(const uint8_t *codeObject, size_t size,
           std::shared_ptr<Architecture> arch,
           std::string_view originalSource = "");

  Emulator(const Emulator &other);
  Emulator &operator=(const Emulator &other);
  Emulator(Emulator &&other) noexcept;
  Emulator &operator=(Emulator &&other) = delete;
  ~Emulator();

  /// Return a formatted profiling report. Entries below minPercentage of
  /// wall-clock time are aggregated into an "other" row.
  std::string getProfileReport(double minPercentage = 1.0) const;

  /// Write a kernel argument into this emulator's kernarg segment.
  void addKernarg(int argNumber, const void *argValue);

  /// Write all kernel arguments at once from a contiguous buffer.
  void addAllKernargs(const void *args);

  /// Run the kernel on the given workgroups (sequentially). Each workgroup
  /// gets its own waves, LDS, and event state — fully independent.
  void run(const std::vector<Dim3d> &wgIds, Dim3d blockDim,
           const RunConfig &config = {});

  /// Convenience: run the kernel on a single workgroup.
  void run(Dim3d wgId, Dim3d blockDim, const RunConfig &config = {}) {
    run(std::vector<Dim3d>{wgId}, blockDim, config);
  }

  std::string getName() const;
  int getKernargSegmentSize() const;
  int getNumKernargs() const;
  int getKernargOffset(int argNumber) const;
  int getKernargSize(int argNumber) const;
  std::string getKernargValueKind(int argNumber) const;
  std::string getKernargAddressSpace(int argNumber) const;
  std::string getKernargName(int argNumber) const;

  const Architecture &getArch() const { return *arch; }

  void appendStr(std::ostream &) const;
  std::string str() const;

private:
  /// Extract structural metadata from parsed assembly into a WorkgroupConfig.
  WorkgroupConfig buildWorkgroupConfig(int nWaves,
                                       const RunConfig &config) const;

  /// Initialize per-wave runtime state (SGPRs, kernarg preload, thread IDs,
  /// PC) on an already-constructed Workgroup.
  void initializeWaveState(Workgroup &workgroup, Dim3d wgId, Dim3d blockDim,
                           int nWaves);
  void initKernargSegment();
  std::shared_ptr<Architecture> arch;
  std::unique_ptr<ParsedAsm> parsedAsm;
  std::vector<char> kernargSegment;
  std::vector<bool> kernargIsSet;
  Profiler emulatorProfiler;
  Profiler workgroupProfiler;
};

} // namespace raceemulator
