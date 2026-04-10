// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "Arch.h"
#include "Parsing.h"
#include "Profiler.h"
#include "RunConfig.h"
#include "Workgroup.h"
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

namespace raceemulator {

struct DisassembledKernel;

/// Emulates an AMD GPU kernel from a code object ELF. Manages kernel arguments,
/// architecture state, and delegates execution to Workgroup. Decorates
/// exceptions with assembly context for diagnostics.
///
/// TODO(newling) handle case of multiple kernels in single assembly file.
class Emulator {

public:
  /// Construct from kernel metadata and disassembly text (llvm-objdump -d
  /// output). Optionally accepts original .s source for diagnostic source
  /// mapping. The Emulator has no dependency on LLVM tools — the caller
  /// is responsible for disassembly and metadata extraction.
  Emulator(KernelInfo metadata, std::string_view disassembly,
           std::shared_ptr<Architecture> arch,
           std::string_view originalSource = "");

  Emulator(const Emulator &) = delete;
  Emulator &operator=(const Emulator &) = delete;
  Emulator(Emulator &&other) noexcept;
  Emulator &operator=(Emulator &&) = delete;
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
  WorkgroupConfig buildWorkgroupConfig(int nWaves,
                                       const RunConfig &config) const;
  void initializeWaveState(Workgroup &workgroup, Dim3d wgId, Dim3d blockDim,
                           int nWaves);
  void initKernargSegment();
  std::shared_ptr<Architecture> arch;
  KernelInfo metadata_;
  std::unique_ptr<DisassembledKernel> disassembledKernel;
  std::vector<char> kernargSegment;
  std::vector<bool> kernargIsSet;
  Profiler emulatorProfiler;
  Profiler workgroupProfiler;
};

} // namespace raceemulator
