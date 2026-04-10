// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "CommonRegister.h"
#include "IntervalSet.h"
#include "Profiler.h"
#include "Types.h"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

namespace raceemulator {

class Workgroup;
class WaveRaceState;

/// A parsed instruction operand: either a literal value or a register
/// reference.
template <typename T> struct Operand {
  bool isLiteral;
  CommonRegister reg;
  T literalValue;

  void appendStr(std::ostream &os) const {
    if (isLiteral) {
      os << "Literal(" << literalValue << ")";
    } else {
      os << "Register";
    }
  }
};


/// A single SIMD wave. Manages VGPRs, SGPRs, exec mask, program counter,
/// and delegates race detection to WaveRaceState (if race checks enabled).
/// Delegates LDS storage to the Workgroup.
class Wave {
  // Workgroup creates waves in its constructor and passes race state pointers.
  friend class Workgroup;

public:
  ~Wave();
  Wave(Wave &&other) noexcept;
  Wave &operator=(Wave &&other) noexcept;

  /// Parse a register string (e.g., "s17", "s[16:17]", "v[4:7]", "vcc",
  /// "exec") and return the first register in the range.
  CommonRegister getFirstRegister(std::string_view regStr) const;

  /// Return the value in the given VGPR. Calls the race handler if race
  /// checks are enabled and an outstanding load targets this register.
  uint32_t getVgpr(int reg, int lane) const;

  /// Read all lanes of a single VGPR into out[0..waveSize-1]. Race checking
  /// is O(1) per register via regEventCount; only falls back to per-lane
  /// checking when outstanding events exist.
  void getVgprs(int reg, uint32_t *out) const;

  /// Return a single byte from a VGPR. Race-checks only that byte.
  /// byteIdx 0 = bits [7:0], 3 = bits [31:24].
  uint8_t getVgprByte(int id, int lane, int byteIdx) const;

  /// Return 16 bits from a VGPR. Race-checks only the relevant 2 bytes.
  /// hi=false: bits [15:0], hi=true: bits [31:16].
  uint16_t getHalfVgpr(int id, int lane, bool hi) const;

  /// Write 16 bits directly to VGPR storage without reading the other half.
  /// Matches ISA d16 semantics.
  void setVgprHalf(int id, int lane, bool hi, uint16_t value);

  uint32_t getSgpr(int id) const;

  void setVgpr(int reg, int lane, uint32_t value) {
    assert((getExecU64() & (1ULL << lane)) != 0 &&
           "Writing to VGPR of inactive lane");
    auto index = reg * waveSize + lane;
    assert(index < static_cast<int64_t>(vgprs.size()));
    vgprs[index] = value;
  }
  /// Write a VGPR without exec mask check (for DPP which ignores exec).
  void setVgprRaw(int reg, int lane, uint32_t value) {
    auto index = reg * waveSize + lane;
    assert(index < static_cast<int64_t>(vgprs.size()));
    vgprs[index] = value;
  }
  void setSgpr(int id, uint32_t value);

  /// Get/set a pair of consecutively numbered 32-bit registers as uint64_t.
  uint64_t getSgpr64(int id) const;
  uint64_t getVgpr64(int id, int lane) const;
  void setSgpr64(int id, uint64_t value);
  void setVgpr64(int id, int lane, uint64_t value);

  WaveSize getWaveSize() const { return waveSize; }

  int getSgprCount() const { return sgprCount; }

  void tryExecute(const std::string &line, bool enableLineCaching);

  /// Number of times tryExecute has compiled an instruction (as opposed to
  /// using a cached executor). With caching enabled, each unique PC should
  /// be compiled at most once. Dual instructions (A :: B) count as one
  /// compilation despite compiling two sub-instructions internally.
  int getCompileCount() const { return compileCount; }

  void setScc(bool value);
  bool getScc() const;

  void setM0(uint32_t value);
  uint32_t getM0() const;

  void setVccU32(uint32_t value);
  uint32_t getVccU32() const;
  void setVccU64(uint64_t value);
  uint64_t getVccU64() const;

  void setExecU32(uint32_t value);
  uint32_t getExecU32() const;

  /// Exec mask as uint64_t. Safe for both wave-32 and wave-64: on wave-32
  /// the upper 32 bits are always zero.
  void setExecU64(uint64_t value);
  uint64_t getExecU64() const;

  Workgroup &getWorkgroup();

  /// Validated LDS read: auto-supplies waveId and delegates to Workgroup.
  template <typename T> T readLds(int addr, int lane) const;

  /// Validated bulk LDS read: validates the full range once, then copies
  /// count elements of type T.
  template <typename T>
  void readLds(int addr, int lane, T *out, int count) const;

  /// Validated LDS write: auto-supplies waveId and delegates to Workgroup.
  template <typename T> void writeLds(int addr, int lane, T value);

  bool isCompleteEmulation() const;

  WaveId getWaveId() const { return waveId; }

  Profiler::ScopedStopwatch profileScope(std::string_view key);

  /// Discard all wave-complete events. Called when all waves have reached
  /// s_barrier (not when individual waves arrive).
  void flushWaveCompleteMemoryEvents();

  void setPendingMemoryEvent(PendingMemoryEvent event);
  void setPendingWaitCount(PendingWaitCount waitCount);

  template <typename T> T getValue(const Operand<T> &operand, int lane) const;
  template <typename T>
  T getSgprOrLiteralValue(const Operand<T> &operand) const;
  template <typename T> Operand<T> parseOperand(std::string_view token) const;

  void setPc(int newPc) { pc = newPc; }
  int getPc() const { return pc; }

  const std::map<std::string, int> &getLabels() const {
    assert(labels != nullptr && "Labels map is null");
    return *labels;
  }

  /// Whether real byte-address PCs are available (disassembly path).
  bool hasPcTable() const { return pcTable != nullptr; }

  /// Return the real byte address for the given token index, or fall back
  /// to the synthetic 4*index when pcTable is not available.
  uint64_t getByteAddress(int tokenIndex) const {
    if (pcTable && tokenIndex >= 0 &&
        static_cast<size_t>(tokenIndex) < pcTable->size()) {
      return (*pcTable)[tokenIndex];
    }
    return static_cast<uint64_t>(4 * tokenIndex);
  }

  /// Look up the token index for a byte address. Returns -1 if not found.
  /// Only works when pcTable is available.
  int getTokenIndexFromByteAddress(uint64_t addr) const {
    if (!pcTable) {
      return static_cast<int>(addr / 4);
    }
    for (size_t i = 0; i < pcTable->size(); ++i) {
      if ((*pcTable)[i] == addr) {
        return static_cast<int>(i);
      }
    }
    return -1;
  }

  void setDsPreserve(bool preserve) { dsPreserve = preserve; }
  bool getDsPreserve() const { return dsPreserve; }

  void deactivate() { active = false; }
  void setWaiting(bool w) { waiting = w; }
  bool isActive() const { return active; }
  bool isWaiting() const { return waiting; }

private:

  // The current program counter (the current line in the assembly).
  int pc{0};

  // accumulator and vector general purpose registers.
  std::vector<uint32_t> vgprs;

  // scalar general purpose registers.
  std::vector<uint32_t> sgprs;

  int agprOffset;
  int sgprCount;
  WaveSize waveSize;

  /// Whether this wave is still executing (false after s_endpgm).
  bool active{true};

  /// Whether this wave is waiting at an s_barrier.
  bool waiting{false};

  /// D16 LDS reads: whether to preserve the non-targeted half of the VGPR.
  bool dsPreserve;

  // The id of this wave within the workgroup.
  WaveId waveId{0};

  // Pointer to the workgroup state (LDS memory, event registry).
  Workgroup *workgroup;

  // Pointer to the label map for the assembly.
  const std::map<std::string, int> *labels;

  // Token index → byte address. Null when running from .s (line-index PCs).
  const std::vector<uint64_t> *pcTable = nullptr;

  std::vector<std::function<int()>> instructionCache;

  // Tracks the number of instruction compilations (calls to compileLine that
  // produce an executor). Used for verifying that caching avoids recompilation.
  int compileCount = 0;

  // Per-wave race detection state. Null when race checks are disabled.
  // Owned by RaceDetector (via Workgroup), not by Wave.
  WaveRaceState *raceState = nullptr;

  std::optional<PendingMemoryEvent> pendingMemoryEvent;
  std::optional<PendingWaitCount> pendingWaitCount;

  Wave(int vgprCount, int agprOffset, int sgprCount, WaveSize, WaveId,
       Workgroup &workgroup, const std::map<std::string, int> *labels);

  std::function<int()> compileLine(const std::string &line);

public:
  template <typename F> void runExecConditionedForLanes(F func) {
    forEachActiveLane(getExecU64(), getWaveSize(), func);
  }
};

} // namespace raceemulator
