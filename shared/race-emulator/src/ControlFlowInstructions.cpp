// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/Instruction.h"
#include "race-emulator/Util.h"
#include "race-emulator/Wave.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace raceemulator {

namespace {

// SOPP: Wait for memory counts (s_waitcnt)
class SOPP_WaitCnt : public Instruction {
public:
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    // Examples: s_waitcnt lgkmcnt(0), s_waitcnt vmcnt(0) expcnt(0)
    auto partitioned = getPartitioned(line);

    int vmcnt = -1;
    int lgkmcnt = -1;

    // Iterate through tokens starting after the mnemonic
    for (size_t i = 1; i < partitioned.size(); ++i) {
      std::string_view token = partitioned[i];

      constexpr const char *vmcntPrefix = "vmcnt(";
      auto lenVmcntPrefix = std::strlen(vmcntPrefix);

      constexpr const char *lgkmcntPrefix = "lgkmcnt(";
      auto lenLgkmcntPrefix = std::strlen(lgkmcntPrefix);

      if (token.substr(0, lenVmcntPrefix) == vmcntPrefix) {
        size_t end = token.find(')');
        assert(end != std::string_view::npos && "Malformed vmcnt token");
        auto valStr = token.substr(lenVmcntPrefix, end - lenVmcntPrefix);
        vmcnt = getIntFromView<int>(valStr);
      } else if (token.substr(0, lenLgkmcntPrefix) == lgkmcntPrefix) {
        size_t end = token.find(')');
        assert(end != std::string_view::npos && "Malformed lgkmcnt token");
        auto valStr = token.substr(lenLgkmcntPrefix, end - lenLgkmcntPrefix);
        lgkmcnt = getIntFromView<int>(valStr);
      } else if (token == "0") {
        // s_waitcnt 0: wait for all counters to reach 0
        vmcnt = 0;
        lgkmcnt = 0;
      } else if (token.find("expcnt(") == 0 || token.find("vscnt(") == 0) {
        // expcnt and vscnt are not modelled by the emulator — ignore.
      } else {
        throw std::runtime_error("Unsupported s_waitcnt counter: " +
                                 std::string(token));
      }
    }

    return [&wave, vmcnt, lgkmcnt]() {
      wave.setPendingWaitCount({vmcnt, lgkmcnt});
      return wave.getPc() + 1;
    };
  }
};

// SOPP: Synchronization barrier (s_barrier)
class SOPP_Barrier : public Instruction {
public:
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    assert(partitioned.size() == 1);

    return [&wave]() {
      wave.setWaiting(true);
      return wave.getPc() + 1;
    };
  }
};

// SOPP: Save execution mask (s_and_saveexec_b64, etc.)
template <typename Op> class SOPP_SaveExec : public Instruction {
  Op op;

public:
  SOPP_SaveExec(Op op) : op(std::move(op)) {}

  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    assert(partitioned.size() == 3);

    auto dst = wave.getFirstRegister(partitioned[1]);
    assert(dst.type == CommonRegister::Type::SGPR &&
           "SaveExec dest must be SGPR");

    auto src0 = wave.parseOperand<uint64_t>(partitioned[2]);
    if (!src0.isLiteral) {
      assert(src0.reg.type == CommonRegister::Type::SGPR &&
             "SaveExec src must be SGPR");
    }

    auto operation = this->op;

    return [&wave, dst, src0, operation]() {
      uint64_t originalExec = wave.getExecU64();
      uint64_t srcVal = wave.getSgprOrLiteralValue(src0);
      uint64_t newExec = operation(srcVal, originalExec);

      // Save old exec and set new exec. Use 32-bit or 64-bit
      // accessors depending on instruction variant (_b32 vs _b64).
      if (wave.getWaveSize() <= 32) {
        wave.setExecU32(static_cast<uint32_t>(newExec));
        wave.setSgpr(dst.index, static_cast<uint32_t>(originalExec));
      } else {
        wave.setExecU64(newExec);
        wave.setSgpr64(dst.index, originalExec);
      }
      wave.setScc(newExec != 0);

      return wave.getPc() + 1;
    };
  }
};

// SOPP: Conditional branch (s_branch, s_cbranch_*)
template <typename Op> class SOPP_Branch : public Instruction {
  Op op;

public:
  SOPP_Branch(Op op) : op(std::move(op)) {}

  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    assert(partitioned.size() == 2);

    auto target = std::string(partitioned[1]);
    int labelIndex;

    // Check if the target is a label name or a numeric offset.
    bool isNumeric = !target.empty() &&
                     (std::isdigit(static_cast<unsigned char>(target[0])) ||
                      target[0] == '-');
    if (isNumeric && wave.hasPcTable()) {
      // Numeric offset: signed 16-bit dword count from PC+4. Large unsigned
      // values (e.g. 65520 = 0xFFF0) are negative offsets (backward jumps).
      int raw = std::stoi(target);
      int16_t offset = static_cast<int16_t>(raw & 0xFFFF);
      uint64_t currentPc = wave.getByteAddress(wave.getPc());
      uint64_t targetAddr = currentPc + 4 + static_cast<int64_t>(offset) * 4;
      labelIndex = wave.getTokenIndexFromByteAddress(targetAddr);
    } else {
      const auto &labels = wave.getLabels();
      auto it = labels.find(target);
      if (it == labels.end()) {
        throw std::runtime_error("Branch target label '" + target +
                                 "' not found in labels map");
      }
      labelIndex = it->second;
    }
    auto operation = this->op;

    return [&wave, operation, labelIndex]() {
      if (operation(wave)) {
        return labelIndex;
      }
      return wave.getPc() + 1;
    };
  }
};

// SOPP: Program counter manipulation (s_getpc, s_setpc)
class SOPP_GetPc : public Instruction {
public:
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    auto dst = partitioned[1];

    return [&wave, dst]() {
      auto dstReg = wave.getFirstRegister(dst);
      assert(dstReg.type == CommonRegister::Type::SGPR);

      wave.setSgpr64(dstReg.index, wave.getByteAddress(wave.getPc() + 1));
      return wave.getPc() + 1;
    };
  }
};

class SOPP_SetPc : public Instruction {
public:
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    auto src = partitioned[1];

    return [&wave, src]() {
      auto srcReg = wave.getFirstRegister(src);
      assert(srcReg.type == CommonRegister::Type::SGPR);
      uint64_t targetAddr = wave.getSgpr64(srcReg.index);
      return wave.getTokenIndexFromByteAddress(targetAddr);
    };
  }
};

// SOPP: End program (s_endpgm)
class SOPP_EndPgm : public Instruction {
public:
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view /*line*/) const final {
    return [&wave]() {
      wave.deactivate();
      return wave.getPc() + 1;
    };
  }
};

// SOPP: No operation (s_nop, etc.)
class SOPP_NoOp : public Instruction {
public:
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view /*line*/) const final {
    return [&]() { return wave.getPc() + 1; };
  }
};

// Debugging utility (Pseudo-instruction)
class DebugPrint : public Instruction {
public:
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    std::string type(partitioned[1]);
    std::string regString = std::string(partitioned[2]);
    auto reg = wave.getFirstRegister(regString);
    int lane = getIntFromView<int>(partitioned[3]);

    std::string comment;
    constexpr auto minLength = 5;
    if (partitioned.size() >= minLength) {
      comment = std::string(partitioned[4]);
    }

    return [&wave, type, reg, lane, comment, regString]() {
      std::ostringstream oss;
      oss << "DebugPrint: " << type << " " << regString << ": ";

      if (reg.type == CommonRegister::Type::VGPR) {
        oss << wave.getVgpr(reg.index, lane);
      } else if (reg.type == CommonRegister::Type::SGPR) {
        oss << wave.getSgpr(reg.index);
      } else {
        oss << "(unknown register type)";
      }

      if (!comment.empty()) {
        oss << "   " << comment;
      }

      std::cout << oss.str() << "\n";
      return wave.getPc() + 1;
    };
  }
};

// Helpers for templates
template <typename Op>
static std::unique_ptr<Instruction> makeSOPP_SaveExec(Op op) {
  return std::make_unique<SOPP_SaveExec<Op>>(std::move(op));
}

template <typename Op>
static std::unique_ptr<Instruction> makeSOPP_Branch(Op op) {
  return std::make_unique<SOPP_Branch<Op>>(std::move(op));
}

// Registration

// Standard adapter
template <typename InstT> struct Register {
  template <typename... Args>
  Register(const std::string &name, Args &&...args) {
    InstructionRegistry::instance().add(
        name, std::make_unique<InstT>(std::forward<Args>(args)...));
  }
};

// Factory adapter
struct RegisterFactory {
  RegisterFactory(const std::string &name, std::unique_ptr<Instruction> inst) {
    InstructionRegistry::instance().add(name, std::move(inst));
  }
};

// s_swappc_b64: on real HW this saves PC+4 (return address) to the
// destination SGPR pair and jumps to the byte address in the source pair.
//
// When instructionAddresses is available (disassembly path), this is fully functional:
// saves the return address (next instruction) to dst and jumps to the
// byte address in src. When instructionAddresses is not available (.s path), falls back
// to no-op behaviour (correct only for activationType=0).
class SOPP_SwapPc : public Instruction {
public:
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    assert(partitioned.size() == 3);
    auto dst = partitioned[1];
    auto src = partitioned[2];

    return [&wave, dst, src]() {
      // Without instructionAddresses, byte addresses are synthetic (4 * lineIndex) and
      // s_getpc + s_add_u32 patterns produce wrong targets. Fall back to
      // no-op, which is correct for activationType=0 (identity).
      if (!wave.hasPcTable()) {
        return wave.getPc() + 1;
      }

      auto dstReg = wave.getFirstRegister(dst);
      auto srcReg = wave.getFirstRegister(src);
      assert(dstReg.type == CommonRegister::Type::SGPR);
      assert(srcReg.type == CommonRegister::Type::SGPR);

      uint64_t targetAddr = wave.getSgpr64(srcReg.index);
      int targetIndex = wave.getTokenIndexFromByteAddress(targetAddr);

      if (targetIndex < 0) {
        fprintf(stderr, "WARNING: s_swappc_b64 target address 0x%lx not found "
                        "in instructionAddresses, emulating as no-op.\n", targetAddr);
        return wave.getPc() + 1;
      }

      // Save return address (next instruction after this one).
      wave.setSgpr64(dstReg.index, wave.getByteAddress(wave.getPc() + 1));
      return targetIndex;
    };
  }
};

// --- Program Control ---
const Register<SOPP_GetPc> s_getpc("s_getpc_b64");
const Register<SOPP_SetPc> s_setpc("s_setpc_b64");
const Register<SOPP_SwapPc> s_swappc("s_swappc_b64");
const Register<SOPP_NoOp> s_nop("s_nop");
const Register<SOPP_NoOp> s_setprio("s_setprio");
const Register<SOPP_EndPgm> s_endpgm("s_endpgm");
const Register<SOPP_NoOp> s_clause("s_clause");
const Register<SOPP_NoOp> s_delay_alu("s_delay_alu");
const Register<SOPP_NoOp> s_code_end("s_code_end");
const Register<SOPP_NoOp> s_sendmsg("s_sendmsg");
// Cache invalidation -- no semantic effect in the emulator.
const Register<SOPP_NoOp> buffer_gl0_inv("buffer_gl0_inv");
const Register<SOPP_WaitCnt> s_waitcnt("s_waitcnt");
const Register<SOPP_Barrier> s_barrier("s_barrier");
const Register<DebugPrint> debug_print("print");

// --- Execution Mask (SaveExec) ---
const RegisterFactory
    s_and_exec("s_and_saveexec_b64",
               makeSOPP_SaveExec([](uint64_t a, uint64_t b) { return a & b; }));

const RegisterFactory s_andn2_exec(
    "s_andn2_saveexec_b64",
    makeSOPP_SaveExec([](uint64_t a, uint64_t b) { return a & (~b); }));

// Wave-32 variants (RDNA)
const RegisterFactory s_and_exec_b32(
    "s_and_saveexec_b32",
    makeSOPP_SaveExec([](uint64_t a, uint64_t b) { return a & b; }));
const RegisterFactory s_andn2_exec_b32(
    "s_andn2_saveexec_b32",
    makeSOPP_SaveExec([](uint64_t a, uint64_t b) { return a & (~b); }));
// RDNA rename of s_andn2_saveexec_b32
const RegisterFactory s_and_not1_exec_b32(
    "s_and_not1_saveexec_b32",
    makeSOPP_SaveExec([](uint64_t a, uint64_t b) { return a & (~b); }));

// --- Branching ---
const RegisterFactory s_cb_scc0("s_cbranch_scc0", makeSOPP_Branch([](Wave &w) {
                                  return w.getScc() == 0;
                                }));

const RegisterFactory s_cb_scc1("s_cbranch_scc1", makeSOPP_Branch([](Wave &w) {
                                  return w.getScc() != 0;
                                }));

const RegisterFactory s_branch("s_branch",
                               makeSOPP_Branch([](Wave &) { return true; }));

const RegisterFactory s_cb_ez("s_cbranch_execz", makeSOPP_Branch([](Wave &w) {
                                return w.getExecU64() == 0;
                              }));

const RegisterFactory s_cb_enz("s_cbranch_execnz", makeSOPP_Branch([](Wave &w) {
                                 return w.getExecU64() != 0;
                               }));

const RegisterFactory s_cb_vnz("s_cbranch_vccnz", makeSOPP_Branch([](Wave &w) {
                                 if (w.getWaveSize() <= 32) {
                                   return w.getVccU32() != 0;
                                 }
                                 return w.getVccU64() != 0;
                               }));

const RegisterFactory s_cb_vz("s_cbranch_vccz", makeSOPP_Branch([](Wave &w) {
                                if (w.getWaveSize() <= 32) {
                                  return w.getVccU32() == 0;
                                }
                                return w.getVccU64() == 0;
                              }));

} // namespace
} // namespace raceemulator
