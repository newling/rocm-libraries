// Minimal integration test: verify that RaceConditionException thrown inside
// the race-emulator shared library is correctly caught by a consumer compiled
// with a potentially different compiler/C++ standard.

#include "race-emulator/Arch.h"
#include "race-emulator/CodeObject.h"
#include "race-emulator/Emulator.h"
#include "race-emulator/EmulatorException.h"
#include "race-emulator/Parsing.h"
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

std::string captureCommand(const std::string &cmd) {
  auto tmpPath = fs::temp_directory_path() / "race_emu_integ.txt";
  int status = std::system((cmd + " > " + tmpPath.string() + " 2>&1").c_str());
  std::ifstream ifs(tmpPath);
  std::string result((std::istreambuf_iterator<char>(ifs)),
                      std::istreambuf_iterator<char>());
  fs::remove(tmpPath);
  if (status != 0)
    throw std::runtime_error("Command failed: " + cmd + "\n" + result);
  return result;
}

raceemulator::Emulator emulatorFromAssembly(
    std::string_view assembly,
    std::shared_ptr<raceemulator::Architecture> arch) {
  auto sPath = fs::temp_directory_path() / "race_emu_integ.s";
  auto objPath = fs::temp_directory_path() / "race_emu_integ.o";
  { std::ofstream ofs(sPath); ofs << assembly; }

  captureCommand(std::string(LLVM_MC_PATH) +
                 " -triple=amdgcn-amd-amdhsa -mcpu=" + arch->getName() +
                 " -filetype=obj " + sPath.string() +
                 " -o " + objPath.string());

  std::ifstream ifs(objPath, std::ios::binary | std::ios::ate);
  auto size = static_cast<size_t>(ifs.tellg());
  std::vector<uint8_t> coData(size);
  ifs.seekg(0);
  ifs.read(reinterpret_cast<char *>(coData.data()), size);

  std::string disasm = captureCommand(
      std::string(LLVM_OBJDUMP_PATH) + " -d --show-all-symbols " +
      objPath.string());

  fs::remove(sPath);
  fs::remove(objPath);

  auto metaResult = raceemulator::parseCodeObjectMetadata(
      coData.data(), coData.size());
  if (std::holds_alternative<std::string>(metaResult))
    throw std::runtime_error("Failed to parse code object: " +
                             std::get<std::string>(metaResult));

  return raceemulator::Emulator(
      std::get<raceemulator::KernelInfo>(std::move(metaResult)),
      disasm, std::move(arch));
}

// Kernel metadata shared by both test kernels. Describes a single kernel "foo"
// with one 8-byte global_buffer argument, targeting gfx942 with wave size 64.
const std::string kKernelMetadata = R"ASM(
  .amdhsa_kernel foo
    .amdhsa_group_segment_fixed_size 1024
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 8
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_dispatch_ptr 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_user_sgpr_dispatch_id 0
    .amdhsa_enable_private_segment 0
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_sgpr 10
    .amdhsa_next_free_vgpr 10
    .amdhsa_accum_offset 10
  .end_amdhsa_kernel
  .amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 1024
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .name:           foo
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
...
  .end_amdgpu_metadata
)ASM";

const std::string kBoilerHeader = R"ASM(
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .text
  .globl foo
  .p2align 8
)ASM";

// A minimal kernel with a VGPR race: global_load writes to v1, then ds_write
// reads v1 without a s_waitcnt vmcnt(0) in between.
const std::string kRaceyAsm = kBoilerHeader + R"ASM(
foo:
  s_load_dwordx2 s[0:1], s[0:1], 0x0
  v_lshlrev_b32_e32 v0, 2, v0
  s_waitcnt lgkmcnt(0)
  global_load_dword v1, v0, s[0:1]
  ; Missing s_waitcnt vmcnt(0) -- this is the race.
  ds_write_b32 v0, v1
  s_waitcnt lgkmcnt(0)
  s_endpgm
)ASM" + kKernelMetadata;

// A race-free version of the same kernel (with the wait).
const std::string kCleanAsm = kBoilerHeader + R"ASM(
foo:
  s_load_dwordx2 s[0:1], s[0:1], 0x0
  v_lshlrev_b32_e32 v0, 2, v0
  s_waitcnt lgkmcnt(0)
  global_load_dword v1, v0, s[0:1]
  s_waitcnt vmcnt(0)
  ds_write_b32 v0, v1
  s_waitcnt lgkmcnt(0)
  s_endpgm
)ASM" + kKernelMetadata;

raceemulator::Emulator makeEmulator(const std::string &asmText) {
  auto arch = std::make_shared<raceemulator::Gfx942>();
  auto emulator = emulatorFromAssembly(asmText, arch);

  static std::vector<int> data(4, 0);
  int *ptr = data.data();
  emulator.addKernarg(0, &ptr);
  return emulator;
}

} // namespace

static constexpr raceemulator::RunConfig kRaceCheckConfig{
    .raceChecks = true, .completeEmulation = false};

TEST(RaceEmulatorIntegration, DetectsVgprRaceAcrossLibraryBoundary) {
  auto emulator = makeEmulator(kRaceyAsm);
  EXPECT_THROW(emulator.run({0, 0, 0}, {64, 1, 1}, kRaceCheckConfig),
               raceemulator::RaceConditionException);
}

TEST(RaceEmulatorIntegration, CleanAsmPassesWithoutException) {
  auto emulator = makeEmulator(kCleanAsm);
  EXPECT_NO_THROW(emulator.run({0, 0, 0}, {64, 1, 1}));
}
