// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/Wave.h"
#include "race-emulator/Workgroup.h"
#include <cstring>
#include <gtest/gtest.h>

using namespace raceemulator;

void tryExecute(Wave &wave, const std::string &line) {
  wave.tryExecute(line, false);
  wave.getWorkgroup().dispatchPendingRaceEvents(wave.getWaveId());
}

// TODO(newling) add tests of control flow instructions

TEST(Instructions, SAndSaveExecB64) {

  Workgroup wg({.vgprCount = 1, .sgprCount = 10, .waveSize = WaveSize{64}});
  auto &wave = wg.getWave(0);

  // populate vcc first 33 bits to 1.
  // populate exec final 33 bits with 1.
  // We then expect the & to be 1 at bits 31 and 32.
  // We check that
  // 1) exec is now these 2 bits
  // 2) s[2:3] is the old exec value.
  // 3) SCC is 1 (because the new exec is non-zero).
  //
  // We then check that EXEC is exactly this value.
  // s_and_saveexec_b64 s[2:3], vcc

  uint64_t vcc = (uint64_t(1) << 33) - 1;
  uint64_t exec = (vcc << 31);
  auto observed = vcc & exec;
  auto expected = uint64_t(3) << 31;
  EXPECT_EQ(observed, expected);

  wave.setExecU64(exec);
  wave.setVccU64(vcc);
  tryExecute(wave, "s_and_saveexec_b64 s[2:3], vcc");
  // The checks:
  EXPECT_EQ(wave.getExecU64(), expected);
  EXPECT_EQ(wave.getSgpr64(2), exec); // old exec saved
  EXPECT_TRUE(wave.getScc());         // new exec non-zero
}

// s_swappc_b64 saves the return byte address
// (next instruction) to the destination SGPR pair and jumps to the byte
// address in the source SGPR pair.
//
// Simulated layout (4 instructions):
//   instruction 0: addr 0x100  (the swappc lives here)
//   instruction 1: addr 0x108  (return point)
//   instruction 2: addr 0x200  (jump target)
//   instruction 3: addr 0x208  (instruction after target)
TEST(Instructions, S_SwapPc_B64) {
  Workgroup wg({.vgprCount = 4,
                .sgprCount = 16,
                .waveSize = WaveSize{32},
                .instructionAddresses = {0x100, 0x108, 0x200, 0x208}});
  auto &wave = wg.getWave(0);

  // Source SGPRs hold the target byte address (0x200 = instruction 2).
  wave.setSgpr64(12, 0x200);

  // Destination SGPRs pre-filled with sentinels.
  wave.setSgpr(4, 0xAAAAAAAA);
  wave.setSgpr(5, 0xBBBBBBBB);

  wave.setPc(0);
  tryExecute(wave, "s_swappc_b64 s[4:5], s[12:13]");

  // PC should jump to instruction index 2 (byte address 0x200).
  EXPECT_EQ(wave.getPc(), 2);

  // Destination SGPRs should hold the return byte address (0x108 = token 1).
  EXPECT_EQ(wave.getSgpr64(4), 0x108u);

  // Source SGPRs are unchanged.
  EXPECT_EQ(wave.getSgpr64(12), 0x200u);
}

// s_sendmsg is a no-op in the emulator (hardware sends a message to the
// SQUEUE, which has no emulator equivalent). Verify it executes without
// throwing and leaves registers unchanged.
TEST(Instructions, S_SENDMSG) {
  Workgroup wg({.vgprCount = 0, .sgprCount = 4, .waveSize = WaveSize{64}});
  auto &wave = wg.getWave(0);
  wave.setSgpr(2, 0xDEADBEEF);
  tryExecute(wave, "s_sendmsg sendmsg(MSG_INTERRUPT)");
  EXPECT_EQ(wave.getSgpr(2), 0xDEADBEEFu); // unchanged
}

// Verify that instruction caching works: when enableLineCaching is true,
// each line should only be compiled once regardless of how many times the
// PC revisits it.
TEST(Instructions, InstructionCaching) {
  std::map<std::string, int> labels = {{"loop", 0}};
  Workgroup wg({.vgprCount = 4,
                .sgprCount = 4,
                .waveSize = WaveSize{1},
                .labels = &labels,
                });
  auto &wave = wg.getWave(0);

  // A 3-instruction loop: set v0, set v1, branch back to start.
  // Run it 5 times (15 instruction executions total).
  std::vector<std::string> program = {
      "v_mov_b32 v0, 42", // PC 0
      "v_mov_b32 v1, 99", // PC 1
      "s_branch loop",    // PC 2 -> jumps to PC 0
  };

  int iterations = 5;
  int totalInstructions = iterations * static_cast<int>(program.size());

  EXPECT_EQ(wave.getCompileCount(), 0);
  for (int i = 0; i < totalInstructions; ++i) {
    int pc = wave.getPc();
    wave.tryExecute(program[pc], /*enableLineCaching=*/true);
  }

  // Each of the 3 unique lines should be compiled exactly once.
  EXPECT_EQ(wave.getCompileCount(), 3);
}

// Verify that without caching, instructions are recompiled each time.
TEST(Instructions, NoCachingRecompiles) {
  std::map<std::string, int> labels = {{"loop", 0}};
  Workgroup wg({.vgprCount = 4,
                .sgprCount = 4,
                .waveSize = WaveSize{1},
                .labels = &labels,
                });
  auto &wave = wg.getWave(0);

  std::vector<std::string> program = {
      "v_mov_b32 v0, 42",
      "v_mov_b32 v1, 99",
      "s_branch loop",
  };

  int iterations = 3;
  int totalInstructions = iterations * static_cast<int>(program.size());

  for (int i = 0; i < totalInstructions; ++i) {
    int pc = wave.getPc();
    wave.tryExecute(program[pc], /*enableLineCaching=*/false);
  }

  // Without caching, each execution compiles fresh.
  EXPECT_EQ(wave.getCompileCount(), totalInstructions);
}
