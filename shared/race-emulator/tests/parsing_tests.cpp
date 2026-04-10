// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/Arch.h"
#include "race-emulator/Emulator.h"
#include "race-emulator/Parsing.h"
#include "race-emulator/Util.h"
#include "test_utils.h"
#include <cstring>
#include <gtest/gtest.h>
#include <string_view>

using namespace raceemulator;


TEST(Parser, Gfx1151ArchProperties) {
  Gfx1151 arch;
  EXPECT_EQ(arch.getName(), "gfx1151");
  EXPECT_EQ(arch.getMaxLdsSize(), 65536);
}

TEST(Parser, ArchitectureFromTargetGfx1151) {
  auto arch = architectureFromTarget("amdgcn-amd-amdhsa--gfx1151");
  EXPECT_EQ(arch->getName(), "gfx1151");
  EXPECT_EQ(arch->getMaxLdsSize(), 65536);
}

// --- stripComments unit tests ---

TEST(StripComments, LineCommentSemicolon) {
  bool inBlock = false;
  EXPECT_EQ(stripComments("code ; comment", inBlock), "code ");
  EXPECT_FALSE(inBlock);
}

TEST(StripComments, LineCommentDoubleSlash) {
  bool inBlock = false;
  EXPECT_EQ(stripComments("code // comment", inBlock), "code ");
  EXPECT_FALSE(inBlock);
}

TEST(StripComments, SingleLineBlockComment) {
  bool inBlock = false;
  EXPECT_EQ(stripComments("a /* comment */ b", inBlock), "a  b");
  EXPECT_FALSE(inBlock);
}

TEST(StripComments, MultipleBlockCommentsOnOneLine) {
  bool inBlock = false;
  EXPECT_EQ(stripComments("/* x */ a /* y */ b", inBlock), " a  b");
  EXPECT_FALSE(inBlock);
}

TEST(StripComments, BlockCommentOpensMultiLine) {
  bool inBlock = false;
  EXPECT_EQ(stripComments("code /* start of block", inBlock), "code ");
  EXPECT_TRUE(inBlock);
}

TEST(StripComments, InsideBlockComment) {
  bool inBlock = true;
  EXPECT_EQ(stripComments("  still in comment", inBlock), "");
  EXPECT_TRUE(inBlock);
}

TEST(StripComments, BlockCommentCloses) {
  bool inBlock = true;
  EXPECT_EQ(stripComments("  end of block */ code", inBlock), " code");
  EXPECT_FALSE(inBlock);
}

TEST(StripComments, BlockCommentClosesThenReopens) {
  bool inBlock = true;
  EXPECT_EQ(stripComments("in */ out /* in */ out", inBlock), " out  out");
  EXPECT_FALSE(inBlock);
}

TEST(StripComments, BlockCommentInsideLineComment) {
  bool inBlock = false;
  EXPECT_EQ(stripComments("code // not /* a block", inBlock), "code ");
  EXPECT_FALSE(inBlock);
}

TEST(StripComments, BlockCommentInsideSemicolonComment) {
  bool inBlock = false;
  EXPECT_EQ(stripComments("code ; not /* a block", inBlock), "code ");
  EXPECT_FALSE(inBlock);
}

TEST(StripComments, NoComments) {
  bool inBlock = false;
  EXPECT_EQ(stripComments("v_mov_b32_e32 v0, 0", inBlock),
            "v_mov_b32_e32 v0, 0");
  EXPECT_FALSE(inBlock);
}

TEST(StripComments, EmptyLine) {
  bool inBlock = false;
  EXPECT_EQ(stripComments("", inBlock), "");
  EXPECT_FALSE(inBlock);
}

TEST(Parser, ParseDisassembly) {
  static constexpr std::string_view disasm = R"(
/tmp/test.o:	file format elf64-amdgpu

Disassembly of section .text:

0000000000000000 <my_kernel>:
	s_load_b64 s[0:1], s[0:1], 0x0                             // 000000000000: F4040000 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000000008: BF89FC07
	v_add_f32_e32 v0, s0, v0                                   // 00000000000C: 06000000
	s_cbranch_scc0 done                                        // 000000000010: BFA10001
	v_mul_f32_e32 v0, s1, v0                                   // 000000000014: 10000001

0000000000000018 <done>:
	global_store_b32 v1, v0, s[0:1]                            // 000000000018: DC680000 00000100
	s_endpgm                                                   // 000000000020: BFB00000
)";

  auto result = parseDisassembly(disasm);

  // Check tokens: 2 labels + 7 instructions = 9 entries.
  EXPECT_EQ(result.tokens.size(), 9u);

  // Check labels.
  ASSERT_EQ(result.labels.count("my_kernel"), 1u);
  ASSERT_EQ(result.labels.count("done"), 1u);
  EXPECT_EQ(result.labels.at("my_kernel"), 0);
  EXPECT_EQ(result.labels.at("done"), 6);

  // Check pcTable matches byte addresses.
  ASSERT_EQ(result.pcTable.size(), result.tokens.size());
  EXPECT_EQ(result.pcTable[0], 0x0u);  // my_kernel label
  EXPECT_EQ(result.pcTable[1], 0x0u);  // s_load_b64
  EXPECT_EQ(result.pcTable[2], 0x8u);  // s_waitcnt
  EXPECT_EQ(result.pcTable[3], 0xCu);  // v_add_f32
  EXPECT_EQ(result.pcTable[4], 0x10u); // s_cbranch_scc0
  EXPECT_EQ(result.pcTable[5], 0x14u); // v_mul_f32
  EXPECT_EQ(result.pcTable[6], 0x18u); // done label
  EXPECT_EQ(result.pcTable[7], 0x18u); // global_store_b32
  EXPECT_EQ(result.pcTable[8], 0x20u); // s_endpgm

  // Check instruction text is clean (no trailing comments).
  EXPECT_EQ(result.tokens[1].commentFreeLine, "s_load_b64 s[0:1], s[0:1], 0");
  EXPECT_EQ(result.tokens[3].commentFreeLine, "v_add_f32_e32 v0, s0, v0");
  EXPECT_EQ(result.tokens[4].commentFreeLine, "s_cbranch_scc0 done");

}
