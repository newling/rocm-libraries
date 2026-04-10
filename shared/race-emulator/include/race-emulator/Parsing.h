// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "Types.h"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace raceemulator {

/// Tracks which section of the assembly file the parser is currently in.
enum class ParserState { Root, Amdhsa, Kernels, Args };

/// A single line of assembly after parsing: comments stripped, symbols
/// resolved, labels and key-value pairs identified.
struct ParsedLine {
  ParsedLine(std::string_view originalLine, std::string_view commentFreeLine,
             int lineNumber, ParserState precedingParserState,
             const std::map<std::string, uint32_t> &symbolTable);

  /// The raw source line, with comments. Used in error messages.
  std::string originalLine;
  /// Comments stripped, .set symbols replaced by their numeric values.
  /// This is the line that instruction executors see via tryExecute().
  std::string commentFreeLine;
  bool isEmptyLine{false};
  /// 0-based line number in the original assembly.
  int lineNumber;
  ParserState precedingParserState;
  /// Column index of the first non-whitespace character.
  int indent{-1};
  bool isListItem{false};
  bool isLabel{false};
  bool isKeyValue{false};
  std::string key{"none"};
  std::string value{"none"};

  std::string str() const;
};

/// A kernel argument descriptor from the AMDGPU metadata YAML section.
struct KernelArg {
  int size = 0;
  int offset = 0;
  std::string valueKind;
  std::string valueType;
  std::string addressSpace;
  std::string name;
};

/// Pre-allocated scalar register ranges (e.g., kernarg pointer, workgroup ID),
/// determined by parsing the assembly metadata.
struct RegisterMapping {
  int start_register;
  int count;
};
struct AllocationResult {
  std::map<std::string, RegisterMapping> registers;
};

/// Top-level result of parsing an AMD GPU assembly file: kernel metadata,
/// parsed instruction lines, labels, and macros.
struct ParsedAsm {
  ParsedAsm(std::string_view assembly);

  std::string name;
  std::string assembly;
  std::string target;
  int kernargSegmentSize = 0;
  WaveSize wavefrontSize{0};
  std::vector<KernelArg> args;
  std::vector<ParsedLine> tokens;
  AllocationResult initialRegisterAllocation;
  std::vector<std::pair<std::string, int>> amdhsa;

  /// Kernarg preload: hardware preloads this many dwords from the kernarg
  /// segment into SGPRs starting right after the kernarg segment pointer.
  int kernargPreloadLength = 0;
  int kernargPreloadOffset = 0;

  /// Label name -> line index.
  std::map<std::string, int> labels;

  /// Token index -> byte address. Populated by parseDisassembly, empty for
  /// .s-parsed assemblies. When non-empty, s_getpc/s_setpc/s_swappc use
  /// real byte addresses instead of the synthetic 4*lineIndex scheme.
  std::vector<uint64_t> pcTable;

  /// Source mapping for diagnostics: maps disassembly token index to the
  /// corresponding line index in the original source (sourceLines). When
  /// populated, decorateException shows original source lines with comments
  /// instead of raw disassembly lines. -1 means no mapping for that token.
  std::vector<int> sourceLineMap;

  /// Original source lines (from the .s file). Used by decorateException
  /// when sourceLineMap is populated.
  std::vector<std::string> sourceLines;

  void appendStr(std::ostream &os) const;
  std::string str() const;
  void appendTokensStr(std::ostream &os) const;
};

/// Build a source line mapping from disassembly token indices to original
/// source line indices. Uses label matching as anchor points and sequential
/// instruction matching between anchors. Populates result.sourceLineMap
/// and result.sourceLines.
void buildSourceMapping(ParsedAsm &result, const ParsedAsm &sourceAsm);

/// Parse llvm-objdump -d output into a ParsedAsm. Populates tokens, labels,
/// and pcTable. Metadata fields (name, wavefrontSize, args, amdhsa, etc.)
/// are NOT populated — they must be set by the caller from the code object.
ParsedAsm parseDisassembly(std::string_view disassemblyText);

} // namespace raceemulator
