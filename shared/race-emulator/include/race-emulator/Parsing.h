// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "Types.h"
#include <cstdint>
#include <map>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

namespace raceemulator {

/// Tracks which section of the assembly file the parser is currently in.
enum class ParserState { Root, Amdhsa, Kernels, Args };

/// A single parsed line of assembly or disassembly.
struct ParsedLine {
  ParsedLine(std::string_view originalLine, std::string_view processedLine,
             int lineNumber, ParserState precedingParserState,
             const std::map<std::string, uint32_t> &symbolTable);

  /// The raw source line (with comments). Used for diagnostic display.
  std::string originalLine;
  /// Processed line: .set symbols resolved (for .s path), or trailing
  /// // ADDR: HEX comment stripped (for disassembly path).
  std::string processedLine;
  bool isEmptyLine{false};
  int lineNumber;
  ParserState precedingParserState;
  int indent{-1};
  bool isListItem{false};
  bool isLabel{false};
  bool isKeyValue{false};
  std::string key{"none"};
  std::string value{"none"};

  std::string str() const;
};

/// A kernel argument descriptor from the AMDGPU metadata.
struct KernelArg {
  int size = 0;
  int offset = 0;
  std::string valueKind;
  std::string valueType;
  std::string addressSpace;
  std::string name;
};

/// Pre-allocated scalar register ranges (e.g., kernarg pointer, workgroup ID).
struct RegisterMapping {
  int start_register;
  int count;
};
struct AllocationResult {
  std::map<std::string, RegisterMapping> registers;
};

/// Parsed kernel metadata and instruction tokens. Populated from either the
/// .s parser (for source mapping) or parseDisassembly + parseCodeObjectMetadata.
struct ParsedAsm {
  /// Parse .s assembly source for metadata and labels (used for source mapping).
  ParsedAsm(std::string_view assembly);

  std::string name;
  std::string assembly;
  int kernargSegmentSize = 0;
  WaveSize wavefrontSize{0};
  std::vector<KernelArg> args;
  std::vector<ParsedLine> tokens;
  AllocationResult initialRegisterAllocation;
  std::vector<std::pair<std::string, int>> amdhsa;
  int kernargPreloadLength = 0;
  int kernargPreloadOffset = 0;

  /// Label name -> line index.
  std::map<std::string, int> labels;

  /// Token index -> byte address. Populated by parseDisassembly.
  std::vector<uint64_t> pcTable;

  /// Source mapping for diagnostics: disassembly token index -> original
  /// source line index. -1 = no mapping.
  std::vector<int> sourceLineMap;

  /// Original source lines for diagnostic display.
  std::vector<std::string> sourceLines;

  void appendStr(std::ostream &os) const;
  std::string str() const;
  void appendTokensStr(std::ostream &os) const;
};

/// Build source mapping from disassembly tokens to original source lines.
void buildSourceMapping(ParsedAsm &result, const ParsedAsm &sourceAsm);

/// Parse llvm-objdump -d output into a ParsedAsm (tokens, labels, pcTable).
ParsedAsm parseDisassembly(std::string_view disassemblyText);

} // namespace raceemulator
