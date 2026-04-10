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

/// A contiguous range of scalar general-purpose registers.
struct RegisterRange {
  int startRegister;
  int count;
};

/// Describes which SGPRs the hardware pre-loads with special values at kernel
/// launch. For example, the kernarg segment pointer occupies s[0:1], and
/// workgroup IDs occupy s[2], s[3], s[4]. The emulator uses this to initialize
/// wave state before execution.
struct PreloadedRegisters {
  std::map<std::string, RegisterRange> registers;
};

/// Information about a kernel extracted from a code object (.co) file.
struct KernelInfo {
  std::string name;
  WaveSize wavefrontSize{0};
  int kernargSegmentSize = 0;
  std::vector<KernelArg> args;
  std::vector<std::pair<std::string, int>> amdhsa;
  PreloadedRegisters preloadedRegisters;
  int kernargPreloadLength = 0;
  int kernargPreloadOffset = 0;
};

/// Parsed instruction stream. Produced either from llvm-objdump disassembly
/// (via parseDisassembly) or from compiler-emitted .s source (via
/// parseAssemblySource, used only for diagnostic source mapping).
struct DisassembledKernel {
  std::vector<ParsedLine> instructions;

  /// Label name -> instruction index.
  std::map<std::string, int> labels;

  /// Instruction index -> byte address in the code object.
  /// Empty when parsed from .s source.
  std::vector<uint64_t> instructionAddresses;

  /// Source mapping (populated by buildSourceMapping).
  /// Maps each instruction index to its line in the original .s source.
  /// -1 means no mapping for that instruction.
  std::vector<int> instructionToSourceLine;

  /// Raw lines of the original .s source (populated by buildSourceMapping).
  std::vector<std::string> sourceLines;

  void appendStr(std::ostream &os) const;
  std::string str() const;
  void appendInstructionsStr(std::ostream &os) const;
};

/// Parse compiler-emitted .s source into a DisassembledKernel. Only populates
/// instructions and labels — no metadata. Used for diagnostic source mapping.
DisassembledKernel parseAssemblySource(std::string_view assemblySource);

/// Build source mapping by matching labels between disassembly and .s source.
/// Populates result.instructionToSourceLine and result.sourceLines.
void buildSourceMapping(DisassembledKernel &result,
                        const DisassembledKernel &sourceAsm);

/// Parse llvm-objdump -d output into a DisassembledKernel.
DisassembledKernel parseDisassembly(std::string_view disassemblyText);

} // namespace raceemulator
