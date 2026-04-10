// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/Parsing.h"
#include "race-emulator/Util.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace raceemulator {

namespace {

void processInRoot(ParsedLine token, std::map<std::string, int> &labels,
                   std::map<std::string, uint32_t> &symbolTable) {
  // 1. handle labels
  if (token.isLabel) {
    labels.insert({token.key, token.lineNumber});
    return;
  }

  // 2. handle .set directives
  if (token.originalLine.find(".set") == 0) {
    std::string_view lineView = token.originalLine;
    std::string_view content = lineView.substr(4); // skip ".set"

    size_t commaPos = content.find(',');
    if (commaPos == std::string::npos) {
      throw std::runtime_error("parser error: .set directive missing comma");
    }

    // extract key
    std::string key(trim(content.substr(0, commaPos)));

    // extract and evaluate value (supports chained math "A + B + 4")
    std::string_view expr = trim(content.substr(commaPos + 1));
    std::string resultStr = maybeEvaluateExpression(expr, symbolTable);

    // store as integer in map
    uint32_t finalVal;
    if (!parseNumber(resultStr, finalVal)) {
      throw std::runtime_error(
          "internal error: evaluator returned non-number string");
    }
    symbolTable[key] = finalVal;
  }
}

std::string parserStateStr(ParserState state) {
  switch (state) {
  case ParserState::Root:
    return "Root";
  case ParserState::Amdhsa:
    return "Amdhsa";
  case ParserState::Kernels:
    return "Kernels";
  case ParserState::Args:
    return "Args";
  default:
    return "Unknown";
  }
}

// determine how sgprs are initialized.
class KernelStateParser {
public:
  static PreloadedRegisters
  Parse(const std::vector<std::pair<std::string, int>> &metadata) {
    std::map<std::string, int> meta_map;
    for (const auto &item : metadata) {
      meta_map[item.first] = item.second;
    }

    PreloadedRegisters result;
    int current_sgpr = 0;

    // 1. identify user sgpr count
    // this is the explicit boundary between user and system sgprs.
    int user_sgpr_count = 0;
    if (meta_map.find(".amdhsa_user_sgpr_count") != meta_map.end()) {
      user_sgpr_count = meta_map[".amdhsa_user_sgpr_count"];
    }

    // 2. allocate named user sgprs
    // these start at s0 and grow upwards.

    // kernarg segment ptr (s0, s1)
    if (meta_map[".amdhsa_user_sgpr_kernarg_segment_ptr"] != 0) {
      result.registers[".amdhsa_user_sgpr_kernarg_segment_ptr"] = {current_sgpr,
                                                                   2};
      current_sgpr += 2;
    }

    // ... (add other user sgprs here if you support them later) ...

    // 3. enforce the user sgpr boundary
    if (current_sgpr < user_sgpr_count) {
      current_sgpr = user_sgpr_count;
    }

    // 4. allocate system sgprs (workgroup ids)
    const std::vector<std::string> workgroup_ids = {
        ".amdhsa_system_sgpr_workgroup_id_x",
        ".amdhsa_system_sgpr_workgroup_id_y",
        ".amdhsa_system_sgpr_workgroup_id_z"};

    for (const auto &key : workgroup_ids) {
      if (meta_map.count(key) && meta_map[key] != 0) {
        result.registers[key] = {current_sgpr, 1};
        current_sgpr += 1;
      }
    }

    // enforce currently unsupported sgprs are 0
    const std::vector<std::string> unsupported_sgprs = {
        ".amdhsa_user_sgpr_dispatch_ptr",
        ".amdhsa_user_sgpr_queue_ptr",
        ".amdhsa_user_sgpr_dispatch_id",
        ".amdhsa_user_sgpr_flat_scratch_init",
        ".amdhsa_user_sgpr_private_segment_size",
        ".amdhsa_system_sgpr_workgroup_info",
        ".amdhsa_system_sgpr_private_segment_wave_byte_offset"};

    for (const auto &key : unsupported_sgprs) {
      auto found = meta_map.find(key);
      if (found != meta_map.end() && found->second != 0) {
        throw std::runtime_error(
            "parser error: currently unsupported metadata field " + key +
            " is set to " + std::to_string(meta_map[key]) + " (expected 0)");
      }
    }

    return result;
  }
};

} // namespace

std::string ParsedLine::str() const {
  std::ostringstream oss;
  oss << "ParsedLine(indent=";
  oss << indent;
  oss << ", isListItem=";
  oss << isListItem;
  oss << ", isLabel=";
  oss << isLabel;
  oss << ", key='";
  oss << key;
  oss << "'";
  oss << ", value='";
  oss << value << "')";
  return oss.str();
}

ParsedLine::ParsedLine(std::string_view originalLineIn,
                       std::string_view processedLineIn, int n,
                       ParserState state,
                       const std::map<std::string, uint32_t> &symbolTable)
    : originalLine(originalLineIn), lineNumber(n), precedingParserState(state) {

  // In Root state, replace .set symbols with their numeric values.
  processedLine = std::string(processedLineIn);
  if (state == ParserState::Root) {
    processedLine = getSymbolReducedLine(processedLine, symbolTable);
  }

  // Structural parsing uses the input before symbol substitution.
  auto trimmed = trim(processedLineIn);

  // Calculate indentation.
  size_t firstChar = processedLineIn.find_first_not_of(" \t");
  if (firstChar == std::string::npos) {
    isEmptyLine = true;
    return;
  }
  indent = static_cast<int>(firstChar);

  // Check if it is a YAML list item (leading "- ").
  if (trimmed.find("- ", 0) == 0) {
    isListItem = true;
    trimmed = trimmed.substr(2);
  }

  // Check if it is a label (e.g. "foo:").
  size_t colonPos = trimmed.find(':');
  size_t spacePos = trimmed.find_first_of(" \t\n");
  bool hasSpace = (spacePos != std::string::npos);
  bool hasColon = (colonPos != std::string::npos);

  if (indent == 0 && hasColon && !hasSpace) {
    isLabel = true;
    key = trim(trimmed.substr(0, colonPos));
  }

  // Check if key-value pair (e.g. ".name: foo").
  if (!isLabel && hasColon && (!hasSpace || spacePos == colonPos + 1)) {
    isKeyValue = true;
    key = trim(trimmed.substr(0, colonPos));
    value = trim(trimmed.substr(colonPos + 1));
  }

  // AMDHSA section uses space-separated key-value (e.g.
  // ".amdhsa_next_free_vgpr 10").
  if (state == ParserState::Amdhsa) {
    size_t sp = trimmed.find_first_of(" \t");
    if (sp == std::string::npos) {
      return;
    }
    isKeyValue = true;
    key = trim(trimmed.substr(0, sp));
    value = trim(trimmed.substr(sp + 1));
  }
}

DisassembledKernel parseAssemblySource(std::string_view assemblySource) {
  DisassembledKernel result;

  std::vector<std::string> assemblyLines;
  {
    std::string source{assemblySource};
    std::istringstream stream(source);
    std::string currentLine;
    while (std::getline(stream, currentLine)) {
      assemblyLines.push_back(currentLine);
    }
  }

  ParserState state = ParserState::Root;

  auto updateParserState = [&](ParsedLine token) {
    if (state == ParserState::Root &&
        token.originalLine.find(".amdhsa_kernel") != std::string::npos) {
      state = ParserState::Amdhsa;
    }

    if (state == ParserState::Amdhsa &&
        token.originalLine.find(".end_amdhsa_kernel") != std::string::npos) {
      state = ParserState::Root;
    }

    if (state == ParserState::Root && token.key == "amdhsa.kernels") {
      state = ParserState::Kernels;
    }

    if (state == ParserState::Kernels &&
        token.originalLine.find(".end_amdgpu_metadata") != std::string::npos) {
      state = ParserState::Root;
    }
  };

  std::map<std::string, uint32_t> symbolTable;
  symbolTable["UNDEF"] = 0xFFFFFFFF;
  for (unsigned i = 0; i < assemblyLines.size(); ++i) {
    const auto &originalLine = assemblyLines[i];
    ParsedLine token(originalLine, originalLine, i, state, symbolTable);
    result.instructions.push_back(token);
    updateParserState(token);
    if (state == ParserState::Root) {
      processInRoot(token, result.labels, symbolTable);
    }
  }

  return result;
}

void DisassembledKernel::appendStr(std::ostream &os) const {
  os << "Labels:\n";
  std::vector<std::pair<int, std::string>> sortedLabels;
  for (const auto &label : labels) {
    sortedLabels.push_back({label.second, label.first});
  }
  std::sort(sortedLabels.begin(), sortedLabels.end());
  for (const auto &label : sortedLabels) {
    os << "  " << label.second << ": line " << label.first << "\n";
  }
  os << "Instructions: " << instructions.size() << "\n";
}

void DisassembledKernel::appendInstructionsStr(std::ostream &os) const {
  for (const auto &token : instructions) {
    os << "[KV=]" << token.isKeyValue << " [LI]=" << token.isListItem
       << " [LB]=" << token.isLabel << " "
       << parserStateStr(token.precedingParserState) << " "
       << token.processedLine << "\n";
  }
}

std::string DisassembledKernel::str() const {
  std::ostringstream oss;
  appendStr(oss);
  return oss.str();
}

DisassembledKernel parseDisassembly(std::string_view disassemblyText) {
  DisassembledKernel result;

  std::string disasmStr{disassemblyText};
  std::istringstream stream{disasmStr};
  std::string line;
  std::map<std::string, uint32_t> emptySymbolTable;

  while (std::getline(stream, line)) {
    // Skip empty lines and the file format header.
    if (line.empty() || line.find("file format") != std::string::npos ||
        line.find("Disassembly of section") != std::string::npos) {
      continue;
    }

    // Label line: "0000000000000068 <label_LoadArgsEnd>:"
    if (!line.empty() && std::isxdigit(static_cast<unsigned char>(line[0]))) {
      auto openAngle = line.find('<');
      auto closeAngle = line.find('>');
      if (openAngle != std::string::npos && closeAngle != std::string::npos) {
        std::string labelName =
            line.substr(openAngle + 1, closeAngle - openAngle - 1);
        int instructionIndex = static_cast<int>(result.instructions.size());

        // Create a label ParsedLine (skipped by compileLine as a no-op).
        ParsedLine token(line, labelName + ":", instructionIndex,
                         ParserState::Root, emptySymbolTable);
        token.isLabel = true;
        token.key = labelName;
        result.instructions.push_back(std::move(token));
        result.labels[labelName] = instructionIndex;

        // Parse the hex address.
        uint64_t addr = std::stoull(line.substr(0, openAngle), nullptr, 16);
        result.instructionAddresses.push_back(addr);
      }
      continue;
    }

    // Instruction line: "\tmnemonic operands  // ADDR: HEX"
    // Strip leading whitespace.
    auto firstNonSpace = line.find_first_not_of(" \t");
    if (firstNonSpace == std::string::npos) {
      continue;
    }
    std::string trimmed = line.substr(firstNonSpace);

    // Extract PC from trailing "// ADDR: HEX" comment.
    uint64_t pc = 0;
    auto commentPos = trimmed.find("//");
    std::string instructionText = trimmed;
    if (commentPos != std::string::npos) {
      // Parse address from "// 000000001600: F40009C0"
      std::string comment = trimmed.substr(commentPos + 3);
      auto colonPos = comment.find(':');
      if (colonPos != std::string::npos) {
        std::string addrStr = comment.substr(0, colonPos);
        // Trim whitespace from address string.
        auto addrStart = addrStr.find_first_not_of(" \t");
        if (addrStart != std::string::npos) {
          addrStr = addrStr.substr(addrStart);
        }
        pc = std::stoull(addrStr, nullptr, 16);
      }
      // Remove trailing comment and whitespace from instruction text.
      instructionText = trimmed.substr(0, commentPos);
      auto lastNonSpace = instructionText.find_last_not_of(" \t");
      if (lastNonSpace != std::string::npos) {
        instructionText = instructionText.substr(0, lastNonSpace + 1);
      }
    }

    int instructionIndex = static_cast<int>(result.instructions.size());
    ParsedLine token(line, instructionText, instructionIndex, ParserState::Root,
                     emptySymbolTable);
    result.instructions.push_back(std::move(token));
    result.instructionAddresses.push_back(pc);
  }

  return result;
}

SourceMapping buildSourceMapping(const DisassembledKernel &disassembly,
                                 const DisassembledKernel &sourceAsm) {
  SourceMapping mapping;

  // Store original source lines from the parsed source instructions.
  mapping.sourceLines.reserve(sourceAsm.instructions.size());
  for (const auto &inst : sourceAsm.instructions) {
    mapping.sourceLines.push_back(inst.originalLine);
  }

  // Initialize mapping: -1 = no mapping.
  mapping.instructionToSourceLine.assign(disassembly.instructions.size(), -1);

  // Build anchor points from label matching. For each label that exists in
  // both the disassembly and the source, record (disasm instruction index, source
  // line index).
  struct Anchor {
    int disasmIndex;
    int sourceIndex;
  };
  std::vector<Anchor> anchors;
  for (const auto &[name, disasmIndex] : disassembly.labels) {
    auto it = sourceAsm.labels.find(name);
    if (it != sourceAsm.labels.end()) {
      anchors.push_back({disasmIndex, it->second});
    }
  }
  std::sort(anchors.begin(), anchors.end(),
            [](const Anchor &a, const Anchor &b) {
              return a.disasmIndex < b.disasmIndex;
            });

  // Between each pair of consecutive anchors (and before the first / after the
  // last), match instruction lines sequentially. Skip non-instruction lines
  // (comments, directives, blank lines, labels) in the source.
  auto isSourceInstruction = [&](int srcLine) -> bool {
    if (srcLine < 0 || srcLine >= static_cast<int>(sourceAsm.instructions.size())) {
      return false;
    }
    const auto &tok = sourceAsm.instructions[srcLine];
    if (tok.isEmptyLine || tok.isLabel || tok.isKeyValue || tok.isListItem) {
      return false;
    }
    auto trimmed = tok.processedLine;
    auto pos = trimmed.find_first_not_of(" \t");
    if (pos == std::string::npos) {
      return false;
    }
    // Skip directives.
    if (trimmed[pos] == '.') {
      return false;
    }
    // Skip comments.
    if (trimmed[pos] == ';' || trimmed[pos] == '/') {
      return false;
    }
    return true;
  };

  // Add sentinel anchors at start and end.
  anchors.insert(anchors.begin(), {0, 0});
  anchors.push_back({static_cast<int>(disassembly.instructions.size()),
                     static_cast<int>(sourceAsm.instructions.size())});

  for (size_t a = 0; a + 1 < anchors.size(); ++a) {
    int disasmStart = anchors[a].disasmIndex;
    int disasmEnd = anchors[a + 1].disasmIndex;
    int srcCursor = anchors[a].sourceIndex;
    int srcEnd = anchors[a + 1].sourceIndex;

    for (int di = disasmStart; di < disasmEnd; ++di) {
      // Skip label instructions in disassembly (they have their own mapping).
      if (disassembly.instructions[di].isLabel) {
        auto it = sourceAsm.labels.find(disassembly.instructions[di].key);
        if (it != sourceAsm.labels.end()) {
          mapping.instructionToSourceLine[di] = it->second;
        }
        continue;
      }

      // Advance source cursor past non-instruction lines.
      while (srcCursor < srcEnd && !isSourceInstruction(srcCursor)) {
        srcCursor++;
      }

      if (srcCursor < srcEnd) {
        mapping.instructionToSourceLine[di] = srcCursor;
        srcCursor++;
      }
    }
  }

  return mapping;
}

} // namespace raceemulator
