// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "race-emulator/Arch.h"
#include "race-emulator/CodeObject.h"
#include "race-emulator/Emulator.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace raceemulator::test {

namespace fs = std::filesystem;

/// Run a shell command and return its stdout. Throws on failure.
inline std::string captureCommand(const std::string &cmd) {
  auto tmpPath = fs::temp_directory_path() /
      ("race_emu_" + std::to_string(std::hash<std::thread::id>{}(
                         std::this_thread::get_id())) + ".txt");
  int status = std::system((cmd + " > " + tmpPath.string() + " 2>&1").c_str());
  std::ifstream ifs(tmpPath);
  std::string result((std::istreambuf_iterator<char>(ifs)),
                     std::istreambuf_iterator<char>());
  fs::remove(tmpPath);
  if (status != 0) {
    throw std::runtime_error("Command failed (status " +
                              std::to_string(status) + "): " + cmd +
                              "\nOutput: " + result);
  }
  return result;
}

/// Assemble a .s string into a code object using llvm-mc.
/// LLVM_MC_PATH is set by CMake at configure time.
inline std::vector<uint8_t> assembleToCodeObject(std::string_view assembly,
                                                  const std::string &mcpu) {
  auto tid = std::to_string(
      std::hash<std::thread::id>{}(std::this_thread::get_id()));
  auto sPath = fs::temp_directory_path() / ("race_emu_asm_" + tid + ".s");
  auto objPath = fs::temp_directory_path() / ("race_emu_co_" + tid + ".o");
  {
    std::ofstream ofs(sPath);
    ofs << assembly;
  }
  captureCommand(std::string(LLVM_MC_PATH) +
                 " -triple=amdgcn-amd-amdhsa -mcpu=" + mcpu +
                 " -filetype=obj " + sPath.string() + " -o " + objPath.string());

  std::ifstream ifs(objPath, std::ios::binary | std::ios::ate);
  auto size = static_cast<size_t>(ifs.tellg());
  std::vector<uint8_t> coData(size);
  ifs.seekg(0);
  ifs.read(reinterpret_cast<char *>(coData.data()), size);

  fs::remove(sPath);
  fs::remove(objPath);
  return coData;
}

/// Load a kernel .s file from TEST_KERNEL_DIR.
inline std::string loadKernelFile(const std::string &filename) {
  fs::path filepath = fs::path(TEST_KERNEL_DIR) / filename;
  std::ifstream file(filepath);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open kernel file: " +
                             filepath.string());
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

/// Disassemble a code object using llvm-objdump.
/// LLVM_OBJDUMP_PATH is set by CMake at configure time.
inline std::string disassembleCodeObject(const std::vector<uint8_t> &co) {
  auto tid = std::to_string(
      std::hash<std::thread::id>{}(std::this_thread::get_id()));
  auto coPath = fs::temp_directory_path() / ("race_emu_co_" + tid + ".o");
  {
    std::ofstream ofs(coPath, std::ios::binary);
    ofs.write(reinterpret_cast<const char *>(co.data()), co.size());
  }
  std::string disasm = captureCommand(
      std::string(LLVM_OBJDUMP_PATH) + " -d --show-all-symbols " +
      coPath.string());
  fs::remove(coPath);
  return disasm;
}

/// Assemble .s source, disassemble, parse metadata, create Emulator.
/// Optionally builds a source mapping from .s labels for diagnostics.
inline Emulator emulatorFromAssembly(std::string_view assembly,
                                     std::shared_ptr<Architecture> arch,
                                     bool withSourceMapping = true) {
  auto co = assembleToCodeObject(assembly, arch->getName());
  std::string disassembly = disassembleCodeObject(co);

  auto metaResult = parseCodeObjectMetadata(co.data(), co.size());
  if (std::holds_alternative<std::string>(metaResult)) {
    throw std::runtime_error("Failed to parse code object metadata: " +
                             std::get<std::string>(metaResult));
  }

  auto kernelInfo = std::get<KernelInfo>(std::move(metaResult));

  if (withSourceMapping) {
    auto disassembled = parseDisassembly(disassembly);
    auto sourceAsm = parseAssemblySource(assembly);
    return Emulator(std::move(kernelInfo), disassembly, std::move(arch),
                    buildSourceMapping(disassembled, sourceAsm));
  }

  return Emulator(std::move(kernelInfo), disassembly, std::move(arch));
}

} // namespace raceemulator::test
