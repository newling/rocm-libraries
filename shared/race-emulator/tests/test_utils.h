// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "race-emulator/Arch.h"
#include "race-emulator/Emulator.h"
#include <cstdlib>
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

/// Find an LLVM tool by name (checks LLVM_BIN_DIR env, then PATH).
inline std::string findLlvmTool(const std::string &name) {
  if (const char *dir = std::getenv("LLVM_BIN_DIR")) {
    std::string path = std::string(dir) + "/" + name;
    if (fs::exists(path)) {
      return path;
    }
  }
  try {
    std::string result = captureCommand("which " + name + " 2>/dev/null");
    if (!result.empty() && result.back() == '\n') {
      result.pop_back();
    }
    return result;
  } catch (...) {
    return "";
  }
}

/// Assemble a .s string into a code object using llvm-mc.
/// Returns the .co bytes. Throws if llvm-mc is not found or assembly fails.
inline std::vector<uint8_t> assembleToCodeObject(std::string_view assembly,
                                                  const std::string &mcpu) {
  std::string llvmMc = findLlvmTool("llvm-mc");
  if (llvmMc.empty()) {
    throw std::runtime_error("llvm-mc not found. Set LLVM_BIN_DIR or PATH.");
  }

  auto tid = std::to_string(
      std::hash<std::thread::id>{}(std::this_thread::get_id()));
  auto sPath = fs::temp_directory_path() / ("race_emu_asm_" + tid + ".s");
  auto objPath = fs::temp_directory_path() / ("race_emu_co_" + tid + ".o");
  {
    std::ofstream ofs(sPath);
    ofs << assembly;
  }
  captureCommand(llvmMc + " -triple=amdgcn-amd-amdhsa -mcpu=" + mcpu +
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

/// Assemble .s text and create an Emulator from the resulting code object.
/// Passes the original .s source for diagnostic source mapping.
inline Emulator emulatorFromAssembly(std::string_view assembly,
                                     std::shared_ptr<Architecture> arch,
                                     bool withSourceMapping = true) {
  auto co = assembleToCodeObject(assembly, arch->getName());
  return Emulator(co.data(), co.size(), std::move(arch),
                  withSourceMapping ? assembly : "");
}

} // namespace raceemulator::test
