// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/CodeObject.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <map>
#include <sstream>
#include <variant>

namespace raceemulator {

namespace {

// Minimal ELF64 types (no external headers needed).
struct Elf64_Ehdr {
  uint8_t e_ident[16];
  uint16_t e_type, e_machine;
  uint32_t e_version;
  uint64_t e_entry, e_phoff, e_shoff;
  uint32_t e_flags;
  uint16_t e_ehsize, e_phentsize, e_phnum;
  uint16_t e_shentsize, e_shnum, e_shstrndx;
};

struct Elf64_Shdr {
  uint32_t sh_name, sh_type;
  uint64_t sh_flags, sh_addr, sh_offset, sh_size;
  uint32_t sh_link, sh_info;
  uint64_t sh_addralign, sh_entsize;
};

constexpr uint32_t SHT_NOTE = 7;
constexpr uint32_t NT_AMDGPU_METADATA = 32;

// Minimal msgpack decoder for AMDGPU metadata.
// Supports: positive/negative fixint, uint8/16/32, int8/16/32,
// fixstr/str8/str16/str32, fixmap/map16, fixarray/array16.
struct MsgpackValue {
  enum Type { NIL, INT, UINT, STR, ARRAY, MAP } type = NIL;
  int64_t intVal = 0;
  uint64_t uintVal = 0;
  std::string strVal;
  std::vector<MsgpackValue> arrayVal;
  // Map stored as flat alternating key, value, key, value, ...
  // (avoids incomplete-type issues with std::pair<MsgpackValue, MsgpackValue>)
  std::vector<MsgpackValue> mapEntries;

  int64_t asInt() const { return type == UINT ? static_cast<int64_t>(uintVal) : intVal; }
};

class MsgpackReader {
  const uint8_t *data;
  size_t size;
  size_t pos = 0;

  uint8_t readU8() { return data[pos++]; }
  uint16_t readU16() {
    uint16_t v = (uint16_t(data[pos]) << 8) | data[pos + 1];
    pos += 2;
    return v;
  }
  uint32_t readU32() {
    uint32_t v = (uint32_t(data[pos]) << 24) | (uint32_t(data[pos + 1]) << 16) |
                 (uint32_t(data[pos + 2]) << 8) | data[pos + 3];
    pos += 4;
    return v;
  }

public:
  MsgpackReader(const uint8_t *data, size_t size) : data(data), size(size) {}

  MsgpackValue read() {
    if (pos >= size) {
      return {};
    }
    uint8_t tag = readU8();
    MsgpackValue v;

    // Positive fixint (0x00-0x7f).
    if (tag <= 0x7f) {
      v.type = MsgpackValue::UINT;
      v.uintVal = tag;
      return v;
    }
    // Negative fixint (0xe0-0xff).
    if (tag >= 0xe0) {
      v.type = MsgpackValue::INT;
      v.intVal = static_cast<int8_t>(tag);
      return v;
    }
    // Fixmap (0x80-0x8f).
    if ((tag & 0xf0) == 0x80) {
      v.type = MsgpackValue::MAP;
      uint32_t n = tag & 0x0f;
      for (uint32_t i = 0; i < n; ++i) {
        auto key = read();
        auto val = read();
        v.mapEntries.push_back(std::move(key));
      v.mapEntries.push_back(std::move(val));
      }
      return v;
    }
    // Fixarray (0x90-0x9f).
    if ((tag & 0xf0) == 0x90) {
      v.type = MsgpackValue::ARRAY;
      uint32_t n = tag & 0x0f;
      for (uint32_t i = 0; i < n; ++i) {
        v.arrayVal.push_back(read());
      }
      return v;
    }
    // Fixstr (0xa0-0xbf).
    if ((tag & 0xe0) == 0xa0) {
      uint32_t len = tag & 0x1f;
      v.type = MsgpackValue::STR;
      v.strVal.assign(reinterpret_cast<const char *>(data + pos), len);
      pos += len;
      return v;
    }

    switch (tag) {
    case 0xc0: v.type = MsgpackValue::NIL; return v;
    case 0xc2: v.type = MsgpackValue::UINT; v.uintVal = 0; return v; // false
    case 0xc3: v.type = MsgpackValue::UINT; v.uintVal = 1; return v; // true
    case 0xcc: v.type = MsgpackValue::UINT; v.uintVal = readU8(); return v;
    case 0xcd: v.type = MsgpackValue::UINT; v.uintVal = readU16(); return v;
    case 0xce: v.type = MsgpackValue::UINT; v.uintVal = readU32(); return v;
    case 0xd0: v.type = MsgpackValue::INT; v.intVal = static_cast<int8_t>(readU8()); return v;
    case 0xd1: v.type = MsgpackValue::INT; v.intVal = static_cast<int16_t>(readU16()); return v;
    case 0xd2: v.type = MsgpackValue::INT; v.intVal = static_cast<int32_t>(readU32()); return v;
    case 0xd9: { // str8
      uint32_t len = readU8();
      v.type = MsgpackValue::STR;
      v.strVal.assign(reinterpret_cast<const char *>(data + pos), len);
      pos += len;
      return v;
    }
    case 0xda: { // str16
      uint32_t len = readU16();
      v.type = MsgpackValue::STR;
      v.strVal.assign(reinterpret_cast<const char *>(data + pos), len);
      pos += len;
      return v;
    }
    case 0xdb: { // str32
      uint32_t len = readU32();
      v.type = MsgpackValue::STR;
      v.strVal.assign(reinterpret_cast<const char *>(data + pos), len);
      pos += len;
      return v;
    }
    case 0xdc: { // array16
      uint32_t n = readU16();
      v.type = MsgpackValue::ARRAY;
      for (uint32_t i = 0; i < n; ++i) {
        v.arrayVal.push_back(read());
      }
      return v;
    }
    case 0xdd: { // array32
      uint32_t n = readU32();
      v.type = MsgpackValue::ARRAY;
      for (uint32_t i = 0; i < n; ++i) {
        v.arrayVal.push_back(read());
      }
      return v;
    }
    case 0xde: { // map16
      uint32_t n = readU16();
      v.type = MsgpackValue::MAP;
      for (uint32_t i = 0; i < n; ++i) {
        auto key = read();
        auto val = read();
        v.mapEntries.push_back(std::move(key));
      v.mapEntries.push_back(std::move(val));
      }
      return v;
    }
    case 0xdf: { // map32
      uint32_t n = readU32();
      v.type = MsgpackValue::MAP;
      for (uint32_t i = 0; i < n; ++i) {
        auto key = read();
        auto val = read();
        v.mapEntries.push_back(std::move(key));
      v.mapEntries.push_back(std::move(val));
      }
      return v;
    }
    default:
      // Skip unknown types.
      return v;
    }
  }
};

const MsgpackValue *findInMap(const MsgpackValue &map, std::string_view key) {
  if (map.type != MsgpackValue::MAP) {
    return nullptr;
  }
  for (size_t i = 0; i + 1 < map.mapEntries.size(); i += 2) {
    if (map.mapEntries[i].type == MsgpackValue::STR &&
        map.mapEntries[i].strVal == key) {
      return &map.mapEntries[i + 1];
    }
  }
  return nullptr;
}

} // namespace

CodeObjectResult parseCodeObjectMetadata(const uint8_t *elfData, size_t elfSize,
                                         const std::string &kernelName) {
  auto fail = [](std::string msg) -> CodeObjectResult { return msg; };

  if (elfSize < sizeof(Elf64_Ehdr)) {
    return fail("ELF too small");
  }

  Elf64_Ehdr ehdr;
  std::memcpy(&ehdr, elfData, sizeof(ehdr));
  if (ehdr.e_shoff + ehdr.e_shnum * sizeof(Elf64_Shdr) > elfSize) {
    return fail("section header table extends beyond ELF");
  }

  // Find .note section with msgpack metadata.
  const uint8_t *noteData = nullptr;
  for (uint16_t i = 0; i < ehdr.e_shnum; ++i) {
    Elf64_Shdr shdr;
    std::memcpy(&shdr, elfData + ehdr.e_shoff + i * sizeof(Elf64_Shdr),
                sizeof(shdr));
    if (shdr.sh_type == SHT_NOTE &&
        shdr.sh_offset + shdr.sh_size <= elfSize) {
      noteData = elfData + shdr.sh_offset;
      break;
    }
  }
  if (!noteData) {
    return fail("no SHT_NOTE section found");
  }

  // Parse the note header.
  uint32_t namesz, descsz, ntype;
  std::memcpy(&namesz, noteData, 4);
  std::memcpy(&descsz, noteData + 4, 4);
  std::memcpy(&ntype, noteData + 8, 4);
  if (ntype != NT_AMDGPU_METADATA) {
    return fail("note type " + std::to_string(ntype) + " != 32");
  }

  uint32_t namePadded = (namesz + 3) & ~3u;
  const uint8_t *msgpackData = noteData + 12 + namePadded;

  // Parse msgpack.
  MsgpackReader reader(msgpackData, descsz);
  auto top = reader.read();
  auto *kernelsVal = findInMap(top, "amdhsa.kernels");
  if (!kernelsVal || kernelsVal->type != MsgpackValue::ARRAY ||
      kernelsVal->arrayVal.empty()) {
    return fail("amdhsa.kernels not found or empty");
  }

  // Find the target kernel.
  const MsgpackValue *kernel = nullptr;
  for (const auto &k : kernelsVal->arrayVal) {
    auto *nameVal = findInMap(k, ".name");
    if (nameVal && nameVal->type == MsgpackValue::STR) {
      if (kernelName.empty() || nameVal->strVal == kernelName) {
        kernel = &k;
        break;
      }
    }
  }
  if (!kernel) {
    // Fall back to first kernel.
    kernel = &kernelsVal->arrayVal[0];
  }

  KernelMetadata result;

  auto *nameVal = findInMap(*kernel, ".name");
  if (nameVal && nameVal->type == MsgpackValue::STR) {
    result.name = nameVal->strVal;
  }

  auto *wfSize = findInMap(*kernel, ".wavefront_size");
  if (wfSize) {
    result.wavefrontSize = WaveSize{static_cast<int>(wfSize->asInt())};
  }

  auto *kargSize = findInMap(*kernel, ".kernarg_segment_size");
  if (kargSize) {
    result.kernargSegmentSize = static_cast<int>(kargSize->asInt());
  }

  // Parse kernel args.
  auto *argsVal = findInMap(*kernel, ".args");
  if (argsVal && argsVal->type == MsgpackValue::ARRAY) {
    for (const auto &arg : argsVal->arrayVal) {
      KernelArg ka;
      auto *offset = findInMap(arg, ".offset");
      if (offset) {
        ka.offset = static_cast<int>(offset->asInt());
      }
      auto *sz = findInMap(arg, ".size");
      if (sz) {
        ka.size = static_cast<int>(sz->asInt());
      }
      auto *vk = findInMap(arg, ".value_kind");
      if (vk && vk->type == MsgpackValue::STR) {
        ka.valueKind = vk->strVal;
      }
      auto *nm = findInMap(arg, ".name");
      if (nm && nm->type == MsgpackValue::STR) {
        ka.name = nm->strVal;
      }
      auto *as = findInMap(arg, ".address_space");
      if (as && as->type == MsgpackValue::STR) {
        ka.addressSpace = as->strVal;
      }
      if (!ka.valueKind.empty()) {
        result.args.push_back(std::move(ka));
      }
    }
  }

  // Build amdhsa metadata from kernel descriptor fields in msgpack.
  auto addAmdhsa = [&](const char *key, const MsgpackValue *val) {
    if (val) {
      result.amdhsa.emplace_back(key, static_cast<int>(val->asInt()));
    }
  };

  auto *sgprCount = findInMap(*kernel, ".sgpr_count");
  auto *vgprCount = findInMap(*kernel, ".vgpr_count");
  auto *groupSize = findInMap(*kernel, ".group_segment_fixed_size");

  addAmdhsa(".amdhsa_next_free_sgpr", sgprCount);
  addAmdhsa(".amdhsa_next_free_vgpr", vgprCount);
  addAmdhsa(".amdhsa_group_segment_fixed_size", groupSize);


  // Parse the kernel descriptor binary for register allocation.
  // Find the .kd symbol in the symbol table.
  struct Elf64_Sym {
    uint32_t st_name;
    uint8_t st_info, st_other;
    uint16_t st_shndx;
    uint64_t st_value, st_size;
  };

  for (uint16_t si = 0; si < ehdr.e_shnum; ++si) {
    Elf64_Shdr shdr;
    std::memcpy(&shdr, elfData + ehdr.e_shoff + si * sizeof(Elf64_Shdr),
                sizeof(shdr));
    if (shdr.sh_type != 2) { // SHT_SYMTAB
      continue;
    }
    if (shdr.sh_entsize == 0 || shdr.sh_link >= ehdr.e_shnum) {
      continue;
    }

    // Read the string table for this symbol table.
    Elf64_Shdr strShdr;
    std::memcpy(&strShdr,
                elfData + ehdr.e_shoff + shdr.sh_link * sizeof(Elf64_Shdr),
                sizeof(strShdr));

    size_t numSyms = shdr.sh_size / shdr.sh_entsize;
    for (size_t j = 0; j < numSyms; ++j) {
      Elf64_Sym sym;
      std::memcpy(&sym, elfData + shdr.sh_offset + j * sizeof(Elf64_Sym),
                  sizeof(sym));
      if (sym.st_name >= strShdr.sh_size) {
        continue;
      }
      const char *symName = reinterpret_cast<const char *>(
          elfData + strShdr.sh_offset + sym.st_name);
      std::string name(symName);
      if (name.size() < 3 || name.substr(name.size() - 3) != ".kd") {
        continue;
      }
      if (sym.st_shndx >= ehdr.e_shnum || sym.st_size < 64) {
        continue;
      }

      // Read the 64-byte kernel descriptor.
      Elf64_Shdr secShdr;
      std::memcpy(&secShdr,
                  elfData + ehdr.e_shoff + sym.st_shndx * sizeof(Elf64_Shdr),
                  sizeof(secShdr));
      size_t kdOff = secShdr.sh_offset + sym.st_value;
      if (kdOff + 64 > elfSize) {
        continue;
      }
      const uint8_t *kd = elfData + kdOff;

      uint32_t rsrc1, rsrc2, rsrc3;
      uint16_t kcp, preloadSpec;
      std::memcpy(&rsrc3, kd + 44, 4);
      std::memcpy(&rsrc1, kd + 48, 4);
      std::memcpy(&rsrc2, kd + 52, 4);
      std::memcpy(&kcp, kd + 56, 2);
      std::memcpy(&preloadSpec, kd + 58, 2);

      // Extract kernarg preload from kernarg_preload_spec (offset 58).
      result.kernargPreloadLength = preloadSpec & 0x3F;
      result.kernargPreloadOffset = (preloadSpec >> 6) & 0x3F;

      // Extract VGPR/SGPR counts from compute_pgm_rsrc1.
      uint32_t vgprGran = (rsrc1 & 0x3F) + 1;
      uint32_t sgprGran = ((rsrc1 >> 6) & 0x0F) + 1;
      int vgprs = static_cast<int>(vgprGran * 8);
      int sgprs = static_cast<int>(sgprGran * 8);

      // Extract accum_offset from compute_pgm_rsrc3 bits[5:0].
      uint32_t accumOffsetField = rsrc3 & 0x3F;
      int accumOffset = static_cast<int>((accumOffsetField + 1) * 4);
      if (accumOffset > 4) {
        result.amdhsa.emplace_back(".amdhsa_accum_offset", accumOffset);
        // Ensure vgprCount covers both VGPRs and AGPRs.
        vgprs = std::max(vgprs, accumOffset * 2);
      }

      // Override msgpack vgpr/sgpr counts if they were lower.
      bool hasVgprFromKd = true;
      for (auto &[k, v] : result.amdhsa) {
        if (k == ".amdhsa_next_free_vgpr") {
          v = std::max(v, vgprs);
          hasVgprFromKd = false;
        }
        if (k == ".amdhsa_next_free_sgpr") {
          v = std::max(v, sgprs);
        }
      }
      if (hasVgprFromKd) {
        result.amdhsa.emplace_back(".amdhsa_next_free_vgpr", vgprs);
        result.amdhsa.emplace_back(".amdhsa_next_free_sgpr", sgprs);
      }

      bool kernargPtr = (kcp >> 3) & 1;
      bool wgIdX = (rsrc2 >> 7) & 1;
      bool wgIdY = (rsrc2 >> 8) & 1;
      bool wgIdZ = (rsrc2 >> 9) & 1;

      // Build amdhsa key-value pairs for register allocation.
      if (kernargPtr) {
        result.amdhsa.emplace_back(
            ".amdhsa_user_sgpr_kernarg_segment_ptr", 1);
      }
      if (wgIdX) {
        result.amdhsa.emplace_back(".amdhsa_system_sgpr_workgroup_id_x", 1);
      }
      if (wgIdY) {
        result.amdhsa.emplace_back(".amdhsa_system_sgpr_workgroup_id_y", 1);
      }
      if (wgIdZ) {
        result.amdhsa.emplace_back(".amdhsa_system_sgpr_workgroup_id_z", 1);
      }

      // Build initial register allocation. User SGPRs come first (kernarg
      // pointer at s[0:1]), then system SGPRs (workgroup IDs) start after
      // all user SGPRs. The user_sgpr_count from compute_pgm_rsrc2 gives
      // the total number of user SGPRs including preloaded kernargs.
      uint32_t userSgprCount = (rsrc2 >> 1) & 0x1F;
      int currentSgpr = 0;
      if (kernargPtr) {
        result.initialRegisterAllocation
            .registers[".amdhsa_user_sgpr_kernarg_segment_ptr"] = {
            currentSgpr, 2};
        // System SGPRs start after ALL user SGPRs, not after the pointer.
        currentSgpr = static_cast<int>(userSgprCount);
      }
      if (wgIdX) {
        result.initialRegisterAllocation
            .registers[".amdhsa_system_sgpr_workgroup_id_x"] = {currentSgpr,
                                                                 1};
        currentSgpr += 1;
      }
      if (wgIdY) {
        result.initialRegisterAllocation
            .registers[".amdhsa_system_sgpr_workgroup_id_y"] = {currentSgpr,
                                                                 1};
        currentSgpr += 1;
      }
      if (wgIdZ) {
        result.initialRegisterAllocation
            .registers[".amdhsa_system_sgpr_workgroup_id_z"] = {currentSgpr,
                                                                 1};
        currentSgpr += 1;
      }
      break; // Found the kernel descriptor, done.
    }
    break; // Only check the first symbol table.
  }

  return result;
}

} // namespace raceemulator
