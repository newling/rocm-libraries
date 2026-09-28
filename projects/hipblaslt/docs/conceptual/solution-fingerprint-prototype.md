# Solution fingerprint prototype

See the [review and related branches](pr12585-review.md) for the published review, concrete review fixes, and comparison with PR #12585.

This explores a stronger identity check for the offline tuning records in PR
#12585. A kernel name can stay unchanged when a solution's launch defaults or
compiled code changes. The prototype records a build-time fingerprint alongside
the solution index and kernel name, then requires that fingerprint to match at
replay. The index remains the lookup location; moved solutions are not searched
for by fingerprint.

During TensileCreateLibrary, each generated code-object file is hashed once with
SHA-256. After final launch metadata and names have been assigned, each solution
gets a `sha256-v1:<64 lowercase hex digits>` fingerprint of:

- Its serialized solution state, including launch settings, predicates and custom
  kernel metadata. Only index, library-logic index, ranking estimates (`ideals`
  and `linearModel`), and the fingerprint itself are excluded.
- Its main compiled `.co` bundle and every generated helper `.hsaco` variant for
  the architecture.
- The architecture's output identity, including an ASIC revision when present.

Dictionary order and installation path do not affect the fingerprint. Bundle
contents do: an unrelated kernel change in a shared bundle can invalidate other
solutions. This conservative invalidation keeps the implementation small. Missing
generated artifacts fail generation instead of producing a partial fingerprint.

The fingerprint is stored as optional solution metadata, so old device libraries
remain readable. `hipblaslt-bench` obtains it through the experimental C++ accessor
`getSolutionFingerprintFromAlgo` and writes a `solution_fingerprint` CSV column.
Fingerprint rows require a recognized, nonempty, matching value; they never fall
back to matching names or build stamps. Both C and C++ tuning replay use this check.
Duplicate detection includes the fingerprint, so a stale row cannot hide a later
valid row at the same index. Existing files without the column retain #12585's
legacy behavior and its weaker guarantee. Strict checking requires the new
runtime; older readers ignore the new column and still check names.

New tuning rows are skipped with a diagnostic when metadata lacks a fingerprint
or explicit split-K/WGM overrides were measured. Those overrides are not restored
by the tuning-file reader. Normal benchmark output remains available. This also
means a newly built benchmark cannot record tuning against older device libraries
until those libraries are regenerated.

## Scope of the guarantee

This identifies generated solution defaults and device code, assuming that code
objects and metadata are deployed together. It does not attest files on disk at
runtime, establish equal performance, or identify the host runtime, compiler flags
used to build that runtime, driver, GPU clocks, or environment settings such as
`TENSILE_AUTO_GSU_ALGO`, `TENSILE_ADAPTIVE_GEMM_NTAB_ALGO` and Stream-K/debug overrides.
The existing replay problem/support/workspace checks still apply.

Consequently this prototype is not a universal guarantee across arbitrary builds
and execution environments. A production compatibility contract must also cover
host dispatch changes and runtime overrides (for example, a host compatibility
identity and recorded/restored execution settings). The `v1` prefix versions the
fingerprint recipe; it does not currently enforce that broader contract.

## Validation and cost

There is no hashing during heuristic queries or GEMM execution. Runtime work is
metadata lookup and comparison of a 74-character value. A local build-time sample
hashed 21 installed compiled artifacts totaling 55.3 MiB in 61 ms. Generating 100
fingerprints from real serialized solution metadata took 16 ms, including Python
state conversion. These are illustrative measurements, not full-build benchmarks.

The focused Python suite covers changed launch settings, main/helper bytes,
architecture, stable ordering/relocation, missing artifacts and serialization.
The `hipblaslt-tuning-parser-test` target exercises the real parser without GPU
execution. The existing tuning-cache suite additionally has C/C++ replay tests
that require a GPU and regenerated fingerprinted device metadata.

From an out-of-tree working directory, with the TensileLite venv and a configured
hipBLASLt client-test build:

```sh
python -m pytest "$SRC/projects/hipblaslt/tensilelite/Tensile/Tests/unit/test_SolutionFingerprint.py" -q
cmake --build "$BUILD" --target hipblaslt-tuning-parser-test --parallel 4
"$BUILD/clients/tests/src/hipblaslt-tuning-parser-test"
```

Local validation also loaded metadata from a real logic fixture through the C++
MessagePack and indexed MessagePack loaders: WGMXCC 1 -> 8 preserved both names,
changed the fingerprint, and failed the identity check. Old metadata loaded with
an empty fingerprint. That demonstration used synthetic artifact bytes and did
not execute GPU kernels. Host library, benchmark and GPU-test compilation were
checked; a full device-library/GPU run remains outstanding because the installed
ROCm 7.1 assembler rejects this tree's `+real-true16` option.
