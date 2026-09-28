# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Build-time identity for serialized solution defaults and deployed device code.

Hash each generated artifact once. Whole bundles (including all helper variants
for the architecture) deliberately trade precise invalidation for simplicity.
This is not a fingerprint of the host runtime or the execution environment.
"""

import hashlib
import json
from pathlib import Path

from .Common import state

PREFIX = "sha256-v1:"
_NON_IDENTITY_KEYS = {"index", "libraryLogicIndex", "ideals", "linearModel", "fingerprint"}


def solution_fingerprint(solution_state, architecture, main_digest, helper_digests):
    payload = {
        "schema": PREFIX,
        "architecture": architecture,
        "solution": {k: v for k, v in solution_state.items() if k not in _NON_IDENTITY_KEYS},
        "main": main_digest,
        "helpers": sorted(set(helper_digests)),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return PREFIX + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SolutionFingerprints:
    def __init__(self, code_objects):
        # Use the builders' outputs, never a glob that could pick up stale files.
        self.digests = {}
        for path in sorted({Path(p).resolve() for p in code_objects}):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            self.digests[path] = digest.hexdigest()

    def library_state(self, library, base_arch, arch_directory):
        """Call after applyNaming and passPostKernelInfoToLibrary have finished."""
        document = state(library)
        directory = Path(arch_directory).resolve()
        helpers = [
            digest
            for path, digest in self.digests.items()
            if path.parent == directory and path.suffix == ".hsaco"
        ]
        for serialized in document["solutions"]:
            solution = library.solutions[serialized["index"]]
            bundle = solution.originalSolution.get("codeObjectFile", f"TensileLibrary_{base_arch}")
            main = directory / (bundle + ".co")
            if main not in self.digests or not helpers:
                raise RuntimeError(
                    f"Cannot fingerprint solution {serialized['index']}: "
                    f"missing generated main or helper code objects in {directory}"
                )
            serialized["fingerprint"] = solution_fingerprint(
                serialized, directory.name, self.digests[main], helpers
            )
        return document
