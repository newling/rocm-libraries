<!--
Copyright Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Metadata in shipped custom kernels

Custom kernel assembly files contain a YAML document inside `.amdgpu_metadata`.
TensileLite reads the complete document using PyYAML's safe loader, preferring
the libyaml implementation when available. The parsed document supplies both
`custom.config` and any arguments needed to infer a missing `CustomKernel`
descriptor. Invalid YAML anywhere in the document causes a read error.

Shipped rocRoller kernels retain their configuration, arguments, runtime fields
and `.command`. They omit `.kernel_graph` and `.kernel_graph_dot`, optional
diagnostic records that describe how the generator constructed the kernel.
These graphs are much larger than the metadata needed to use the assembly.
Keeping them out of shipped files reduces parsing time without depending on
the spelling, order or indentation of YAML keys.

## Exporting rocRoller assembly

Set both existing rocRoller controls in the process that generates or captures
assembly for the CustomKernels tree:

```sh
export ROCROLLER_SERIALIZE_KERNEL_GRAPH=0
export ROCROLLER_SERIALIZE_KERNEL_GRAPH_DOT=0
```

Set the DOT control explicitly: its default follows the assembly-saving
setting, so disabling only the serialized graph may still embed the DOT text.
Keep `.command`; rocRoller uses it when loading saved assembly. Retain every
other metadata field and the assembly instructions.

When using `rocroller-gemm generate`, its `--graph` option saves the diagnostic
graph in a separate file. Keep diagnostic artifacts outside the shipped
CustomKernels tree. This policy applies to the distributed kernels, without
changing rocRoller's debugging defaults for other users.

The unit test `test_shipped_rocroller_metadata_omits_diagnostic_graphs` checks
every shipped rocRoller kernel and reports any graph record that returns.
Future kernel additions must use these export settings; they need no changes
to TensileLite's reader.
