# Lessons from the `numpyexample` manual integration case

This reference compresses the reusable integration knowledge from the local
`numpyexample` case that integrated `smooth_l1_loss_v2` and
`apply_gradient_descent`. It deliberately omits the case's repository fetches,
dependency patching, machine-specific paths, high-parallel build commands, and
persistent shell changes.

Use it only when the target AsNumpy checkout has no manifest generator.

## End-to-end call chain

The case proves that a usable integration must connect all of these layers:

```text
asnumpy.<public_api>
  -> Python argument normalization and NPU-array conversion
  -> pybind custom_ops binding
  -> AsNumpy C++ wrapper
  -> aclnn<Name>GetWorkspaceSize
  -> workspace allocation
  -> aclnn<Name> execution on the active stream/device
  -> synchronization/error propagation required by the target runtime
  -> NPUArray result
```

Changing only Python exports or linking a library is not an integration.

## Files the legacy case had to connect

The older checkout used explicit C++ declarations/implementations, a shared
pybind source, a Python API module, package lazy exports, CMake source/link
registration, and NPU tests. Use these as layer categories, not fixed file
names. Inspect the target's equivalents before editing.

Simple tensor-in/same-shape-out operators may fit the target's unary/binary
helpers. Attribute-bearing, reduction, mutation, dimension-changing, or
multi-output operators require an explicit wrapper whose GetWorkspaceSize and
execute signatures exactly match the installed header.

## Private OPP facts

The case's most important build/runtime lesson is path identity:

- the installed vendor contains `op_api/include`,
  `op_api/lib/libcust_opapi.so`, and kernel metadata/binaries;
- the build needs the vendor header and library locations;
- the runtime needs the same vendor identity and kernel metadata;
- a parent OPP root and a concrete vendor root are different path levels;
- a trailing colon produces an empty path entry and must not be treated as a
  valid vendor;
- different vendors can ship the same `libcust_opapi.so` SONAME, so resolve by
  an absolute vendor path or the target's explicit loader instead of assuming
  one global `-lcust_opapi` is unambiguous.

Verify dependencies and exported symbols read-only before building. Never copy
random libraries or headers into a vendor to make discovery pass; rebuild or
repair the responsible model artifact instead.

## Semantic lessons

The case preserved mutation semantics for gradient descent by returning a
separate NPU result rather than silently mutating an unrelated Python object.
It also translated a public reduction string only at the Python/API boundary,
then passed the exact integer attribute expected by ACLNN. Apply the same rule:
public NumPy-compatible behavior is authoritative, while ABI-specific values
remain localized to the integration wrapper.
