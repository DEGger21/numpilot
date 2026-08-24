# Manifest-driven AsNumpy workflow

Use this reference when the target AsNumpy checkout contains
`scripts/gen_custom_ops.py` and `csrc/custom_ops/registry/`.

## Read the target contract first

Read these files from the target checkout rather than relying on a memorized
schema:

- `scripts/gen_custom_ops.py`;
- one enabled `csrc/custom_ops/registry/<reference>/op.yaml`;
- its test under `tests/asnumpy_tests/custom_ops_tests/`;
- `cmake/CustomOps.cmake` and `csrc/custom_ops/CMakeLists.txt`;
- `include/asnumpy/custom_ops/loader.hpp` and
  `csrc/custom_ops/loader.cpp`.

The generator is authoritative. Reject unknown fields, unsupported operation
types, or paths outside the target's declared schema.

## Minimal overlay

For a tensor unary/binary operator supported by the target generator, the model
normally returns exactly:

```text
csrc/custom_ops/registry/<python_name>/op.yaml
tests/asnumpy_tests/custom_ops_tests/test_<python_name>.py
```

Do not return or edit generator-owned outputs such as:

```text
csrc/custom_ops/generated/*
bindings/python/generated/*
src/asnumpy/custom_ops.py
src/asnumpy/_custom_ops_registry.py
```

The orchestrator runs the generator after applying the overlay and verifies
`python scripts/gen_custom_ops.py --check`.

## Manifest decisions

Common fields in the current example architecture are:

```yaml
name: example
op_type: unary
cpp_name: Example
opp_vendor: exact_installed_vendor_directory
enabled: true
aclnn_header: aclnn_example.h
aclnn_get_workspace: aclnnExampleGetWorkspaceSize
aclnn_execute: aclnnExample
out_dtype: preserve
top_level_export: true
doc: "Public API behavior."
```

Treat this as a shape example, not a substitute for reading the target
generator. Every value must be grounded:

- `name` and public export come from the adapter contract;
- `opp_vendor` is the exact directory under an accepted OPP entry, commonly
  `<opp-root>/vendors/<vendor>`. It is not the operator name unless the
  installed directory proves that;
- header and two-stage ACLNN symbols must exist in the private OPP. Inspect the
  header and dynamic symbol table; do not infer spelling;
- use generated `unary`/`binary` only if the target generator's emitted call
  signature matches the operator ABI and output shape/dtype contract;
- use the target's `manual` mechanism for attributes, multiple outputs,
  reductions, dimension changes, mutation, workspace differences, or any ABI
  the unary/binary renderer cannot express;
- admit only dtypes and shapes allowed by the signed operator artifact.

The manifest vendor participates in both build-time discovery and runtime
`dlopen`/symbol resolution. The same name must resolve in both places.

## Test contract

Test the public `asnumpy.<name>` path as well as any lower-level custom-op path
needed for diagnosis. Compare real NPU results with NumPy for:

- ordinary positive and negative values;
- zero and semantic edge values defined by NumPy;
- multiple admitted shapes, including non-flat arrays;
- every admitted dtype;
- output dtype and shape;
- tolerances justified by the operator algorithm and dtype.

Tests must not add a CPU implementation, catch and hide runtime errors, accept
missing custom operators, or turn absence into a passing skip. Environment-wide
tests may use explicit skips for unrelated hardware, but the qualification run
for this operator must execute and pass on NPU.
