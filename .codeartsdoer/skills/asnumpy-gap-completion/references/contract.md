# AsNumpy gap completion contract

This reference defines the deterministic handoff between gap analysis, CANNBot,
AsNumpy integration, and qualification.

## Manifest identity

The adapter manifest is `asnumpy.adapter.manifest.v1`. Its operator identity is
`operator.<symbol>` -> `asnumpy.<symbol>`. The Python binding is incremental:
`asnumpy._core` with a `custom_ops` submodule. A legacy `_asnumpy` extension or a
full module rebuild is rejected.

The ACLNN mapping contains:

```json
{
  "call_style": "two-stage",
  "get_workspace_size": "aclnn<Name>GetWorkspaceSize",
  "execute": "aclnn<Name>",
  "symbol_resolution": {
    "strategy": "explicit-handle-dlsym",
    "library_soname": "libcust_opapi.so",
    "global_symbol_call": false
  }
}
```

The logical ACLNN names remain in the manifest for audit and header checks. The
wrapper obtains function pointers from the custom DSO handle. This is required
because CANN's official `libopapi.so` may export a same-named API with a
 different ABI.

## Overlay and provenance

The overlay is limited to the target's declared include/C++/binding/Python/CMake
markers and its conformance test. Custom OPP provenance covers `version.info`,
`op_api/include/aclnn_<name>.h`, and `op_api/lib/libcust_opapi.so`; each file is
read-only hashed. The provenance checker never loads or executes the library.

## Qualification evidence

A qualification receipt must identify the GapSpec, candidate, source/component
identity, custom OPP receipt, wheel SHA, runtime/SoC, input profile, NumPy oracle,
NPU execution source, output error, repeated-run result, and performance protocol.
It must state whether the formal component changed. Minimal isolated success does
not authorize activation.
