# Installed official ACLNN existing-module integration

Use this reference only when the signed adapter contract declares
`strategy=official_aclnn_existing_module` and the candidate origin is
`official_binary`.

The kernel already exists in the installed CANN toolkit. Do not generate an
Ascend C kernel, create a custom OPP, add `cust_opapi`, invent a vendor, or
invoke a private operator loader. The task is to connect the admitted public
AsNumpy profile to the existing official ACLNN call chain.

## Inputs are authoritative

Treat these signed values as immutable:

- candidate ID and `official_binary` origin;
- GapSpec API, signature variants, dtype/shape rules and NumPy oracle;
- selected CANN header/library hashes and required exported symbols;
- the reviewed target-link proof showing that the selected library is the
  exact DSO named by the AsNumpy build graph; another DSO exporting the same
  symbols is not interchangeable;
- catalog semantic-review restrictions;
- target SoC and CANN version;
- the existing-module context hashes and exact overlay allow-list.

Stop if they conflict. Never broaden the admitted dtype, rank, shape,
broadcast, keyword, empty-input, or layout profile. Do not claim that a header
or symbol match proves semantic correctness.

## Existing-module overlay rules

Read every supplied AsNumpy reference file before producing a replacement.
Return exactly the requested complete file and no other path. Preserve public
behavior outside the admitted profile.

For the reviewed `numpy.matmul`/`asnumpy.matmul` plan:

- the reviewed public `ndarray` surface provides `ndim`, `shape`, `dtype`,
  `itemsize`, and byte-valued `strides`; do not invent NumPy's `.flags` API;
- the native C++ implementation is in `csrc/linalg/product.cpp` and must call
  the selected `aclnnMatmulGetWorkspaceSize`/`aclnnMatmul` ABI;
- use `cubeMathType=0` (`KEEP_DTYPE`) for the admitted `float32` profile;
- `src/asnumpy/linalg/direct.py` may select the native call only for two
  AsNumpy ndarrays that meet the exact admitted rank/dtype/inner-dimension
  contract passed as exactly two positional arguments; keyword calls and every
  other supported public case must preserve the pre-existing compatibility
  behavior;
- preserve Python's existing two-argument binding behavior without guessing
  how already-bound parameters were supplied: keep a private host-fallback
  function with the original `(x1, x2)` signature, and let a public
  `*args, **kwargs` wrapper select native only when `len(args) == 2` and
  `kwargs` is empty; delegate every other form to the private function and use
  `functools.wraps` so public signature introspection remains compatible;
- the public test in
  `tests/asnumpy_tests/linalg_tests/test_product.py` must prove that admitted
  inputs do not call the host conversion fallback and must compare real NPU
  output, dtype, and shape with NumPy;
- the boundary tests must prove that 1-D, batched/broadcast, mixed-dtype,
  empty, non-C-contiguous, and keyword forms do not call the native binding;
- do not edit bindings or CMake when the signed library-binding proof already
  matches the selected DSO and exposes the native `matmul` call. Stop on a
  mismatch; do not add another library with overlapping ACLNN symbols or make
  unrelated build-option changes.

An empty dimension is outside the current non-empty qualification profile.
Do not silently admit it because NumPy can evaluate it.

## Output and claims

The overlay is `generated_unverified`. Do not claim compilation, wheel
installation, NPU execution, numerical correctness, activation, or performance.
A later isolated integration transaction, build, strict-NPU qualification,
activation, capability rescan, and final script verification must prove those
stages independently.
