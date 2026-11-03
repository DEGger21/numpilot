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

Read every supplied AsNumpy reference file before proposing edits. Return
exactly the requested `NUMPILOT_EXACT_EDIT_V1` response for the requested file
and no other path. Each `OLD` block must be copied verbatim from one supplied
target edit window. Preserve public behavior outside the admitted profile.

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

For the reviewed `numpy.concatenate`/`asnumpy.concatenate` plan:

- the admitted form is a non-empty sequence of AsNumpy `ndarray` inputs with an
  explicit integer `axis` (the default `axis=0` is an integer). `axis=None`
  and the `dtype`, `out`, `casting`, and `order` keywords are outside the
  reviewed mapping; reject them rather than guessing a host fallback;
- `include/asnumpy/array/basic.hpp` declares the new existing-module entry
  point and `csrc/array/basic.cpp` implements it with the signed
  `aclnn_cat.h` ABI (`aclnnCatGetWorkspaceSize`, `aclnnCat`);
- the signed `aclnn_cat.h` does not declare tensor-list helpers. Build the
  input list with
  `aclCreateTensorList(const aclTensor* const*, uint64_t)` and destroy it with
  `aclDestroyTensorList` on every success and failure path. Both are declared
  by `aclnn/acl_meta.h`, which `npu_array.hpp` already includes through
  `aclnn/aclnn_base.h`; do not redeclare them;
- normalize a negative `axis` with `axis += rank`. Require one shared rank,
  equal extents on every non-axis dimension, identical `float32` dtype, and
  C-contiguous inputs. Reject an empty sequence and any axis outside
  `[-rank, rank)`. Allocate the output `NPUArray` with the explicitly computed
  output shape and the input dtype before calling ACLNN;
- `bindings/python/bind_array.cpp` must expose the same argument order and
  types as the C++ entry point. `src/asnumpy/array.py` must keep the public
  wrapper narrow (`concatenate(arrays, axis=0)`), pass the sequence through
  unchanged, and wrap the native result in the public `ndarray` type.
  `src/asnumpy/__init__.py` adds only the lazy export;
- preserve the existing `Zeros`/`Ones`/`Linspace` behavior and the reviewed
  CMake link items; do not change unrelated creation functions or add a DSO;
- tests in `tests/asnumpy_tests/creation_tests/test_basic.py` must compare NPU
  output, dtype, and shape with `numpy.concatenate` for admitted float32 cases,
  including a negative axis and a single-input sequence, and must prove that
  empty sequences, `axis=None`, mismatched non-axis extents, mixed dtype, and
  non-contiguous forms do not enter the native path.

An empty dimension is outside the current non-empty qualification profile.
Do not silently admit it because NumPy can evaluate it.

## Output and claims

The overlay is `generated_unverified`. Do not claim compilation, wheel
installation, NPU execution, numerical correctness, activation, or performance.
A later isolated integration transaction, build, strict-NPU qualification,
activation, capability rescan, and final script verification must prove those
stages independently.
