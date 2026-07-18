---
name: numpilot-migration-gate
description: Run NumPilot's gated NumPy-to-AsNumpy/NPU migration workflow. Use when analyzing, migrating, validating, or benchmarking a scientific Python network on Ascend NPU.
---

# NumPilot Migration Gate

## Inputs

- source Python file exposing `build_inputs()` and `forward()`;
- output directory dedicated to this run;
- requested backend (`shim` or `npu`);
- optional approved GapSpec when a capability is missing.

## Workflow

1. Read `docs/numpilot-technical-review.md` and `docs/codearts-integration.md`.
2. Run `python -m numpilot doctor --backend npu` before any bare-metal execution.
3. Analyze the source. Stop on unsupported operations; never silently replace them with NumPy.
4. Run a shim migration first:

   ```bash
   python -m numpilot migrate SOURCE --backend shim --out RUN_DIR/shim
   ```

5. Review the generated source and structured verification fields. The artifact must run its own
   `build_inputs()`, use the whole-forward NPU-island convention, pass the strict forward data-flow
   surface audit, and record actual `ap.*` callsite counts on both verification paths.
6. Run the real backend only when capability/provenance checks pass:

   ```bash
   python -m numpilot migrate SOURCE --backend npu --out RUN_DIR/npu
   ```

7. Establish a CPU baseline and then run strict NPU benchmarking in separate processes:

   ```bash
   python -m numpilot bench-cpu SOURCE --out RUN_DIR/bench --cpu-threads 1
   python -m numpilot bench-npu SOURCE --out RUN_DIR/bench
   ```

8. If strict NPU reports CPU fallback or unknown provenance, treat performance as invalid. The
   escape hatch `--allow-cpu-fallback` is diagnostic only and must not produce an NPU speedup claim.
9. Ask `numpilot-evidence-auditor` to independently re-run and audit the evidence.

## Gap handling

For a missing capability, produce a GapSpec and route it in this order: existing ACLNN, composition,
custom Ascend C operator, explicit CPU fallback, unsupported. Only the first three can become strict
NPU capabilities, and only after semantic and performance gates pass.

## Required artifacts

- migrated source and diff;
- verification report including failure path/index or exception phase;
- capability and backend provenance;
- benchmark JSON/Markdown with `performance_valid`;
- build/test/profiler evidence for newly generated operators;
- exact reproduction commands and known limitations.
