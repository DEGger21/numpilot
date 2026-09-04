# Changelog

Notable user-facing changes to NumPilot are recorded here.

## 0.3.0a1.dev0 - Unreleased

### Added

- Persistent interactive terminal with explicit model-check, generation, and source-execution approvals.
- Repository-owned Agent Skill workflow for sidebar AI use without `llm.json`.
- Registered `add_one` API-gap route with separated operator draft, run-scoped interface qualification,
  migration, numerical verification, and performance states.
- Hash-bound internal AsNumpy component, SBOM, bootstrap receipt, runtime identity, and real-NPU gates.
- Ordinary single-file analysis, configured-model migration, isolated numerical comparison, and scope-bound
  strict-NPU benchmarking with JSON/Markdown reports.

### Changed

- Reworked the English and Chinese READMEs around installation, installed contents, `llm.json`, and the two
  supported model-source workflows.
- Added `install.sh --python`; frozen-component installation now checks the Python ABI and architecture before
  pip, then pins NumPy and Loguru to the exact component-manifest versions.
- Converted the repository audit to a public-product release tree and removed event-specific documents,
  notebooks, historical run evidence, presentation decks, and their dedicated tooling.
- Kept generated candidates fail-closed until their applicable provenance, structure, numerical, runtime, and
  performance gates pass.

### Security

- Private model configuration remains outside the repository with directory mode `0700` and file mode `0600`.
- Optional proxy enforcement, bounded model transport, write-once receipts, hash-bound execution approvals,
  and symlink/path checks remain enabled.

## 0.2.0a2 - 2026-08-25

- Added the explicit source installer, persistent activation environment, pinned component bootstrap, and
  CANNBot license gate.
- Added exact runtime capability identities and environment-isolated installation checks.

## 0.2.0a1 - 2026-08-25

- Introduced ordinary NumPy script analysis, configured-model migration, API-gap recovery, numerical
  verification, and strict-NPU benchmark contracts.
- Moved generated reports, caches, and runtime state outside the Git working tree.

## 0.1.0

- Initial NumPilot CLI, strategy-based migration engine, shim/NPU backends, reports, and tests.
