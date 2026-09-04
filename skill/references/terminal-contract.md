# Persistent terminal contract

Read this reference when operating or changing NumPilot's human terminal. The interactive experience
is framework behavior, not a recording-only wrapper.

## Start and commands

In a two-way TTY, bare `numpilot` opens a persistent shell and renders the NumPilot product mark. It
must remain usable after a command succeeds, fails, or is cancelled.

```text
numpilot › check llm
numpilot › migrate
numpilot › help
numpilot › exit
```

- `check llm` performs one real, bounded request using the selected private `llm.json`. A non-empty,
  valid reply establishes endpoint usability for the current check; it does not prove migration or
  operator quality. Print only a redacted model identity and pass/fail result.
- `migrate` routes to the same framework-owned interactive migration used by the lower-level
  `interactive-migrate` command. It enables full public CANNBot Skill display when an eligible gap is
  reached, but it never silently accepts a license or invents a missing configuration.
- Other documented CLI subcommands may be dispatched through the shell without duplicating their
  implementation in the REPL.

## Configured interactive migration

The visible order is fixed across ordinary programs:

1. Ask for a stable ordinary Python file by absolute path, then bind path, SHA-256, bytes, and lines.
2. Run inert static analysis and display every NumPy API/callsite with its current AsNumpy decision.
3. Display the configured model's bounded interpretation as `interpreted_unverified`.
4. Ask `y/N`. The prompt must disclose the NPU work and, when a CANNBot route is pending, that `y`
   also accepts the pinned CANNBot/CANN license for this invocation. Immediately before any original
   or normalized user script is executed, require a separate exact `execute` confirmation. Automated
   runs require the equivalent dedicated flags; `--non-interactive` alone is not consent.
5. No gap: run `01-analyze → 02-migrate → 03-verify → 04-benchmark` as applicable.
6. Gap: run the official-first gap branch and continue only if that branch publishes an available,
   qualified interface. A generated file by itself cannot open the migration gate.
7. Write one private session directory containing the interpretation, gap, migration, verification,
   benchmark, inventory, and final `migration-report.json` plus `migration-report.md` receipts that
   actually exist. Show absolute report and migrated-source paths at the end.

The configured interpretation is explanatory only. Do not reuse its prose as migration source or
persist the response body in the report. Cancellation or EOF retains a declined receipt and keeps
all executable stages `not_run`.

## Registered add-one branch

The sole registered signature-gap demo recognizes exact direct calls shaped as
`numpy.add(array, 1)` or `numpy.add(array, 1.0)`. The screen should explain that the missing contract
is `asnumpy.add_one(x)`, `float32`, shape-preserving `y = x + 1`.

After confirmation, show each framework stage:

1. official binary/source resolution and its receipt;
2. pinned CANNBot bundle identity, every public Skill file embedded in the request, and a periodic
   elapsed-time heartbeat during the real model call;
3. the generated Ascend C `Adds` fragment and status `generated_unverified`;
4. separate materialization and real-NPU qualification of a run-scoped `asnumpy.add_one` composition
   built from qualified `asnumpy.ones_like` plus `asnumpy.add`;
5. exact scalar-one lowering, a separate source-execution confirmation, ordinary migration, numeric
   verification, and benchmark.

Never say the generated Ascend C fragment was built or activated. The qualified interface has
`interface_provenance=composite_npu`, is visible only within the run, and does not modify upstream
AsNumpy or a shared OPP. If its isolated NPU qualification fails, stop before whole-program migration.
For any unregistered gap, `generated_unverified` is terminal until a generic integration path exists.

## Direct and non-TTY entry

Automation can call the same framework boundary with explicit consent and a fresh output directory:

```text
python -m numpilot --llm-config /absolute/private/llm.json \
  interactive-migrate --source /absolute/source.py --yes --execute-source \
  --cannbot-gap-draft --accept-cannbot-license --show-cannbot-skill full
```

Do not use `--yes` or `--execute-source` in a human recording; let the operator type `y` and then
`execute` at their distinct prompts. In automation, `--yes` does not imply source execution;
`--execute-source` is additionally required. Low-level manual workflows can omit `llm.json`, but they
do not claim the product configured-model experience.

## Model-source gate

If the user's opening instruction says to read the Skill, use agent mode, or avoid `llm.json`, stop
this configured lane before reading or probing any model configuration. The interacting AI performs
interpretation, completion, operator/integration work, and migration authoring itself, then invokes
only deterministic NumPilot commands whose input contract accepts its artifact. Keep
`agent_authored_draft` provenance; absent a registered adapter, report `migrate=draft/stopped`,
`verify=not_run`, and `benchmark=not_run`.

## Private receipt and display rules

Keep the session directory mode `0700` and regular files private. Bind source, model request/response,
Skill bundle, generated artifacts, qualification, migrated source, and final report with hashes. A
public summary may use portable paths, but the private report should retain absolute paths needed for
reproduction.

Never display or publish tokens, private endpoint URLs, proxy URLs, full request/response bodies,
environment dumps, or terminal control characters. Proxy variables are optional site-specific
settings supplied outside NumPilot; the only framework requirement is that `check llm` receives a
usable reply.

Required non-claims:

- source interpretation is not execution or verification;
- `generated_unverified` is not a compiled or activated operator;
- run-scoped composition qualification is not an upstream AsNumpy change;
- numeric agreement is not strict-NPU performance proof;
- resident timing is not cold-start end-to-end latency;
- one measured size does not establish a general crossover threshold.
