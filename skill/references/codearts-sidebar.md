# CodeArts sidebar adapter

This reference describes the conversational adapter around the deterministic NumPilot CLI. It is
intended for a CodeArts project skill, Agent, or AgentTeam teammate; it does not grant extra shell,
network, model, NPU, or installation permissions. It supports the terminal/configured and
AI/CodeArts-agent model sources, with an optional hybrid handoff; select one before touching model
configuration.

The adapter has no independent execution engine. Its canonical calls are analyze → migrate → verify → replay; each call targets the same private run directory and receipt schema used by the terminal CLI. The entry fields are entry_lane and model_source, while the CLI remains responsible for all deterministic gates.

## Recommended interaction

Use a short, explicit request such as:

```text
请使用 $numpilot-migration-gate 迁移 <source.py>。
先做只读五态分析；在我确认后再进行远程模型调用、脚本执行和性能测试。
请逐阶段展示状态、回执路径、SHA-256、运行时身份和不能外推的结论。
```

当用户希望完全由 CodeArts 交互、明确不使用 NumPilot 内置模型时，可直接使用下面的 no-llm
指令。它同时声明了模型配置的硬边界和没有适配器时的停止状态：

```text
请阅读 $numpilot-migration-gate 和 NumPilot 仓库，使用 agent 模式，明确不用 llm.json 和
NumPilot 内置模型。由你与我交互完成源码解读、缺口补全、算子生成、接入和迁移候选，再执行
框架支持的确定性验证；如果没有受支持的 migration-report 适配器，就保留
migrate=draft/stopped、verify=not_run、benchmark=not_run，报告回执哈希和不能外推的结论。
```

The sidebar agent should then:

1. Resolve the source and a fresh private run directory. Show the planned commands without printing
   credentials or a full model request. State `entry_lane` and `model_source` in the handoff.
2. Read the repository Skill and relevant docs, then run `analyze`. Keep the exact five-state decision
   (`DIRECT`, `FALLBACK_CONSENT`, `GAP_REQUIRED`, `CAPABILITY_AUDIT_REQUIRED`, or `BLOCKED`).
3. Pause at the authorization boundaries listed in the parent `SKILL.md`. If the user declines a
   boundary, publish the completed receipt and stop; do not reinterpret the decline as a gap or a
   successful fallback.
4. In the configured terminal sub-run, invoke `migrate`. In CodeArts-agent or hybrid draft mode,
   first ask whether a registered adapter accepts the candidate and can materialize the supported,
   hash-bound ordinary migration report. The current CLI has no generic adapter for arbitrary
   hand-authored ordinary files: when none is registered, record `migrate=draft/stopped`, leave
   `verify=not_run` and `benchmark=not_run`, and never synthesize or hand-edit a report. Invoke
   `verify` only for an adapter-accepted report, and invoke `benchmark-script` only when that
   report is fully qualified and the user requests it. Stream concise stage events, but use JSON
   receipts as the authority.
5. If the source uses the registered exact `numpy.add(array, 1)` form, preserve the same claim split
   as the terminal: an agent-authored Ascend C fragment remains an uncompiled/unactivated draft, while
   any run-scoped `asnumpy.add_one` composition must pass the framework's separate real-NPU
   qualification before it is described as available. Do not generalize this adapter to other gaps.
6. End with a compact handoff containing the stage table, absolute artifact paths, hashes, and
   explicit non-claims. Link the actual CodeArts task/event record separately when proving that the
   sidebar loaded or executed this skill.

## Control plane versus execution plane

CodeArts supplies conversation, task decomposition, approvals, and optional collaboration. NumPilot
supplies static analysis, isolated execution, capability provenance, numeric checks, and performance
gates. A CodeArts response that contains plausible code is not by itself a NumPilot migration
artifact.

Select one entry/model source before reading any configuration:

* **Terminal/manual:** the user operates the shell and may fill a private `llm.json`, but it is
  optional. A model is allowed only for an explicitly authorized migration, adapter, or operator
  generation boundary; analysis and evidence gates stay deterministic.
* **CodeArts-agent:** the user asks for the sidebar/interactive model (or says not to use `llm.json`).
  This is a hard no-read/no-call boundary: do not inspect `llm.json`, invoke `check llm` or
  `configure-llm`, pass `--provider llm`, or probe an endpoint. Let the agent read the repository and
  Skill; the interacting model itself handles interpretation, completion, operator/integration work,
  and migration authoring in an isolated directory, labels it `agent_authored_draft`, and runs only
  deterministic NumPilot gates whose input contract is available. The initial Skill-read/handoff
  receipt itself must remain `candidate_status=not_generated` with `agent_authored_draft=false`; only
  a later receipt bound to the actual candidate may use the draft label. Until a supported adapter binds
  the exact agent/provider/model/response and source/artifact hashes, the result remains a draft and
  cannot carry configured-model, strict-NPU, or performance claims. The agent model's provenance is
  a separate `codearts_agent_receipt`, not a configured-model receipt; if no ordinary-report adapter
  is available, stop before `verify`/`benchmark-script`.
* **Hybrid:** CodeArts plans/reviews or authors the draft, while a terminal invokes deterministic
  `analyze`/hash/verify/benchmark commands. Keep CodeArts task/event evidence and candidate labels
  separate. A configured-model sub-run is permitted only after an explicit switch, authorization, and
  fresh run directory.

This explicit split lets CodeArts be the sole conversational model without silently mixing a second
model's provenance. Switching lanes requires a new run directory and fresh authorization. If the
user's request does not identify a lane, ask before touching model configuration. A proxy URL is only
an optional process environment setting supplied by the user; do not hard-code it in the Skill or
sidebar transcript. The framework requirement is simply that the chosen model returns usable content.

## Suggested final handoff

```text
NumPilot migration: <PASS|STOPPED>
Entry lane / model source: <terminal_manual|codearts_agent|hybrid> / <none|configured|codearts_agent|user_authored>
Decision: <five-state value>
Stages: analyze=<status> · migrate=<status|draft/stopped> · verify=<status|not_run> · benchmark=<status|not_run>
Model/runtime: <recorded identities, or unavailable>
Artifacts: <run-dir and report paths; SHA-256 values>
Claims: numeric=<true|false> · strict_npu=<true|false> · performance_valid=<true|false>
Candidate status: <not_generated|draft|admitted>
Draft label: <none|agent_authored_draft|user_authored_draft|configured_model_artifact>
Non-claims: <fallback, cold-start, transfer, profiler, scale, or CodeArts-product limits>
Next action: <smallest safe remediation or complete>
```

Use `model_source=codearts_agent` and emit a `codearts_agent_receipt` when the sidebar model
authored the candidate. Do not relabel it as `configured`, even after deterministic checks; numeric
consistency and model provenance are separate claims.

The sidebar adapter should treat the reference files as trusted, versioned instructions only after
checking each displayed reference is a regular non-symlink UTF-8 file within the size bound and its
SHA-256 matches the private skill receipt. If a reference is missing, drifting, or unverified, stop
the corresponding lane rather than loading a replacement from an arbitrary path.

Do not paste `llm.json`, tokens, private endpoint URLs, raw response bodies, or unredacted shell
environment into the sidebar transcript or a public issue.
