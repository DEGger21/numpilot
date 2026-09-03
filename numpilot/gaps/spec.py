"""Turn static-analysis gaps into deterministic operator specifications."""

import ast
import hashlib
import json
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from numpilot.contracts import AnalysisResult, OpUse
from numpilot.gaps.models import GapCallsite, GapSpec
from numpilot.operator_specs import OperatorSpec, collect_static_operator_specs


MISSING_API_REASON = "API 不在静态覆盖集"
_SAFE_COMPONENT = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def is_generatable_gap(op: OpUse) -> bool:
    """Only a genuinely absent API can enter operator generation."""
    if op.kind == "domain_operator":
        if not op.supported or not op.requires_qualification:
            return False
        try:
            spec = OperatorSpec.from_dict(op.operator_spec or {})
        except (TypeError, ValueError):
            return False
        return (
            op.name == spec.name
            and op.qualified_name == spec.qualified_api
            and op.target_api == spec.target_api
        )
    if op.supported or op.reason != MISSING_API_REASON:
        return False
    qualified = op.qualified_name or ""
    if not qualified.startswith("numpy."):
        return False
    components = qualified.split(".")[1:]
    if len(components) != 1:
        return False
    return all(
        _SAFE_COMPONENT.fullmatch(component)
        for component in components
    )


def _signature_variants(uses: Iterable[OpUse]) -> Tuple[Dict[str, object], ...]:
    variants = {
        (
            int(op.positional_args),
            tuple(str(item) for item in op.keyword_args),
        )
        for op in uses
    }
    return tuple(
        {
            "positional_args": positional,
            "keyword_args": list(keywords),
        }
        for positional, keywords in sorted(variants)
    )


def _gap_id(
    qualified_api: str,
    signatures: Tuple[Dict[str, object], ...],
    target_soc: str,
    cann_version: str,
) -> str:
    identity = json.dumps(
        {
            "qualified_api": qualified_api,
            "signature_variants": signatures,
            "target_soc": target_soc,
            "cann_version": cann_version,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    stem = qualified_api.split(".", 1)[-1].replace(".", "_")
    return "%s-%s" % (stem, hashlib.sha256(identity).hexdigest()[:16])


def gap_spec_identity_matches(spec: GapSpec) -> bool:
    """Return whether a serialized spec still has its deterministic identity."""

    return spec.gap_id == _gap_id(
        spec.qualified_api,
        tuple(dict(item) for item in spec.signature_variants),
        spec.target_soc,
        spec.cann_version,
    )


def is_custom_generation_operator_gap(source: str, spec: GapSpec) -> bool:
    """Return whether ``spec`` is an explicit, opt-in custom OperatorSpec.

    A ``GapSpec`` is normally produced by :func:`build_gap_specs`, but route
    receipts are persisted and may be supplied by another process.  Checking
    only the serialized route would therefore make it possible to relabel an
    ordinary/unknown gap as a custom operator.  Re-parse the inert source AST
    and require the exact decorator provenance and route as an additional
    identity check.  No user code is imported or executed here.
    """

    if not isinstance(source, str) or not isinstance(spec, GapSpec):
        return False
    if (
        not spec.qualified_api.startswith("operator.")
        or spec.route != "ascendc_registry_candidate"
        or not spec.semantic_source.startswith("operator-spec:")
        or spec.asnumpy_api != "asnumpy." + spec.api_name
    ):
        return False
    # ``build_gap_specs`` records these states in the canonical route trace;
    # retain the check so a manually assembled GapSpec cannot opt in by merely
    # changing its top-level route.
    states = {
        (str(item.get("route", "")), str(item.get("status", "")))
        for item in spec.route_decisions
        if isinstance(item, Mapping)
    }
    if (
        ("ascendc_registry_candidate", "selected_for_qualification") not in states
        or ("direct_generation", "custom_generation_candidate") not in states
    ):
        return False
    try:
        tree = ast.parse(source)
        static_specs = collect_static_operator_specs(tree)
    except (SyntaxError, TypeError, ValueError, MemoryError, RecursionError):
        return False
    for operator in static_specs.specs.values():
        if (
            operator.name == spec.api_name
            and operator.qualified_api == spec.qualified_api
            and operator.target_api == spec.asnumpy_api
            and operator.target_soc == spec.target_soc
            and operator.cann_version == spec.cann_version
            and operator.provenance == "custom_generation_candidate"
            and operator.route == "ascendc_registry_candidate"
        ):
            return True
    return False


_SCRIPT_REPLACEABLE_PROVENANCE = frozenset({
    "cpu_fallback",
    "conditional_cpu_fallback",
    "host_assisted",
})


def build_script_gap_specs(
    source: str,
    source_path: str,
    decisions: Sequence[Mapping[str, object]],
    *,
    target_soc: str,
    cann_version: str,
) -> Tuple[GapSpec, ...]:
    """Build bound GapSpecs for missing or explicitly rejected script APIs.

    Ordinary-script decisions are accepted only for direct ``numpy.<api>``
    calls.  An existing callable can enter this lane only when its provenance
    is an audited fallback class that the user explicitly rejected.  Unknown
    callable provenance therefore cannot be turned into generation work.
    """

    if not target_soc.strip() or not cann_version.strip():
        raise ValueError("script gap target and CANN version must be non-empty")
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    specs: List[GapSpec] = []
    seen = set()
    for decision in decisions:
        api_path = str(decision.get("api_path", ""))
        qualified_api = str(decision.get("qualified_api", ""))
        kind = str(decision.get("kind", ""))
        status = str(decision.get("status", ""))
        provenance = str(decision.get("provenance", ""))
        available = decision.get("available")
        if (
            kind != "call"
            or not _SAFE_COMPONENT.fullmatch(api_path)
            or qualified_api != "numpy." + api_path
            or decision.get("generatable") is not True
        ):
            raise ValueError(
                "script gap must be an explicitly generatable top-level NumPy call"
            )
        if qualified_api in seen:
            raise ValueError("duplicate script gap decision for %s" % qualified_api)
        seen.add(qualified_api)

        replacement = status == "fallback_available"
        missing = (
            status == "unsupported"
            and available is False
            and provenance == "unknown"
        )
        if replacement:
            if available is not True or provenance not in _SCRIPT_REPLACEABLE_PROVENANCE:
                raise ValueError(
                    "only an audited available fallback can become a replacement gap"
                )
            callsite_reason = (
                "user rejected audited AsNumpy %s; a separate NPU runtime must "
                "be generated, integrated, and qualified" % provenance
            )
            route_decisions = (
                {
                    "route": "existing_asnumpy_fallback",
                    "status": "user_rejected",
                    "reason": provenance,
                },
                {
                    "route": "replacement_runtime",
                    "status": "required_before_qualification",
                    "reason": (
                        "integrate a new/private AsNumpy callable; do not relabel "
                        "or overwrite the audited fallback runtime"
                    ),
                },
                {
                    "route": "ascendc_registry",
                    "status": "candidate_for_draft",
                    "reason": (
                        "the configured model may prepare an isolated draft; execution and activation "
                        "remain gated"
                    ),
                },
            )
        elif missing:
            callsite_reason = "AsNumpy runtime does not expose this NumPy API"
            route_decisions = (
                {
                    "route": "asnumpy_runtime",
                    "status": "missing",
                    "reason": "audited AsNumpy runtime did not expose this API",
                },
                {
                    "route": "replacement_runtime",
                    "status": "required_before_qualification",
                    "reason": "integrate a new/private AsNumpy callable",
                },
                {
                    "route": "ascendc_registry",
                    "status": "candidate_for_draft",
                    "reason": (
                        "the configured model may prepare an isolated draft; execution and activation "
                        "remain gated"
                    ),
                },
            )
        else:
            raise ValueError(
                "script gap is neither a missing API nor a user-rejected audited fallback"
            )

        raw_callsites = decision.get("callsites")
        if not isinstance(raw_callsites, (list, tuple)) or not raw_callsites:
            raise ValueError("script gap requires at least one callsite")
        callsites = []
        signature_keys = set()
        for raw in raw_callsites:
            if not isinstance(raw, Mapping):
                raise ValueError("script gap callsite must be an object")
            if (
                str(raw.get("api_path", "")) != api_path
                or str(raw.get("qualified_api", "")) != qualified_api
                or str(raw.get("kind", "")) != "call"
            ):
                raise ValueError("script gap callsite identity is inconsistent")
            try:
                lineno = int(raw["lineno"])
                col_offset = int(raw.get("col_offset", 0))
                positional_args = int(raw.get("positional_args", 0))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("script gap callsite location/signature is invalid") from exc
            keywords_raw = raw.get("keyword_args", ())
            if not isinstance(keywords_raw, (list, tuple)):
                raise ValueError("script gap keyword signature is invalid")
            keywords = tuple(str(item) for item in keywords_raw)
            if (
                lineno < 1
                or col_offset < 0
                or positional_args < 0
                or positional_args > 64
                or len(keywords) > 64
                or any(len(item) > 128 for item in keywords)
            ):
                raise ValueError("script gap callsite location/signature is invalid")
            callsites.append(GapCallsite(
                lineno=lineno,
                col_offset=col_offset,
                qualified_api=qualified_api,
                call_expr=qualified_api + "(...)",
                reason=callsite_reason,
            ))
            signature_keys.add((positional_args, keywords))
        signatures = tuple(
            {
                "positional_args": positional,
                "keyword_args": list(keywords),
            }
            for positional, keywords in sorted(signature_keys)
        )
        specs.append(GapSpec(
            gap_id=_gap_id(
                qualified_api, signatures, target_soc, cann_version
            ),
            api_name=api_path,
            qualified_api=qualified_api,
            asnumpy_api="asnumpy." + api_path,
            source_path=source_path,
            source_sha256=source_hash,
            callsites=tuple(callsites),
            signature_variants=signatures,
            target_soc=target_soc,
            cann_version=cann_version,
            route="ascendc_registry_candidate",
            route_decisions=route_decisions,
            shape_rules=("runtime_profile_required",),
            dtype_rules=("runtime_profile_required",),
            semantic_source=qualified_api,
        ))
    return tuple(specs)


def build_gap_specs(
    source: str,
    analysis: AnalysisResult,
    *,
    target_soc: str,
    cann_version: str,
) -> Tuple[GapSpec, ...]:
    """Aggregate eligible callsites without executing untrusted source code."""
    grouped: Dict[str, List[OpUse]] = {}
    for op in analysis.op_uses:
        if is_generatable_gap(op):
            grouped.setdefault(op.qualified_name or "", []).append(op)

    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    specs = []
    for qualified_api, uses in sorted(grouped.items()):
        domain = uses[0].kind == "domain_operator"
        operator = (
            OperatorSpec.from_dict(uses[0].operator_spec or {})
            if domain else None
        )
        if domain:
            if any(item.operator_spec != uses[0].operator_spec for item in uses):
                raise ValueError("conflicting OperatorSpec declarations for one API")
            if operator.target_soc != target_soc or operator.cann_version != cann_version:
                raise ValueError(
                    "OperatorSpec target %s/CANN %s does not match requested %s/CANN %s"
                    % (
                        operator.target_soc,
                        operator.cann_version,
                        target_soc,
                        cann_version,
                    )
                )
            signatures = tuple(dict(item) for item in operator.signature_variants)
            api_name = operator.name
            asnumpy_api = operator.target_api
            route = operator.route
            shape_rules = operator.shape_rules
            dtype_rules = operator.dtype_rules
            semantic_source = (
                "operator-spec:%s@%s;oracle=%s"
                % (operator.name, operator.version, operator.numpy_oracle)
            )
            if operator.route == "cann_existing_reuse_candidate":
                implementation_decision = {
                    "route": "source_reuse",
                    "status": operator.provenance,
                    "reason": operator.source,
                }
            elif operator.route == "ascendc_registry_candidate":
                implementation_decision = {
                    "route": "direct_generation",
                    "status": operator.provenance,
                    "reason": operator.source,
                }
            else:
                implementation_decision = {
                    "route": "asnumpy_composition",
                    "status": operator.provenance,
                    "reason": operator.source,
                }
            route_decisions = (
                {
                    "route": operator.route,
                    "status": "selected_for_qualification",
                    "reason": "explicit OperatorSpec route; execution remains gated",
                },
                implementation_decision,
                {
                    "route": "license_review",
                    "status": "declared",
                    "reason": operator.license,
                },
            )
        else:
            signatures = _signature_variants(uses)
            api_name = uses[0].name
            asnumpy_api = "asnumpy." + qualified_api.removeprefix("numpy.")
            route = "ascendc_registry_candidate"
            shape_rules = ("runtime_profile_required",)
            dtype_rules = ("runtime_profile_required",)
            semantic_source = qualified_api
            route_decisions = (
                {
                    "route": "asnumpy_composite",
                    "status": "not_evaluated",
                    "reason": "requires a semantic composition resolver",
                },
                {
                    "route": "cann_existing",
                    "status": "review_required",
                    "reason": "check a pinned CANN operator catalog before new kernel work",
                },
                {
                    "route": "ascendc_registry",
                    "status": "candidate_for_draft",
                    "reason": "explicit draft generation is allowed; execution remains gated",
                },
            )
        specs.append(GapSpec(
            gap_id=_gap_id(
                qualified_api, signatures, target_soc, cann_version
            ),
            api_name=api_name,
            qualified_api=qualified_api,
            asnumpy_api=asnumpy_api,
            source_path=analysis.source_path,
            source_sha256=source_hash,
            callsites=tuple(
                GapCallsite(
                    lineno=op.lineno,
                    col_offset=op.col_offset,
                    qualified_api=qualified_api,
                    call_expr=op.call_expr or qualified_api + "(...)",
                    reason=(
                        "explicit OperatorSpec requires six-gate qualification"
                        if domain else (op.reason or MISSING_API_REASON)
                    ),
                )
                for op in uses
            ),
            signature_variants=signatures,
            target_soc=target_soc,
            cann_version=cann_version,
            route=route,
            route_decisions=route_decisions,
            shape_rules=shape_rules,
            dtype_rules=dtype_rules,
            semantic_source=semantic_source,
        ))
    return tuple(specs)


def blocked_unsupported(analysis: AnalysisResult) -> Tuple[Dict[str, object], ...]:
    """Describe unsupported calls that must not be sent to code generation."""
    blocked = []
    for op in analysis.op_uses:
        if op.supported or is_generatable_gap(op):
            continue
        reason = op.reason or "unsupported"
        if (
            op.qualified_name
            and len(op.qualified_name.split(".")) > 2
            and reason == MISSING_API_REASON
        ):
            reason = (
                "嵌套 NumPy namespace 尚未进入补全：当前改写与调用审计"
                "只支持直接 numpy.<api>"
            )
        blocked.append({
            "name": op.name,
            "qualified_api": op.qualified_name,
            "lineno": op.lineno,
            "reason": reason,
        })
    return tuple(blocked)


def extract_semantic_contract(source: str, spec: GapSpec) -> str:
    """Extract the declared NumPy oracle without importing user source."""

    if not spec.semantic_source.startswith("operator-spec:"):
        return ""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError("cannot parse source for semantic contract") from exc
    static_specs = collect_static_operator_specs(tree)
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        operator = static_specs.specs.get(node.name)
        if operator is None or operator.name != spec.api_name:
            continue
        excerpt = ast.get_source_segment(source, node)
        if not isinstance(excerpt, str) or not excerpt.strip():
            raise ValueError("operator semantic contract source is unavailable")
        payload = excerpt.strip()
        if len(payload.encode("utf-8")) > 32 * 1024:
            raise ValueError("operator semantic contract exceeds the size limit")
        return payload
    raise ValueError("declared operator semantic contract was not found")


def render_generation_request(
    spec: GapSpec,
    *,
    skill_root: str = "",
    asnumpy_reference: str = "",
    skill_bundle=None,
    semantic_contract: str = "",
) -> str:
    skill = (
        skill_root.rstrip("/")
        + "/ops/ascendc-registry-invoke-template/SKILL.md"
        if skill_root else "ascendc-registry-invoke-template"
    )
    reference = asnumpy_reference or "numpyexample"
    reused = spec.route == "cann_existing_reuse_candidate"
    direct = spec.route == "ascendc_registry_candidate"
    if reused:
        title = "# Existing operator reuse and qualification request\n\n"
        route_note = (
            "- This spec selected pinned source reuse: adapt and qualify it; "
            "do not claim a newly generated kernel.\n"
        )
    elif direct:
        title = "# Direct local Ascend C operator generation request\n\n"
        route_note = (
            "- This spec selected direct local generation. Generate from the GapSpec, "
            "NumPy oracle contract, and embedded official Skill. Do not fetch, clone, "
            "or require an external operator implementation repository in this stage.\n"
        )
    else:
        title = "# AsNumpy operator completion request\n\n"
        route_note = "- Follow the explicit OperatorSpec route recorded below.\n"
    embedded_note = (
        "- The verified Skill files are embedded below; use their content, not "
        "merely the host path.\n"
        if skill_bundle is not None else
        "- The model cannot read those local paths; this embedded contract is authoritative.\n"
    )
    request = title + "".join((
        "This workspace is an unverified draft. Do not install or activate it "
        "until every gate in operator-artifact.json passes.\n\n",
        "## Required workflow\n\n",
        f"- Follow the registry-invoke skill: `{skill}`\n",
        f"- Use the AsNumpy ACLNN/pybind integration reference: `{reference}`\n",
        embedded_note,
        route_note,
        (
            "- This is an explicit custom OperatorSpec route: official semantic "
            "equivalence is not proven. Do not describe the draft as Huawei-official "
            "or as a validated replacement.\n"
            if any(
                isinstance(item, Mapping)
                and item.get("route") == "direct_generation"
                and item.get("status") == "custom_generation_candidate"
                for item in spec.route_decisions
            ) else ""
        ),
        "- Target only the SoC and CANN version recorded in gap-spec.json.\n",
        "- op_host must define the schema, shape/dtype inference, tiling registration, and tiling data.\n",
        "- op_kernel must use Init -> CopyIn -> Compute -> CopyOut, with arguments ordered as inputs, outputs, workspace, tiling.\n",
        "- op_api must implement the ACLNN GetWorkspaceSize/execute two-stage call and explicit synchronization/error handling.\n",
        "- Deliver isolated build files, NumPy golden tests, numerical checks, and an AsNumpy adapter with explicit Python/C++/ACLNN name mapping.\n",
        "- Public AsNumpy tests must call asnumpy.<api>, not only the private pybind module.\n",
        "- Required gates: build, unit test, NumPy numerical validation, explicit activation.\n",
        "- Never edit a system CANN installation, ~/.bashrc, or a shared AsNumpy checkout implicitly.\n\n",
        "## Gap\n\n",
        f"- API: `{spec.qualified_api}` → `{spec.asnumpy_api}`\n",
        f"- Gap ID: `{spec.gap_id}`\n",
        f"- Target: `{spec.target_soc}`, CANN `{spec.cann_version}`\n",
    ))
    if semantic_contract:
        request += (
            "\n## NumPy semantic oracle (authoritative)\n\n"
            "The following function was extracted statically from the analyzed source. "
            "Implement this exact numerical behavior for the admitted dtype/shape scope.\n\n"
            "```python\n" + semantic_contract + "\n```\n"
        )
    if skill_bundle is not None:
        request += "\n" + skill_bundle.prompt_context()
        safe_name = spec.api_name
        request += "".join((
            "\n## NumPilot remote response envelope (mandatory)\n\n",
            "The official Skill above defines engineering semantics. Return a complete, "
            "self-contained, build-ready local candidate. It remains unverified until "
            "the local qualification gates pass.\n\n",
            "- NumPilot may split direct generation into a fixed per-file stage plan. For "
            "each request, return one `NUMPILOT_BUNDLE_V1` line bundle with exactly "
            "the paths named by that stage and no Markdown fence.\n",
            "- The union of all stages must stay below 2 MiB and use exactly the 17 "
            "locally declared project files; each stage must stay within the model limit.\n",
            "- It must include `CMakeLists.txt`, `build.sh`, and complete files under "
            f"`op_host/`, `op_kernel/`, `op_api/`, `tests/`, and `asnumpy_adapter/` for `{safe_name}`.\n",
            "- Do not emit `TODO_VERIFY_CANN_API`, stub returns, placeholder kernels, "
            "or claims that later qualification will fill in missing implementation.\n",
            "- `build.sh` must use only the local workspace and configured CANN SDK; "
            "it must never install into a system OPP path.\n",
            "- Put raw source between `CONTENT:` and `END_FILE`; do not JSON-escape "
            "quotes or newlines, and never emit `END_FILE` as a content line.\n",
            "- Finish with one `END_BUNDLE` line.\n",
        ))
    return request
