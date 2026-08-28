"""Prepare deterministic workspaces for fail-closed operator completion."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from numpilot.agents.analysis import AnalysisAgent
from numpilot.gaps.models import GapSpec, OperatorDraft
from numpilot.gaps.provider import DraftGenerationError, LlmDraftProvider
from numpilot.gaps.spec import (
    blocked_unsupported,
    build_gap_specs,
    extract_semantic_contract,
    gap_spec_identity_matches,
    render_generation_request,
)


class GapCompletionError(RuntimeError):
    """A gap workspace cannot be prepared without losing safety or evidence."""


@dataclass(frozen=True)
class ScriptGapRequest:
    """Canonical local selection reconstructed from a migration hand-off."""

    specs: Tuple[GapSpec, ...]
    blocked: Tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class GapPreparation:
    """Pure local gap selection completed before any model or Skill access."""

    source_path: str
    source: str
    unsupported: Tuple[str, ...]
    specs: Tuple[GapSpec, ...]
    blocked: Tuple[Mapping[str, Any], ...]


_MAX_SOURCE_BYTES = 4 * 1024 * 1024
_MAX_GAP_REQUEST_BYTES = 2 * 1024 * 1024
_MAX_GAPS_PER_RUN = 32
_SCRIPT_GAP_REQUEST_FIELDS = frozenset({
    "schema_version",
    "request_type",
    "status",
    "source_path",
    "source_sha256",
    "backend",
    "backend_version",
    "audit_id",
    "target_soc",
    "cann_version",
    "gaps",
    "gap_specs",
    "generated_code",
    "qualification_performed",
    "activation_performed",
})


def _strict_json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key: %s" % key)
        value[key] = item
    return value


def _reject_json_constant(value):
    raise ValueError("non-standard JSON constant: %s" % value)


def _read_regular_bounded(path: Path, maximum: int, label: str) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    fd = -1
    try:
        fd = os.open(str(path), flags)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise GapCompletionError("%s must be a regular file" % label)
        if before.st_size > maximum:
            raise GapCompletionError("%s exceeds the size limit" % label)
        chunks = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(fd, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > maximum:
            raise GapCompletionError("%s exceeds the size limit" % label)
        after = os.fstat(fd)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
        ):
            raise GapCompletionError("%s changed while it was being read" % label)
        # Some filesystems expose coarse timestamps.  A second read from the
        # already-open inode closes that gap without resolving the path again.
        os.lseek(fd, 0, os.SEEK_SET)
        verification_chunks = []
        verification_remaining = maximum + 1
        while verification_remaining:
            chunk = os.read(fd, min(verification_remaining, 64 * 1024))
            if not chunk:
                break
            verification_chunks.append(chunk)
            verification_remaining -= len(chunk)
        if b"".join(verification_chunks) != payload:
            raise GapCompletionError("%s changed while it was being read" % label)
        return payload
    except GapCompletionError:
        raise
    except OSError as exc:
        raise GapCompletionError(
            "cannot read %s: %s" % (label, type(exc).__name__)
        ) from exc
    finally:
        if fd >= 0:
            os.close(fd)


def _same_absolute_path(left: str, right: str) -> bool:
    return os.path.abspath(left) == os.path.abspath(right)


def _validate_requested_specs(
    source: str,
    source_path: str,
    specs: Sequence[GapSpec],
    *,
    target_soc: str,
    cann_version: str,
) -> Tuple[GapSpec, ...]:
    """Rebind requested script GapSpecs to the current inert source text."""

    from numpilot.script_migration import scan_numpy_script

    selected = tuple(specs)
    if not selected:
        raise GapCompletionError("gap request contains no generatable GapSpec")
    if len(selected) > _MAX_GAPS_PER_RUN:
        raise GapCompletionError(
            "gap request contains too many specs (%d > %d)"
            % (len(selected), _MAX_GAPS_PER_RUN)
        )
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    try:
        scan = scan_numpy_script(source, source_path)
    except Exception as exc:
        raise GapCompletionError(
            "requested source cannot be scanned safely: %s" % type(exc).__name__
        ) from exc
    if not scan.ok:
        raise GapCompletionError(
            "requested source no longer satisfies ordinary-script scan: %s"
            % "; ".join(scan.issues)
        )
    uses_by_api = {}
    for use in scan.call_uses:
        uses_by_api.setdefault(use.qualified_api, []).append(use)

    seen = set()
    for spec in selected:
        if spec.qualified_api in seen:
            raise GapCompletionError(
                "gap request repeats %s" % spec.qualified_api
            )
        seen.add(spec.qualified_api)
        if (
            spec.source_sha256 != source_hash
            or not _same_absolute_path(spec.source_path, source_path)
        ):
            raise GapCompletionError(
                "GapSpec source binding does not match the current file"
            )
        if spec.target_soc != target_soc or spec.cann_version != cann_version:
            raise GapCompletionError(
                "GapSpec target does not match the requested SoC/CANN identity"
            )
        if (
            spec.qualified_api != "numpy." + spec.api_name
            or spec.asnumpy_api != "asnumpy." + spec.api_name
            or spec.semantic_source != spec.qualified_api
            or spec.route != "ascendc_registry_candidate"
            or not gap_spec_identity_matches(spec)
        ):
            raise GapCompletionError(
                "GapSpec identity/route is not a valid ordinary-script gap"
            )
        route_states = {
            (str(item.get("route", "")), str(item.get("status", "")))
            for item in spec.route_decisions
        }
        if (
            ("replacement_runtime", "required_before_qualification")
            not in route_states
            or ("ascendc_registry", "candidate_for_draft")
            not in route_states
            or not (
                ("existing_asnumpy_fallback", "user_rejected") in route_states
                or ("asnumpy_runtime", "missing") in route_states
            )
        ):
            raise GapCompletionError(
                "GapSpec does not record a missing or user-rejected fallback route"
            )

        current_uses = uses_by_api.get(spec.qualified_api, [])
        expected_locations = sorted(
            (use.lineno, use.col_offset) for use in current_uses
        )
        spec_locations = sorted(
            (call.lineno, call.col_offset) for call in spec.callsites
        )
        expected_signatures = sorted({
            (use.positional_args, tuple(use.keyword_args))
            for use in current_uses
        })
        try:
            spec_signatures = sorted({
                (
                    int(item["positional_args"]),
                    tuple(str(name) for name in item.get("keyword_args", ())),
                )
                for item in spec.signature_variants
            })
        except (KeyError, TypeError, ValueError) as exc:
            raise GapCompletionError(
                "GapSpec signature variants are invalid"
            ) from exc
        if (
            not current_uses
            or expected_locations != spec_locations
            or expected_signatures != spec_signatures
            or any(call.qualified_api != spec.qualified_api for call in spec.callsites)
        ):
            raise GapCompletionError(
                "GapSpec callsites/signatures no longer match the current source"
            )
    return selected


def load_script_gap_request(
    request_path: str,
    source_path: str,
    *,
    target_soc: str,
    cann_version: str,
) -> ScriptGapRequest:
    """Load and validate a migration-produced request before any remote work."""

    request_file = Path(request_path)
    source_file = Path(source_path)
    try:
        payload = _read_regular_bounded(
            request_file, _MAX_GAP_REQUEST_BYTES, "gap request"
        )
        request = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
        source_bytes = _read_regular_bounded(
            source_file, _MAX_SOURCE_BYTES, "requested source"
        )
        source = source_bytes.decode("utf-8")
    except GapCompletionError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise GapCompletionError(
            "cannot read gap request/source: %s" % type(exc).__name__
        ) from exc
    if (
        not isinstance(request, dict)
        or frozenset(request) != _SCRIPT_GAP_REQUEST_FIELDS
    ):
        raise GapCompletionError("gap request fields are invalid")
    if (
        type(request.get("schema_version")) is not int
        or request.get("schema_version") != 1
        or request.get("request_type") != "script_api_gap"
        or request.get("status") != "gap_generation_required"
        or request.get("generated_code") is not False
        or request.get("qualification_performed") is not False
        or request.get("activation_performed") is not False
        or request.get("backend") != "npu"
    ):
        raise GapCompletionError("unsupported or incomplete script gap request")
    for name in (
        "source_path",
        "source_sha256",
        "audit_id",
        "target_soc",
        "cann_version",
    ):
        value = request.get(name)
        if not isinstance(value, str) or not value or len(value) > 4096:
            raise GapCompletionError("gap request field is invalid: %s" % name)
    backend_version = request.get("backend_version")
    if backend_version is not None and (
        not isinstance(backend_version, str)
        or not backend_version
        or len(backend_version) > 512
    ):
        raise GapCompletionError("gap request backend version is invalid")
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if (
        request.get("source_sha256") != source_hash
        or not _same_absolute_path(str(request.get("source_path", "")), source_path)
        or request.get("target_soc") != target_soc
        or request.get("cann_version") != cann_version
    ):
        raise GapCompletionError(
            "gap request source/target binding does not match this command"
        )
    raw_gaps = request.get("gaps")
    raw_specs = request.get("gap_specs")
    if not isinstance(raw_gaps, list) or not isinstance(raw_specs, list):
        raise GapCompletionError("gap request has no decision/GapSpec list")

    # Never send request-controlled prose or serialized specs to the model.
    # Rebuild every authoritative GapSpec from the current inert source and a
    # tiny, enumerated decision vocabulary; the serialized specs are only an
    # exact integrity assertion over that canonical local reconstruction.
    from numpilot.gaps.spec import build_script_gap_specs
    from numpilot.script_migration import scan_numpy_script

    try:
        scan = scan_numpy_script(
            source, str(source_file.absolute())
        )
    except Exception as exc:
        raise GapCompletionError(
            "cannot rescan the gap-request source: %s" % type(exc).__name__
        ) from exc
    if not scan.ok:
        raise GapCompletionError(
            "gap-request source no longer satisfies ordinary-script scan"
        )
    source_calls = {}
    for use in scan.call_uses:
        source_calls.setdefault(use.qualified_api, []).append(use)

    decisions = []
    blocked = []
    seen = set()
    for raw in raw_gaps:
        if not isinstance(raw, Mapping):
            raise GapCompletionError("gap request contains a non-object decision")
        qualified_api = str(raw.get("qualified_api", ""))
        if not qualified_api.startswith("numpy."):
            raise GapCompletionError("script gap decision API is invalid")
        api_path = qualified_api.removeprefix("numpy.")
        if (
            not api_path
            or qualified_api in seen
            or raw.get("asnumpy_api") != "asnumpy." + api_path
        ):
            raise GapCompletionError("script gap decision identity is invalid")
        if raw.get("generatable") is True and "." in api_path:
            raise GapCompletionError("generatable script gap must be top-level")
        seen.add(qualified_api)
        all_source_uses = [
            use for use in scan.uses if use.qualified_api == qualified_api
        ]
        if raw.get("generatable") is not True:
            if (
                not all_source_uses
                or raw.get("callsites")
                != [item.to_dict() for item in all_source_uses]
            ):
                raise GapCompletionError(
                    "blocked script gap decision no longer matches source"
                )
            blocked.append({
                "name": api_path,
                "qualified_api": qualified_api,
                "lineno": all_source_uses[0].lineno,
                "reason": (
                    "ordinary-script gap generation supports direct static "
                    "calls only"
                ),
            })
            continue
        calls = source_calls.get(qualified_api, [])
        expected_callsites = [item.to_dict() for item in calls]
        expected_signatures = [
            [positional, list(keywords)]
            for positional, keywords in sorted({
                (item.positional_args, item.keyword_args) for item in calls
            })
        ]
        if (
            not calls
            or raw.get("callsites") != expected_callsites
            or raw.get("signature_variants") != expected_signatures
        ):
            raise GapCompletionError(
                "script gap decision callsites/signatures do not match source"
            )
        status = str(raw.get("current_status", ""))
        available = raw.get("current_available")
        provenance = str(raw.get("current_provenance", ""))
        fallback = (
            status == "fallback_available"
            and available is True
            and provenance in {
                "cpu_fallback",
                "conditional_cpu_fallback",
                "host_assisted",
            }
        )
        missing = (
            status == "unsupported"
            and available is False
            and provenance == "unknown"
        )
        if not (fallback or missing):
            raise GapCompletionError(
                "script gap decision is not missing or an audited rejected fallback"
            )
        decisions.append({
            "api_path": api_path,
            "qualified_api": qualified_api,
            "kind": "call",
            "status": status,
            "available": available,
            "provenance": provenance,
            "evidence": "canonical local reconstruction",
            "constraints": [],
            "callsites": expected_callsites,
            "generatable": True,
        })
    try:
        specs = build_script_gap_specs(
            source,
            str(source_file.absolute()),
            decisions,
            target_soc=target_soc,
            cann_version=cann_version,
        )
    except (TypeError, ValueError) as exc:
        raise GapCompletionError(
            "cannot reconstruct script GapSpecs"
        ) from exc
    if raw_specs != [spec.to_dict() for spec in specs]:
        raise GapCompletionError(
            "serialized GapSpecs do not match canonical local reconstruction"
        )
    validated = _validate_requested_specs(
        source,
        str(source_file.absolute()),
        specs,
        target_soc=target_soc,
        cann_version=cann_version,
    )
    return ScriptGapRequest(specs=validated, blocked=tuple(blocked))


def _ensure_directory(path: Path, *, private: bool) -> None:
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_dir():
            raise GapCompletionError(
                "completion directory is not a real directory: %s" % path
            )
        return
    try:
        path.mkdir(parents=True, mode=0o700 if private else 0o755)
    except OSError as exc:
        raise GapCompletionError(
            "cannot create completion directory %s: %s"
            % (path, type(exc).__name__)
        ) from exc


@dataclass(frozen=True)
class GapCompletionResult:
    status: str
    source_path: str
    out_dir: str
    unsupported: Tuple[str, ...]
    specs: Tuple[GapSpec, ...]
    blocked: Tuple[Mapping[str, Any], ...]
    drafts: Tuple[OperatorDraft, ...] = ()
    skill_bundle: Optional[Mapping[str, Any]] = None

    @property
    def generated_count(self) -> int:
        return len(self.drafts)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "schema_version": 1,
            "status": self.status,
            "source_path": self.source_path,
            "out_dir": self.out_dir,
            "unsupported": list(self.unsupported),
            "gap_specs": [spec.to_dict() for spec in self.specs],
            "blocked": [dict(item) for item in self.blocked],
            "drafts": [draft.to_dict() for draft in self.drafts],
            "generated_count": self.generated_count,
            "qualified_count": 0,
            "activated_count": 0,
        }
        if self.skill_bundle is not None:
            result["skill_bundle"] = dict(self.skill_bundle)
        return result


def prepare_gap_completion(
    source_path: str,
    *,
    target_soc: str,
    cann_version: str,
    requested_specs: Optional[Sequence[GapSpec]] = None,
    requested_blocked: Sequence[Mapping[str, Any]] = (),
) -> GapPreparation:
    """Read and statically classify a source before any remote readiness gate.

    This split lets local official-operator resolvers inspect canonical GapSpecs
    before NumPilot decides whether a new-kernel model request is necessary.
    It deliberately performs no output writes and reads no external Skill.
    """

    source_file = Path(source_path)
    try:
        source_bytes = _read_regular_bounded(
            source_file, _MAX_SOURCE_BYTES, "source"
        )
        source = source_bytes.decode("utf-8")
    except GapCompletionError:
        raise
    except (OSError, UnicodeError) as exc:
        raise GapCompletionError(
            "cannot read source %s: %s"
            % (source_file, type(exc).__name__)
        ) from exc

    if requested_specs is None:
        try:
            analysis = AnalysisAgent().analyze(source, str(source_file))
        except SyntaxError as exc:
            raise GapCompletionError(
                "cannot parse source %s at line %s"
                % (source_file, exc.lineno or "?")
            ) from exc
        try:
            specs = build_gap_specs(
                source,
                analysis,
                target_soc=target_soc,
                cann_version=cann_version,
            )
        except ValueError as exc:
            raise GapCompletionError(
                "cannot build a target-consistent gap specification: %s" % exc
            ) from exc
        blocked = blocked_unsupported(analysis)
        unsupported = tuple(analysis.unsupported)
    else:
        specs = _validate_requested_specs(
            source,
            str(source_file),
            requested_specs,
            target_soc=target_soc,
            cann_version=cann_version,
        )
        blocked = tuple(dict(item) for item in requested_blocked)
        unsupported = tuple(spec.qualified_api for spec in specs)
    if len(specs) > _MAX_GAPS_PER_RUN:
        raise GapCompletionError(
            "source contains too many generatable gaps (%d > %d)"
            % (len(specs), _MAX_GAPS_PER_RUN)
        )
    return GapPreparation(
        source_path=str(source_file),
        source=source,
        unsupported=unsupported,
        specs=tuple(specs),
        blocked=tuple(blocked),
    )


def _atomic_write(path: Path, payload: bytes, *, replace: bool) -> None:
    _ensure_directory(path.parent, private=True)
    fd = -1
    temp_name = ""
    try:
        fd, temp_name = tempfile.mkstemp(
            prefix="." + path.name + "-", suffix=".tmp", dir=str(path.parent)
        )
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temp_name, path)
            temp_name = ""
        else:
            try:
                os.link(temp_name, path)
            except FileExistsError:
                if path.is_symlink() or not path.is_file():
                    raise GapCompletionError(
                        "refusing existing non-regular file: %s" % path
                    )
                if path.read_bytes() == payload:
                    return
                raise GapCompletionError(
                    "refusing to overwrite existing file: %s" % path
                )
            os.unlink(temp_name)
            temp_name = ""
        os.chmod(path, 0o600)
    except OSError as exc:
        raise GapCompletionError(
            "cannot write completion artifact %s: %s"
            % (path, type(exc).__name__)
        ) from exc
    finally:
        if fd >= 0:
            os.close(fd)
        if temp_name:
            try:
                os.unlink(temp_name)
            except OSError:
                pass


def _write_json(path: Path, data: Mapping[str, Any], *, replace: bool) -> None:
    payload = (
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write(path, payload, replace=replace)


class GapCompletionPipeline:
    """Detect gaps, prepare specs, and optionally create unverified LLM drafts."""

    def __init__(
        self,
        provider: Optional[LlmDraftProvider] = None,
        *,
        skill_root: str = "",
        asnumpy_reference: str = "",
        skill_bundle=None,
    ):
        self.provider = provider
        self.skill_root = skill_root
        self.asnumpy_reference = asnumpy_reference
        self.skill_bundle = skill_bundle

    def run(
        self,
        source_path: str,
        out_dir: str,
        *,
        target_soc: str,
        cann_version: str,
        requested_specs: Optional[Sequence[GapSpec]] = None,
        requested_blocked: Sequence[Mapping[str, Any]] = (),
        preparation: Optional[GapPreparation] = None,
    ) -> GapCompletionResult:
        if preparation is None:
            preparation = prepare_gap_completion(
                source_path,
                target_soc=target_soc,
                cann_version=cann_version,
                requested_specs=requested_specs,
                requested_blocked=requested_blocked,
            )
        elif (
            not isinstance(preparation, GapPreparation)
            or preparation.source_path != str(Path(source_path))
            or any(
                spec.target_soc != target_soc
                or spec.cann_version != cann_version
                for spec in preparation.specs
            )
        ):
            raise GapCompletionError(
                "prepared gap selection does not match the requested source/target"
            )
        source_file = Path(preparation.source_path)
        source = preparation.source
        specs = preparation.specs
        blocked = preparation.blocked
        unsupported = preparation.unsupported
        output = Path(out_dir)
        _ensure_directory(output, private=True)
        gaps_dir = output / "gaps"
        _ensure_directory(gaps_dir, private=True)
        if self.skill_bundle is not None:
            _write_json(
                output / "skill-bundle.json",
                self.skill_bundle.manifest(),
                replace=False,
            )
        drafts = []
        for spec in specs:
            workspace = gaps_dir / spec.gap_id
            _ensure_directory(workspace, private=True)
            _write_json(
                workspace / "gap-spec.json",
                spec.to_dict(),
                replace=False,
            )
            try:
                semantic_contract = extract_semantic_contract(source, spec)
                request = render_generation_request(
                    spec,
                    skill_root=self.skill_root,
                    asnumpy_reference=self.asnumpy_reference,
                    skill_bundle=self.skill_bundle,
                    semantic_contract=semantic_contract,
                ).encode("utf-8")
            except ValueError as exc:
                raise GapCompletionError(
                    "cannot prepare operator semantic contract for %s: %s"
                    % (spec.qualified_api, exc)
                ) from exc
            request_path = workspace / "GENERATION_REQUEST.md"
            draft_path = workspace / "draft"
            if os.path.lexists(draft_path):
                # An existing immutable draft is validated and reused by the
                # provider.  Preserve the exact request that originally
                # produced it instead of retroactively attaching a new Skill.
                if (
                    request_path.is_symlink()
                    or not request_path.is_file()
                ):
                    raise GapCompletionError(
                        "existing draft has no regular generation request"
                    )
                try:
                    request = request_path.read_bytes()
                except OSError as exc:
                    raise GapCompletionError(
                        "cannot read existing generation request: %s"
                        % type(exc).__name__
                    ) from exc
                if len(request) > 2 * 1024 * 1024:
                    raise GapCompletionError(
                        "existing generation request exceeds the size limit"
                    )
            else:
                _atomic_write(request_path, request, replace=False)
            if self.provider is not None:
                try:
                    drafts.append(self.provider.generate(
                        spec,
                        workspace,
                        generation_request=request.decode("utf-8"),
                    ))
                except DraftGenerationError as exc:
                    raise GapCompletionError(
                        "draft generation failed for %s: %s"
                        % (spec.qualified_api, exc)
                    ) from exc

        if drafts and blocked:
            status = "generated_unverified_partially_blocked"
        elif specs and blocked:
            status = "prepared_partially_blocked"
        elif drafts:
            status = "generated_unverified"
        elif specs:
            status = "prepared"
        elif blocked:
            status = "blocked"
        else:
            status = "no_gaps"
        result = GapCompletionResult(
            status=status,
            source_path=str(source_file),
            out_dir=str(output),
            unsupported=unsupported,
            specs=specs,
            blocked=blocked,
            drafts=tuple(drafts),
            skill_bundle=(
                self.skill_bundle.manifest()
                if self.skill_bundle is not None else None
            ),
        )
        _write_json(output / "gap-report.json", result.to_dict(), replace=True)
        return result
