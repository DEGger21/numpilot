"""Prepare deterministic workspaces for fail-closed operator completion."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Optional, Tuple

from numpilot.agents.analysis import AnalysisAgent
from numpilot.gaps.models import GapSpec, OperatorDraft
from numpilot.gaps.provider import DraftGenerationError, LlmDraftProvider
from numpilot.gaps.spec import (
    blocked_unsupported,
    build_gap_specs,
    extract_semantic_contract,
    render_generation_request,
)


class GapCompletionError(RuntimeError):
    """A gap workspace cannot be prepared without losing safety or evidence."""


_MAX_SOURCE_BYTES = 4 * 1024 * 1024
_MAX_GAPS_PER_RUN = 32


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
    ) -> GapCompletionResult:
        source_file = Path(source_path)
        try:
            source_bytes = source_file.read_bytes()
            if len(source_bytes) > _MAX_SOURCE_BYTES:
                raise GapCompletionError(
                    "source exceeds the %d-byte analysis limit"
                    % _MAX_SOURCE_BYTES
                )
            source = source_bytes.decode("utf-8")
        except GapCompletionError:
            raise
        except (OSError, UnicodeError) as exc:
            raise GapCompletionError(
                "cannot read source %s: %s"
                % (source_file, type(exc).__name__)
            ) from exc

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
        if len(specs) > _MAX_GAPS_PER_RUN:
            raise GapCompletionError(
                "source contains too many generatable gaps (%d > %d)"
                % (len(specs), _MAX_GAPS_PER_RUN)
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
            unsupported=tuple(analysis.unsupported),
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
