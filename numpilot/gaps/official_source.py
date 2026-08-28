"""Read-only resolution of version-pinned Huawei ops-nn source candidates.

This module deliberately stops before checkout, build, adapter generation, or
activation.  A name match is discovery evidence, not semantic proof.  Source
candidates remain ambiguous until a separately trusted audit-verification
protocol is implemented.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import stat
import subprocess
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse

from numpilot.licenses import CANN_OSL_V2_ID, CANN_OSL_V2_NAME


OFFICIAL_SOURCE_SCHEMA_VERSION = 1
OPS_NN_LICENSE = CANN_OSL_V2_NAME
OPS_NN_LICENSE_ID = CANN_OSL_V2_ID
SOURCE_SEARCH_CONTRACT_REVISION = "ops-nn-direct-operator-aliases-v1"
_OFFICIAL_HTTPS_ORIGIN = "https://gitcode.com/cann/ops-nn.git"
_HEX_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_VERSION = re.compile(r"^[0-9]+(?:\.[0-9]+){1,3}(?:[-+][A-Za-z0-9.-]+)?$")
_SAFE_OP_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")
_SAFE_QUALIFIED_API = re.compile(
    r"^(?:numpy(?:\.[A-Za-z][A-Za-z0-9_]*)+|"
    r"operator\.[A-Za-z][A-Za-z0-9_]*)$"
)
_SAFE_REF = re.compile(
    r"^refs/(?:tags|remotes/[A-Za-z0-9._-]+)/"
    r"[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$"
)
_SAFE_TREE_PATH = re.compile(r"^[A-Za-z0-9._+/-]{1,512}$")
_VERSION_CMAKE = re.compile(
    r"set_package\(\s*ops-nn\s+VERSION\s+[\"']([^\"']+)[\"']\s*\)"
)
_LICENSE_MARKERS = (
    "CANN Open Software License Agreement Version 2.0",
    "Huawei AI Processors",
)
_INTEGRATION_DIRS = frozenset({"op_api", "op_host", "op_kernel", "op_graph"})
_SOC_TOKEN = re.compile(r"(?i)\b(?:ascend|kirin)[a-z0-9_]+\b")
_VERSION_SUFFIX = re.compile(r"_v[0-9]+$")
_GIT_IDENTITY_OUTPUT_LIMIT = 256
_GIT_TYPE_OUTPUT_LIMIT = 64
_GIT_CONFIG_OUTPUT_LIMIT = 4 * 1024
_GIT_STDERR_OUTPUT_LIMIT = 64 * 1024
_GIT_BLOB_OUTPUT_LIMIT = 4 * 1024 * 1024
_GIT_TREE_OUTPUT_LIMIT = 16 * 1024 * 1024
_GIT_READ_CHUNK = 64 * 1024


def _trusted_git_executable() -> Optional[Path]:
    """Return a fixed system Git binary, never a PATH shadow."""

    for candidate in (Path("/usr/bin/git"), Path("/bin/git")):
        try:
            resolved = candidate.resolve(strict=True)
            metadata = resolved.stat()
        except (OSError, RuntimeError):
            continue
        if (
            stat.S_ISREG(metadata.st_mode)
            and os.access(resolved, os.X_OK)
            and metadata.st_uid == 0
            and not metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            return resolved
    return None


@dataclass(frozen=True)
class OpsNnSourcePin:
    """A local ref plus its expected immutable Git identities."""

    ref: str
    expected_commit: str
    expected_tree: str = ""

    def __post_init__(self) -> None:
        if (
            not _SAFE_REF.fullmatch(self.ref)
            or ".." in self.ref
            or self.ref.endswith("/")
        ):
            raise ValueError("ops-nn source pin requires a safe explicit tag/ref")
        if not _HEX_SHA1.fullmatch(self.expected_commit):
            raise ValueError("ops-nn source pin requires a lowercase commit id")
        if self.expected_tree and not _HEX_SHA1.fullmatch(self.expected_tree):
            raise ValueError("ops-nn source pin tree must be a lowercase tree id")


# The initial production slice is intentionally small.  Adding another CANN
# release requires recording and reviewing its immutable tag target and tree.
DEFAULT_OPS_NN_SOURCE_PINS: Mapping[str, OpsNnSourcePin] = {
    "9.0.0": OpsNnSourcePin(
        ref="refs/tags/v9.0.0",
        expected_commit="0413e303e6cd1986b592c3e039f2579ac0f10b62",
        expected_tree="827d73df88f73353821b63ed4e0d1711b6d2cb50",
    ),
}

# A tree-wide name miss is only meaningful when the requested API has a
# reviewed direct-operator alias set.  APIs absent from this bounded catalog
# stay ``ambiguous``: spelling equality alone cannot rule out a differently
# named semantic equivalent.  Extending this catalog is a framework review,
# not a model-generation action.
REVIEWED_SOURCE_SEARCH_ALIASES: Mapping[Tuple[str, str], Tuple[str, ...]] = {
    ("9.0.0", "numpy.arange"): ("arange", "range"),
    ("9.0.0", "numpy.cbrt"): ("cbrt", "cube_root", "cuberoot"),
    ("9.0.0", "numpy.matmul"): ("matmul", "mat_mul", "mat_mul_v3"),
}


@dataclass(frozen=True)
class OfficialSourceSemanticAudit:
    """Reserved shape for a future trusted semantic-audit receipt.

    Merely constructing this value does not establish trust.  The current
    resolver intentionally refuses to promote a source candidate from it: no
    trusted reviewer registry, signature verification, or caller-supplied
    expected contract digest exists yet.
    """

    candidate_id: str
    qualified_api: str
    cann_version: str
    target_soc: str
    contract_sha256: str
    evidence_sha256: str
    reviewer: str
    conclusion: str = "exact_semantic_match"
    schema_version: int = OFFICIAL_SOURCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OFFICIAL_SOURCE_SCHEMA_VERSION:
            raise ValueError("unsupported official-source audit schema")
        if not _HEX_SHA256.fullmatch(self.candidate_id):
            raise ValueError("semantic audit candidate_id must be a SHA-256")
        if not _SAFE_QUALIFIED_API.fullmatch(self.qualified_api):
            raise ValueError("semantic audit requires a qualified source API")
        if not _SAFE_VERSION.fullmatch(self.cann_version):
            raise ValueError("semantic audit CANN version is invalid")
        if not self.target_soc.strip() or len(self.target_soc) > 128:
            raise ValueError("semantic audit target SoC is invalid")
        if not _HEX_SHA256.fullmatch(self.contract_sha256):
            raise ValueError("semantic contract digest must be a SHA-256")
        if not _HEX_SHA256.fullmatch(self.evidence_sha256):
            raise ValueError("semantic evidence digest must be a SHA-256")
        if not self.reviewer.strip() or len(self.reviewer) > 256:
            raise ValueError("semantic audit requires a bounded reviewer identity")
        if self.conclusion != "exact_semantic_match":
            raise ValueError("semantic audit conclusion is not an exact match")


@dataclass(frozen=True)
class OpsNnCandidate:
    provider: str
    origin: str
    source_ref: str
    source_commit: str
    source_tree: str
    cann_version: str
    license_name: str
    license_sha256: str
    op_path: str
    op_tree: str
    stability: str
    aliases: Tuple[str, ...]
    has_op_api: bool
    has_op_host: bool
    has_op_kernel: bool
    op_api_paths: Tuple[str, ...]
    cmake_paths: Tuple[str, ...]
    soc_cmake_clues: Tuple[str, ...]
    soc_config_clues: Tuple[str, ...]
    target_soc_evidence: str
    evidence_files: Tuple[Tuple[str, str], ...]
    candidate_id: str

    @property
    def integration_surface_complete(self) -> bool:
        return self.has_op_api and self.has_op_host and self.has_op_kernel

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "origin": self.origin,
            "source_ref": self.source_ref,
            "source_commit": self.source_commit,
            "source_tree": self.source_tree,
            "cann_version": self.cann_version,
            "license": {
                "name": self.license_name,
                "id": OPS_NN_LICENSE_ID,
                "sha256": self.license_sha256,
                "acceptance_required_before_use_or_build": True,
            },
            "op_path": self.op_path,
            "op_tree": self.op_tree,
            "stability": self.stability,
            "aliases": list(self.aliases),
            "surfaces": {
                "op_api": self.has_op_api,
                "op_host": self.has_op_host,
                "op_kernel": self.has_op_kernel,
                "integration_surface_complete": self.integration_surface_complete,
            },
            "op_api_paths": list(self.op_api_paths),
            "cmake_paths": list(self.cmake_paths),
            "soc_cmake_clues": list(self.soc_cmake_clues),
            "soc_config_clues": list(self.soc_config_clues),
            "target_soc_evidence": self.target_soc_evidence,
            "evidence_files": dict(self.evidence_files),
            "candidate_id": self.candidate_id,
        }


@dataclass(frozen=True)
class OfficialSourceResolution:
    """Fail-closed outcome of a local official-source lookup."""

    status: str
    reason: str
    cann_version: str
    target_soc: str
    op_name: str
    qualified_api: str
    source_ref: str = ""
    source_commit: str = ""
    source_tree: str = ""
    candidates: Tuple[OpsNnCandidate, ...] = ()
    selected_candidate_id: str = ""
    requires_audit: bool = False
    resolution_kind: str = "none"
    search_contract_revision: str = ""
    searched_aliases: Tuple[str, ...] = ()
    negative_search_complete: bool = False
    activation_allowed: bool = False
    schema_version: int = OFFICIAL_SOURCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        allowed = {
            "exact_candidate", "no_match", "incompatible", "ambiguous", "error"
        }
        if self.status not in allowed:
            raise ValueError("invalid official-source resolution status")
        if self.activation_allowed:
            raise ValueError("official-source resolution cannot activate an operator")
        if self.status == "no_match" and (
            not self.negative_search_complete
            or self.search_contract_revision != SOURCE_SEARCH_CONTRACT_REVISION
            or not self.searched_aliases
        ):
            raise ValueError("source no_match requires reviewed negative-search evidence")
        if self.negative_search_complete and self.status != "no_match":
            raise ValueError("negative source evidence contradicts the resolution status")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider": "ops_nn_source",
            "status": self.status,
            "reason": self.reason,
            "requested": {
                "cann_version": self.cann_version,
                "target_soc": self.target_soc,
                "op_name": self.op_name,
                "qualified_api": self.qualified_api,
            },
            "source_ref": self.source_ref,
            "source_commit": self.source_commit,
            "source_tree": self.source_tree,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selected_candidate_id": self.selected_candidate_id,
            "requires_audit": self.requires_audit,
            "resolution_kind": self.resolution_kind,
            "search_contract_revision": self.search_contract_revision,
            "searched_aliases": list(self.searched_aliases),
            "negative_search_complete": self.negative_search_complete,
            "activation_allowed": False,
        }


class _GitReadError(RuntimeError):
    pass


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _canonical_official_origin(value: str) -> Optional[str]:
    value = value.strip()
    if re.fullmatch(r"git@gitcode\.com:cann/ops-nn(?:\.git)?/?", value):
        return _OFFICIAL_HTTPS_ORIGIN
    parsed = urlparse(value)
    if parsed.scheme not in {"https", "ssh"}:
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    if parsed.password or parsed.query or parsed.fragment or port is not None:
        return None
    if (parsed.hostname or "").casefold() != "gitcode.com":
        return None
    path = parsed.path.rstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    if path != "/cann/ops-nn":
        return None
    if parsed.scheme == "https" and parsed.username is not None:
        return None
    if parsed.scheme == "ssh" and parsed.username not in {None, "git"}:
        return None
    return _OFFICIAL_HTTPS_ORIGIN


class OpsNnSourceResolver:
    """Inspect a pinned ops-nn Git object without touching its worktree."""

    def __init__(
        self,
        checkout: Union[Path, str],
        *,
        pins: Optional[Mapping[str, OpsNnSourcePin]] = None,
        git_timeout_seconds: float = 10.0,
    ) -> None:
        try:
            supplied = Path(checkout).expanduser()
            if not supplied.is_absolute():
                supplied = supplied.absolute()
            resolved = supplied.resolve(strict=True)
        except (OSError, RuntimeError, TypeError) as exc:
            raise ValueError(
                "ops-nn checkout must be a real canonical directory"
            ) from exc
        if supplied != resolved or supplied.is_symlink() or not resolved.is_dir():
            raise ValueError(
                "ops-nn checkout must be a real canonical directory"
            )
        self.checkout = resolved
        self.pins = dict(
            DEFAULT_OPS_NN_SOURCE_PINS if pins is None else pins
        )
        for version, pin in self.pins.items():
            if not _SAFE_VERSION.fullmatch(version) or not isinstance(
                pin, OpsNnSourcePin
            ):
                raise ValueError("invalid ops-nn CANN source-pin mapping")
        if git_timeout_seconds <= 0 or git_timeout_seconds > 60:
            raise ValueError("git timeout must be in (0, 60] seconds")
        self.git_timeout_seconds = git_timeout_seconds
        self.git_executable = _trusted_git_executable()

    @staticmethod
    def _git_stdout_limit(args: Sequence[str]) -> int:
        if not args:
            raise _GitReadError("local git inspection command is empty")
        if args[0] == "rev-parse":
            return _GIT_IDENTITY_OUTPUT_LIMIT
        if args[0] == "config":
            return _GIT_CONFIG_OUTPUT_LIMIT
        if args[0] == "ls-tree":
            return _GIT_TREE_OUTPUT_LIMIT
        if args[0] == "cat-file" and len(args) >= 2:
            if args[1] == "-t":
                return _GIT_TYPE_OUTPUT_LIMIT
            if args[1] == "blob":
                return _GIT_BLOB_OUTPUT_LIMIT
        raise _GitReadError("unsupported local git inspection command")

    @staticmethod
    def _stop_git_process(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _run_git_bounded(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str],
        stdout_limit: int,
    ) -> Tuple[int, bytes, bytes]:
        process: Optional[subprocess.Popen] = None
        selector = selectors.DefaultSelector()
        stdout = bytearray()
        stderr = bytearray()
        deadline = time.monotonic() + self.git_timeout_seconds
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=dict(env),
                close_fds=True,
            )
            if process.stdout is None or process.stderr is None:
                raise _GitReadError("local git inspection pipes are unavailable")
            selector.register(
                process.stdout,
                selectors.EVENT_READ,
                ("stdout", stdout, stdout_limit),
            )
            selector.register(
                process.stderr,
                selectors.EVENT_READ,
                ("stderr", stderr, _GIT_STDERR_OUTPUT_LIMIT),
            )
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _GitReadError("local git inspection timed out")
                events = selector.select(remaining)
                if not events:
                    raise _GitReadError("local git inspection timed out")
                for key, _mask in events:
                    label, output, limit = key.data
                    room = limit - len(output)
                    chunk = os.read(
                        key.fd,
                        min(_GIT_READ_CHUNK, room + 1),
                    )
                    if not chunk:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    if len(chunk) > room:
                        raise _GitReadError(
                            "local git %s exceeded its %d-byte output limit"
                            % (label, limit)
                        )
                    output.extend(chunk)
            remaining = deadline - time.monotonic()
            if remaining <= 0 and process.poll() is None:
                raise _GitReadError("local git inspection timed out")
            returncode = process.wait(timeout=max(remaining, 0.001))
            return returncode, bytes(stdout), bytes(stderr)
        except subprocess.TimeoutExpired as exc:
            raise _GitReadError("local git inspection timed out") from exc
        finally:
            selector.close()
            if process is not None:
                if process.stdout is not None and not process.stdout.closed:
                    process.stdout.close()
                if process.stderr is not None and not process.stderr.closed:
                    process.stderr.close()
                self._stop_git_process(process)

    def _git_bytes(self, *args: str) -> bytes:
        if self.git_executable is None:
            raise _GitReadError("a trusted system Git executable is unavailable")
        stdout_limit = self._git_stdout_limit(args)
        env = {
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_PAGER": "cat",
            "HOME": "/nonexistent-numpilot-git-home",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        }
        command = (
            str(self.git_executable),
            "--no-replace-objects",
            "-c", "gc.auto=0", "-c", "maintenance.auto=false",
            "-C", str(self.checkout), *args,
        )
        try:
            returncode, stdout, stderr = self._run_git_bounded(
                command,
                env=env,
                stdout_limit=stdout_limit,
            )
        except OSError as exc:
            raise _GitReadError("local git inspection failed: %s" % exc) from exc
        if returncode != 0:
            diagnostic = stderr.decode("utf-8", "replace").strip()
            raise _GitReadError(
                "local git inspection failed: %s" % (diagnostic[:400] or "unknown error")
            )
        return stdout

    def _git_text(self, *args: str) -> str:
        return self._git_bytes(*args).decode("utf-8", "strict").strip()

    def _blob(self, commit: str, path: str) -> bytes:
        if not _HEX_SHA1.fullmatch(commit) or not _SAFE_TREE_PATH.fullmatch(path):
            raise _GitReadError("unsafe Git object identity")
        blob_id = self._git_text("rev-parse", "--verify", "%s:%s" % (commit, path))
        if not _HEX_SHA1.fullmatch(blob_id):
            raise _GitReadError("Git did not resolve a blob identity")
        if self._git_text("cat-file", "-t", blob_id) != "blob":
            raise _GitReadError("Git object is not a blob: %s" % path)
        return self._git_bytes("cat-file", "blob", blob_id)

    def _base_result(
        self,
        status: str,
        reason: str,
        *,
        cann_version: str,
        target_soc: str,
        op_name: str,
        qualified_api: str,
        source_ref: str = "",
        source_commit: str = "",
        source_tree: str = "",
        candidates: Sequence[OpsNnCandidate] = (),
        selected_candidate_id: str = "",
        requires_audit: bool = False,
        resolution_kind: str = "none",
        search_contract_revision: str = "",
        searched_aliases: Sequence[str] = (),
        negative_search_complete: bool = False,
    ) -> OfficialSourceResolution:
        return OfficialSourceResolution(
            status=status,
            reason=reason,
            cann_version=cann_version,
            target_soc=target_soc,
            op_name=op_name,
            qualified_api=qualified_api,
            source_ref=source_ref,
            source_commit=source_commit,
            source_tree=source_tree,
            candidates=tuple(candidates),
            selected_candidate_id=selected_candidate_id,
            requires_audit=requires_audit,
            resolution_kind=resolution_kind,
            search_contract_revision=search_contract_revision,
            searched_aliases=tuple(searched_aliases),
            negative_search_complete=negative_search_complete,
        )

    def resolve(
        self,
        *,
        cann_version: str,
        target_soc: str,
        op_name: str,
        qualified_api: str = "",
        semantic_audit: Optional[OfficialSourceSemanticAudit] = None,
    ) -> OfficialSourceResolution:
        """Resolve one name against an explicitly pinned local CANN release."""

        if not qualified_api:
            qualified_api = "operator." + op_name
        common = {
            "cann_version": cann_version,
            "target_soc": target_soc,
            "op_name": op_name,
            "qualified_api": qualified_api,
        }
        if (
            not _SAFE_VERSION.fullmatch(cann_version)
            or not target_soc.strip()
            or len(target_soc) > 128
            or not _SAFE_OP_NAME.fullmatch(op_name)
            or not _SAFE_QUALIFIED_API.fullmatch(qualified_api)
            or qualified_api.rsplit(".", 1)[-1] != op_name
        ):
            return self._base_result(
                "error", "invalid official-source lookup request", **common
            )
        pin = self.pins.get(cann_version)
        if pin is None:
            return self._base_result(
                "incompatible",
                "no reviewed ops-nn source pin exists for CANN %s" % cann_version,
                **common,
            )
        try:
            if not self.checkout.is_dir():
                raise _GitReadError("local ops-nn checkout is unavailable")
            origin_raw = self._git_text(
                "config", "--local", "--get", "remote.origin.url"
            )
            origin = _canonical_official_origin(origin_raw)
            if origin is None:
                return self._base_result(
                    "incompatible", "local repository origin is not official ops-nn",
                    source_ref=pin.ref, **common,
                )
            commit = self._git_text(
                "rev-parse", "--verify", "%s^{commit}" % pin.ref
            )
            tree = self._git_text(
                "rev-parse", "--verify", "%s^{tree}" % pin.ref
            )
            if commit != pin.expected_commit or (
                pin.expected_tree and tree != pin.expected_tree
            ):
                return self._base_result(
                    "incompatible",
                    "local ops-nn ref does not match its reviewed commit/tree pin",
                    source_ref=pin.ref,
                    source_commit=commit if _HEX_SHA1.fullmatch(commit) else "",
                    source_tree=tree if _HEX_SHA1.fullmatch(tree) else "",
                    **common,
                )
            version_text = self._blob(commit, "version.cmake").decode(
                "utf-8", "strict"
            )
            match = _VERSION_CMAKE.search(version_text)
            if match is None or match.group(1) != cann_version:
                return self._base_result(
                    "incompatible",
                    "pinned ops-nn package version does not match requested CANN",
                    source_ref=pin.ref, source_commit=commit, source_tree=tree,
                    **common,
                )
            license_bytes = self._blob(commit, "LICENSE")
            license_text = license_bytes.decode("utf-8", "strict")
            if not all(marker in license_text for marker in _LICENSE_MARKERS):
                return self._base_result(
                    "incompatible",
                    "pinned source does not contain the reviewed CANN OSL v2 license",
                    source_ref=pin.ref, source_commit=commit, source_tree=tree,
                    **common,
                )
            raw_paths = self._git_bytes(
                "ls-tree", "-rz", "--name-only", commit
            ).split(b"\0")
            paths = tuple(
                item.decode("utf-8", "strict")
                for item in raw_paths
                if item and _SAFE_TREE_PATH.fullmatch(item.decode("utf-8", "strict"))
            )
            reviewed_aliases = REVIEWED_SOURCE_SEARCH_ALIASES.get(
                (cann_version, qualified_api)
            )
            searched_aliases = reviewed_aliases or (op_name,)
            candidates = self._find_candidates(
                paths=paths,
                queries=searched_aliases,
                origin=origin,
                pin=pin,
                commit=commit,
                tree=tree,
                cann_version=cann_version,
                target_soc=target_soc,
                license_sha256=hashlib.sha256(license_bytes).hexdigest(),
            )
        except (UnicodeError, _GitReadError) as exc:
            return self._base_result(
                "error", str(exc), source_ref=pin.ref, **common
            )

        source = {
            "source_ref": pin.ref,
            "source_commit": commit,
            "source_tree": tree,
        }
        if not candidates:
            if reviewed_aliases is None:
                return self._base_result(
                    "ambiguous",
                    "the bounded source search catalog has not reviewed aliases "
                    "for this API; a name miss is not a clean official miss",
                    requires_audit=True,
                    resolution_kind="unmapped",
                    searched_aliases=searched_aliases,
                    **source, **common,
                )
            return self._base_result(
                "no_match",
                "no operator matched the complete reviewed direct-operator alias set",
                search_contract_revision=SOURCE_SEARCH_CONTRACT_REVISION,
                searched_aliases=searched_aliases,
                negative_search_complete=True,
                **source, **common,
            )
        if semantic_audit is None:
            return self._base_result(
                "ambiguous",
                "name/API match is only a candidate; semantic, dtype, shape, "
                "and SoC audit is required",
                candidates=candidates,
                requires_audit=True,
                resolution_kind="candidate",
                search_contract_revision=(
                    SOURCE_SEARCH_CONTRACT_REVISION
                    if reviewed_aliases is not None else ""
                ),
                searched_aliases=searched_aliases,
                **source, **common,
            )
        # A caller-controlled digest and reviewer label are not a trust
        # boundary.  Keep the argument for source compatibility, but fail
        # closed until NumPilot has a signed audit receipt whose contract
        # digest is compared with the current GapSpec.
        return self._base_result(
            "error",
            "semantic audit promotion is disabled until trusted signature and "
            "current-contract verification are implemented",
            candidates=candidates,
            requires_audit=True,
            resolution_kind="candidate",
            search_contract_revision=(
                SOURCE_SEARCH_CONTRACT_REVISION
                if reviewed_aliases is not None else ""
            ),
            searched_aliases=searched_aliases,
            **source, **common,
        )

    def _find_candidates(
        self,
        *,
        paths: Sequence[str],
        queries: Sequence[str],
        origin: str,
        pin: OpsNnSourcePin,
        commit: str,
        tree: str,
        cann_version: str,
        target_soc: str,
        license_sha256: str,
    ) -> Tuple[OpsNnCandidate, ...]:
        root_names = set()
        for path in paths:
            parts = path.split("/")
            indexes = [
                index for index, part in enumerate(parts)
                if part in _INTEGRATION_DIRS
            ]
            if not indexes or indexes[0] < 2:
                continue
            root = "/".join(parts[:indexes[0]])
            root_names.add(root)

        roots: Dict[str, list[str]] = {
            root: [path for path in paths if path.startswith(root + "/")]
            for root in root_names
        }

        query_keys = {_normalize_name(query) for query in queries}
        discovered = []
        for root, root_paths in sorted(roots.items()):
            aliases = self._candidate_aliases(root, root_paths)
            if not query_keys.intersection(
                _normalize_name(alias) for alias in aliases
            ):
                continue
            discovered.append(
                self._candidate(
                    root=root,
                    paths=tuple(sorted(root_paths)),
                    aliases=aliases,
                    origin=origin,
                    pin=pin,
                    commit=commit,
                    tree=tree,
                    cann_version=cann_version,
                    target_soc=target_soc,
                    license_sha256=license_sha256,
                )
            )
        return tuple(discovered)

    @staticmethod
    def _candidate_aliases(root: str, paths: Sequence[str]) -> Tuple[str, ...]:
        basename = root.rsplit("/", 1)[-1]
        aliases = {basename, _VERSION_SUFFIX.sub("", basename)}
        for path in paths:
            filename = path.rsplit("/", 1)[-1]
            stem = filename.rsplit(".", 1)[0]
            lowered = stem.casefold()
            if lowered.startswith("aclnn_"):
                aliases.add(stem[6:])
            elif lowered.startswith("aclnn") and len(stem) > 5:
                aliases.add(stem[5:])
        return tuple(sorted(alias for alias in aliases if alias))

    def _candidate(
        self,
        *,
        root: str,
        paths: Sequence[str],
        aliases: Tuple[str, ...],
        origin: str,
        pin: OpsNnSourcePin,
        commit: str,
        tree: str,
        cann_version: str,
        target_soc: str,
        license_sha256: str,
    ) -> OpsNnCandidate:
        prefix = root + "/"
        relative = tuple(path[len(prefix):] for path in paths)
        has_op_api = any("op_api" in path.split("/") for path in relative)
        has_op_host = any("op_host" in path.split("/") for path in relative)
        has_op_kernel = any("op_kernel" in path.split("/") for path in relative)
        op_api_paths = tuple(
            prefix + path for path in relative
            if "op_api" in path.split("/")
            and path.rsplit(".", 1)[-1] in {"h", "hpp", "cpp", "cc"}
        )
        cmake_paths = tuple(
            prefix + path for path in relative
            if path.endswith("CMakeLists.txt") or path.endswith(".cmake")
        )
        config_clues = set()
        for path in relative:
            parts = path.split("/")
            for index, part in enumerate(parts[:-1]):
                if part == "config" and index + 1 < len(parts):
                    token = parts[index + 1].casefold()
                    if _SOC_TOKEN.fullmatch(token):
                        config_clues.add(token)
        cmake_clues = set()
        evidence_paths = set(cmake_paths)
        evidence_paths.update(
            prefix + path for path in relative
            if path == "README.md"
            or path.startswith("docs/aclnn")
            or (
                "op_api" in path.split("/")
                and path.rsplit(".", 1)[-1] in {"h", "hpp"}
            )
        )
        evidence_files = []
        for path in sorted(evidence_paths):
            content = self._blob(commit, path)
            evidence_files.append((path, hashlib.sha256(content).hexdigest()))
            if path in cmake_paths:
                text = content.decode("utf-8", "replace")
                cmake_clues.update(token.casefold() for token in _SOC_TOKEN.findall(text))
        op_tree = self._git_text("rev-parse", "--verify", "%s:%s" % (commit, root))
        if not _HEX_SHA1.fullmatch(op_tree) or self._git_text(
            "cat-file", "-t", op_tree
        ) != "tree":
            raise _GitReadError("operator path did not resolve to a tree")
        target_evidence = self._target_soc_evidence(
            target_soc, tuple(sorted(cmake_clues | config_clues))
        )
        identity = {
            "provider": "ops_nn_source",
            "origin": origin,
            "source_ref": pin.ref,
            "source_commit": commit,
            "source_tree": tree,
            "cann_version": cann_version,
            "license_sha256": license_sha256,
            "op_path": root,
            "op_tree": op_tree,
            "stability": "experimental" if root.startswith("experimental/") else "stable",
            "aliases": list(aliases),
            "has_op_api": has_op_api,
            "has_op_host": has_op_host,
            "has_op_kernel": has_op_kernel,
            "soc_cmake_clues": sorted(cmake_clues),
            "soc_config_clues": sorted(config_clues),
            "evidence_files": dict(evidence_files),
        }
        candidate_id = hashlib.sha256(
            json.dumps(
                identity, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        return OpsNnCandidate(
            provider="ops_nn_source",
            origin=origin,
            source_ref=pin.ref,
            source_commit=commit,
            source_tree=tree,
            cann_version=cann_version,
            license_name=OPS_NN_LICENSE,
            license_sha256=license_sha256,
            op_path=root,
            op_tree=op_tree,
            stability=identity["stability"],
            aliases=aliases,
            has_op_api=has_op_api,
            has_op_host=has_op_host,
            has_op_kernel=has_op_kernel,
            op_api_paths=op_api_paths,
            cmake_paths=cmake_paths,
            soc_cmake_clues=tuple(sorted(cmake_clues)),
            soc_config_clues=tuple(sorted(config_clues)),
            target_soc_evidence=target_evidence,
            evidence_files=tuple(evidence_files),
            candidate_id=candidate_id,
        )

    @staticmethod
    def _target_soc_evidence(target_soc: str, clues: Sequence[str]) -> str:
        target = _normalize_name(target_soc)
        normalized = tuple(_normalize_name(clue) for clue in clues)
        if target in normalized:
            return "exact"
        if any(target.startswith(clue) or clue.startswith(target) for clue in normalized):
            return "family"
        if normalized:
            return "mismatch"
        return "not_declared"


__all__ = [
    "DEFAULT_OPS_NN_SOURCE_PINS",
    "OFFICIAL_SOURCE_SCHEMA_VERSION",
    "OPS_NN_LICENSE",
    "OPS_NN_LICENSE_ID",
    "REVIEWED_SOURCE_SEARCH_ALIASES",
    "SOURCE_SEARCH_CONTRACT_REVISION",
    "OfficialSourceResolution",
    "OfficialSourceSemanticAudit",
    "OpsNnCandidate",
    "OpsNnSourcePin",
    "OpsNnSourceResolver",
]
