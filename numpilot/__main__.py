# This file is part of the NumPilot project at Harbin Institute of Technology (HIT).
#
# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# SPDX-License-Identifier: Apache-2.0
#
# Authors (accounts):
# - Su Tonghua <@sutonghua>
# - Li Xing <@li-xingHIT>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI: python -m numpilot <migrate|bootstrap|complete-gaps|doctor|...> ...

演示三步:
  1) bench-cpu <file>   纯 NumPy 基线，默认请求 BLAS 1 线程并记录实际线程池
  2) bench-npu <file>   审计迁移算子执行来源；仅严格 NPU 结果可与 1) 计算加速比
  migrate <file>        默认把普通纯 NumPy 文件做静态分类 + 已配置模型精确迁移
"""
import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import uuid


def parse_cpu_threads(val):
    """1(默认, 单核) / 0|all|max|full(全核, 不限制)→ int 或 None。"""
    if val is None:
        return 1
    if str(val).lower() in ("0", "all", "max", "full"):
        return None
    n = int(val)
    return n if n > 0 else None


def _early_pin_for_bench_cpu():
    """bench-cpu 是纯 numpy 进程(不碰 NPU/CANN),可安全地全局钉核。

    必须在 import numpy 之前设置线程环境变量；它们只是请求值，实际生效线程数
    由 benchmark 在 NumPy 加载后通过 threadpoolctl 观测并写入证据。
    bench-npu / migrate 不走这里，故只对 bench-cpu 生效。bench-npu 会另行记录
    自身进程中真实可观测的 BLAS 线程池，避免把“未限制”误写成“全线程”。
    """
    argv = sys.argv[1:]
    if not argv or argv[0] != "bench-cpu":
        return
    n = "1"
    if "--cpu-threads" in argv:
        i = argv.index("--cpu-threads")
        n = argv[i + 1] if i + 1 < len(argv) else "1"
    if parse_cpu_threads(n) is None:
        return                            # 0/all → 不限制
    n = str(parse_cpu_threads(n))
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[k] = n


_early_pin_for_bench_cpu()                # 在下面 import numpy(经 numpilot 传递)前完成

from numpilot import __version__                              # noqa: E402
from numpilot.backends.base import select_backend            # noqa: E402
from numpilot.asnumpy_skill import (                         # noqa: E402
    ASNUMPY_INTEGRATION_SKILL_FILES,
    ASNUMPY_OFFICIAL_ADAPTER_SKILL_FILES,
    AsnumpyIntegrationSkillError,
    load_asnumpy_integration_skill,
)
from numpilot.bench import run_cpu_baseline, run_npu_compare  # noqa: E402
from numpilot.capabilities import load_analysis_profile_set    # noqa: E402
from numpilot.cli_ui import (                                 # noqa: E402
    flow_print,
    render_doctor,
    render_home,
    render_paths,
    rich_available,
    terminal_status,
)
from numpilot.coverage import is_custom                       # noqa: E402
from numpilot.engines.llm import (LlmEngine, LlmError,        # noqa: E402
                                  LlmHttpError, LlmResponseError,
                                  LlmTransportError,
                                  llm_proxy_route_status)
from numpilot.env_check import Env                            # noqa: E402
from numpilot.llm_config import (LlmConfigError,               # noqa: E402
                                 OPENAI_CHAT_COMPLETIONS_PROTOCOL,
                                 configure_llm_interactive,
                                 init_llm_config_template,
                                 llm_config_path,
                                 load_llm_config,
                                 validate_llm_timeout)
from numpilot.model_contract import (                          # noqa: E402
    ModelContractError,
    require_model_status,
    validate_configured_model_receipt,
)
from numpilot.orchestrator import Orchestrator, _make_engine  # noqa: E402
from numpilot.paths import (default_benchmark_dir,                    # noqa: E402
                            default_bootstrap_receipt, default_demo_dir,
                            default_gap_dir, path_summary)
from numpilot.report import save_migration_report            # noqa: E402
from numpilot.script_migration import ScriptMigrationError   # noqa: E402


class RuntimeActivationConfigError(ValueError):
    """Raised for an incomplete or unsafe explicit qualification activation."""


class CannbotSkillError(RuntimeError):
    """Stable CLI boundary for optional CANNBot failures."""


class AnalyzeCliError(ValueError):
    """A static-analysis CLI input or publication boundary was rejected."""


def _load_cannbot_api():
    """Load optional gap-lane tooling without coupling ordinary migration."""

    from numpilot import cannbot_skills

    return cannbot_skills


_ASNUMPY_ADAPTER_MANIFEST_SCHEMA = "asnumpy.adapter.manifest.v1"
_ASNUMPY_ADAPTER_API_NAME_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")


def _resolve_asnumpy_adapter_api_name(adapter_data, spec):
    """Validate an adapter contract and return its canonical API name.

    Older locally signed artifacts expose ``api_name`` at the top level.  The
    generated AsNumpy adapter manifest uses the versioned v1 contract instead:
    the source/public qualified APIs live under ``operator`` and the Python
    public symbol lives under ``name_mapping.python``.  Both forms remain
    accepted, but every identity field that is present must agree with the
    requested gap.  Returning ``None`` keeps the caller's existing fail-closed
    error message and prevents a valid-looking manifest for another operator
    from being submitted to the model.
    """

    if not isinstance(adapter_data, Mapping):
        return None
    expected_api_name = getattr(spec, "api_name", None)
    expected_source_api = getattr(spec, "qualified_api", None)
    expected_public_api = getattr(spec, "asnumpy_api", None)
    if (
        not isinstance(expected_api_name, str)
        or not _ASNUMPY_ADAPTER_API_NAME_RE.fullmatch(expected_api_name)
        or not isinstance(expected_source_api, str)
        or not isinstance(expected_public_api, str)
    ):
        return None

    # Preserve the legacy contract exactly.  If a producer includes the old
    # field alongside a v1 manifest, it must still agree with the requested
    # gap rather than silently overriding a conflicting nested identity.
    if "api_name" in adapter_data:
        legacy_api_name = adapter_data.get("api_name")
        if (
            not isinstance(legacy_api_name, str)
            or not _ASNUMPY_ADAPTER_API_NAME_RE.fullmatch(legacy_api_name)
            or legacy_api_name != expected_api_name
        ):
            return None

    # A legacy object may carry unrelated metadata, so only enter the v1 path
    # when its nested identity fields are present.  This leaves the
    # pre-existing top-level ``api_name`` acceptance backward compatible even
    # if an older producer added unrelated top-level keys.
    has_nested_identity = any(
        key in adapter_data for key in ("operator", "name_mapping")
    )
    if not has_nested_identity:
        return expected_api_name if "api_name" in adapter_data else None
    if adapter_data.get("schema") != _ASNUMPY_ADAPTER_MANIFEST_SCHEMA:
        return None

    operator = adapter_data.get("operator")
    if not isinstance(operator, Mapping):
        return None
    if (
        operator.get("source_api") != expected_source_api
        or operator.get("public_api") != expected_public_api
    ):
        return None

    # The current producer emits name_mapping at the manifest root.  Accept
    # the equivalent operator.name_mapping spelling as well because both are
    # unambiguous v1 identities; if both are present, validate both so a
    # conflicting duplicate cannot bypass the check.
    name_mappings = [
        value
        for value in (adapter_data.get("name_mapping"), operator.get("name_mapping"))
        if value is not None
    ]
    if not name_mappings:
        return None
    for name_mapping in name_mappings:
        if not isinstance(name_mapping, Mapping):
            return None
        python_mapping = name_mapping.get("python")
        if not isinstance(python_mapping, Mapping):
            return None
        if python_mapping.get("public_symbol") != expected_api_name:
            return None
        # These fields are part of the v1 producer output.  Treat optional
        # qualified/private spellings as consistency checks when present,
        # while keeping the required identity to the three fields above.
        if (
            "public_qualified_name" in python_mapping
            and python_mapping.get("public_qualified_name") != expected_public_api
        ):
            return None
        if (
            "private_extension_symbol" in python_mapping
            and python_mapping.get("private_extension_symbol") != expected_api_name
        ):
            return None

    # CANNBot operator manifests produced for an injected/local candidate may
    # already carry the target toolchain and ACLNN boundary.  When those fields
    # are present, bind them before the adapter generator sees the artifact.
    # Minimal legacy v1 identity manifests remain accepted for compatibility.
    operator = adapter_data.get("operator")
    if (
        isinstance(operator, Mapping)
        and "target_soc" in operator
        and "cann_version" in operator
    ):
        try:
            from numpilot.gaps.generated_adapter import (
                validate_operator_manifest_for_overlay,
            )
            validate_operator_manifest_for_overlay(
                adapter_data,
                expected_source_api=expected_source_api,
                expected_public_api=expected_public_api,
                expected_target_soc=getattr(spec, "target_soc", None),
                expected_cann_version=getattr(spec, "cann_version", None),
            )
        except (ImportError, ValueError):
            return None
    return expected_api_name


def resolve_skill_bundle(*args, **kwargs):
    """Lazy compatibility wrapper retained for callers and test doubles."""

    cannbot = _load_cannbot_api()
    try:
        return cannbot.resolve_skill_bundle(*args, **kwargs)
    except cannbot.CannbotSkillError as exc:
        raise CannbotSkillError(str(exc)) from exc


def _qualified_runtime_inputs(args, backend_name):
    """Build explicit overlay/profile inputs without environment discovery."""

    overlay_path = getattr(args, "capability_overlay", None)
    evidence_root = getattr(args, "qualification_evidence_root", None)
    profiles_path = getattr(args, "analysis_profiles", None)
    if bool(overlay_path) != bool(evidence_root):
        raise RuntimeActivationConfigError(
            "--capability-overlay 与 --qualification-evidence-root 必须成对提供"
        )
    if backend_name != "npu" and any(
        (overlay_path, evidence_root, profiles_path)
    ):
        raise RuntimeActivationConfigError(
            "qualification overlay/profile 只允许用于 --backend npu"
        )
    profiles = None
    if profiles_path:
        try:
            profiles = load_analysis_profile_set(profiles_path)
        except (OSError, ValueError) as exc:
            raise RuntimeActivationConfigError(
                "analysis profile 加载失败: %s" % exc
            ) from exc
    if backend_name != "npu":
        return select_backend(backend_name), profiles
    if not overlay_path:
        return select_backend("npu"), profiles

    from numpilot.backends.npu import NpuBackend, OverlayRuntimeConfig

    return (
        NpuBackend(
            overlay_config=OverlayRuntimeConfig(
                overlay_path=overlay_path,
                evidence_root=evidence_root,
            )
        ),
        profiles,
    )


def _net_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def _render_terminal(res):
    """把迁移结果渲染成分段叙事:分析→迁移→卸载→执行→校验→性能→产物。"""
    analysis = res.get("analysis")
    net = _net_name(res.get("source_path", "<?>"))
    out = []
    out.append(f"── NumPilot 迁移 · {net} " + "─" * max(4, 34 - len(net)))

    # [分析] NumPy 调用点 + 显式 OperatorSpec 调用点 / 去重算子数
    if analysis is not None:
        ops = analysis.op_uses
        distinct = sorted({o.name for o in ops})
        domain_count = sum(o.kind == "domain_operator" for o in ops)
        numpy_count = len(ops) - domain_count
        call_summary = f"{numpy_count} 处 NumPy 调用"
        if domain_count:
            call_summary += f" + {domain_count} 处 OperatorSpec 调用"
        if res["status"] == "unsupported":
            out.append(f"[分析] {call_summary} · {len(distinct)} 个算子 · "
                       f"命中 {len(res['unsupported'])} 个阻断项 ✗")
            out.append("[缺口] " + ", ".join(res["unsupported"]) + "  (无法迁移)")
            return "\n".join(out)
        suffix = (
            "NumPy API 已覆盖；领域算子已显式映射（strict-NPU 仍需资格化） ✓"
            if domain_count else "全部在 asnumpy 覆盖集 ✓"
        )
        out.append(f"[分析] {call_summary} · {len(distinct)} 个算子 · {suffix}")

    # [迁移] 引擎 + 替换计数
    rewrites = res.get("rewrites") or {}
    total = sum(rewrites.values())
    domain_count = sum(
        op.kind == "domain_operator"
        for op in (analysis.op_uses if analysis is not None else [])
    )
    mapping = "np.*/OperatorSpec helper → ap.*" if domain_count else "np.* → ap.*"
    out.append(f"[迁移] 引擎={res.get('engine')} · 替换 {total} 处 {mapping}")
    if rewrites:
        items = "  ".join(f"{op}×{n}" for op, n in
                          sorted(rewrites.items(), key=lambda kv: (-kv[1], kv[0])))
        out.append("       " + items)

    # [卸载] CPU/NPU 分布
    plan = res.get("plan")
    if plan is not None and plan.placement:
        npu = sorted(o for o, p in plan.placement.items() if p == "NPU")
        cpu = sorted(o for o, p in plan.placement.items() if p == "CPU")
        out.append(
            f"[卸载] 策略={plan.strategy} · NPU: {', '.join(npu) or '—'}   "
            f"CPU: {', '.join(cpu) or '—'}"
        )

    rep = res["report"]

    # [执行] 模块命中不等于 strict NPU；必须同时展示 capability provenance。
    if rep.backend_version:
        loc = rep.backend_module or "?"
        out.append(f"[执行] 后端={rep.backend} · 已加载 asnumpy {rep.backend_version}")
        out.append(f"       {loc}")
    else:
        out.append(f"[执行] 后端={rep.backend} · shim 仿真(numpy 承载 asnumpy 语义,数值恒等)")
    provenance = rep.backend_provenance or {}
    if provenance:
        if provenance.get("strict_npu"):
            gate = "通过 ✓" if provenance.get("ok") else "失败 ✗"
        else:
            gate = "未执行(shim/非 strict)"
        out.append(
            f"       capability audit={provenance.get('audit_id', '?')} · "
            f"strict-NPU={gate}"
        )

    # [校验] 数值一致性
    mark = "✓" if rep.numerically_consistent else "✗"
    if rep.exception_type:
        out.append(
            f"[校验] 失败 ✗  {rep.exception_phase}: "
            f"{rep.exception_type}: {rep.exception_message}"
        )
    else:
        out.append(
            f"[校验] 数值一致 {mark}  max_abs={rep.max_abs_diff:.2e}  "
            f"max_norm={rep.max_rel_diff:.2e}  "
            f"(atol={rep.atol:.0e}, rtol={rep.rtol:.0e})"
        )
        out.append(
            "       同输入计算主体=%s · 产物输入=%s · 产物端到端=%s"
            % (rep.kernel_consistent, rep.input_consistent, rep.artifact_consistent)
        )

    # [产物]
    arts = [res["migrated_path"]] if res.get("migrated_path") else []
    if arts:
        out.append("[产物] " + " · ".join(arts))

    # [性能] 交给两个独立 bench 程序(公平口径:numpy 单核 vs NPU 满线程)
    src = res.get("source_path", "<file>")
    bench_dir = default_benchmark_dir()
    out.append("[性能] 本命令不发布加速比；请运行独立、fail-closed 的两步评测:")
    out.append(f"       ① python -m numpilot bench-cpu {src} --out {bench_dir}")
    out.append(f"       ② python -m numpilot bench-npu {src} --out {bench_dir}")
    return "\n".join(out)


def _resolve_backend_name(args):
    return args.backend or os.getenv("NUMPILOT_BACKEND", "shim")


def _llm_config_arg(value):
    """Validate a CLI config path before it can influence model loading."""

    try:
        return str(llm_config_path(value))
    except LlmConfigError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


_ANALYZE_MAX_SOURCE_BYTES = 4 * 1024 * 1024
_ANALYZE_AUDIT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:-]{0,255}$")


def _read_analyze_source(value):
    """Read one bounded regular UTF-8 file without importing or executing it."""

    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or len(value) > 4096
    ):
        raise AnalyzeCliError("source path is invalid")
    if _analyze_path_has_symlink(value):
        raise AnalyzeCliError("source path contains a symlink")
    source_path = os.path.abspath(os.path.expanduser(value))
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(source_path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AnalyzeCliError("source must be a regular file: %s" % source_path)
        if before.st_size > _ANALYZE_MAX_SOURCE_BYTES:
            raise AnalyzeCliError(
                "source exceeds the %d-byte analysis limit"
                % _ANALYZE_MAX_SOURCE_BYTES
            )
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            raw = handle.read(_ANALYZE_MAX_SOURCE_BYTES + 1)
            after = os.fstat(handle.fileno())
        if len(raw) > _ANALYZE_MAX_SOURCE_BYTES:
            raise AnalyzeCliError(
                "source exceeds the %d-byte analysis limit"
                % _ANALYZE_MAX_SOURCE_BYTES
            )
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise AnalyzeCliError("source changed while it was being read")
    except AnalyzeCliError:
        raise
    except OSError as exc:
        raise AnalyzeCliError(
            "source cannot be read: %s (%s)"
            % (source_path, type(exc).__name__)
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        return raw.decode("utf-8"), source_path
    except UnicodeDecodeError as exc:
        raise AnalyzeCliError("source must be valid UTF-8: %s" % source_path) from exc


def _analyze_paths_alias(source_path, output_path):
    source = os.path.abspath(source_path)
    output = os.path.abspath(str(output_path))
    if os.path.realpath(source) == os.path.realpath(output):
        return True
    if os.path.lexists(output):
        try:
            return os.path.samefile(source, output)
        except OSError:
            return False
    return False


def _analyze_path_has_symlink(value):
    """Check a path's lexical components before ``abspath`` can hide ``..``.

    A spelling such as ``link/../file`` is normalized by ``abspath`` before
    ordinary component checks, although the kernel follows ``link`` first.
    Walk the original components and reject any existing symlink while still
    allowing a normal relative path such as ``nested/../file``.
    """

    try:
        rendered = os.fspath(value)
        if not isinstance(rendered, str):
            return True
        expanded = os.path.expanduser(rendered)
        current = (
            Path(os.path.abspath(os.sep))
            if os.path.isabs(expanded)
            else Path(os.getcwd())
        )
        for component in expanded.split(os.sep):
            if not component or component == ".":
                continue
            if component == "..":
                current = current.parent
                continue
            candidate = current / component
            if os.path.lexists(candidate) and candidate.is_symlink():
                return True
            current = candidate
    except (OSError, RuntimeError, TypeError, ValueError):
        return True
    return False


def _analyze_parent_has_symlink(path):
    return _analyze_path_has_symlink(path)


def _write_analyze_receipt(output_value, source_path, payload):
    """Atomically publish one private JSON receipt without replacing a path."""

    if (
        not isinstance(output_value, str)
        or not output_value
        or "\x00" in output_value
        or len(output_value) > 4096
    ):
        raise AnalyzeCliError("output path is invalid")
    # Check source/output aliasing before rejecting symlink spelling.  A
    # symlink output pointing at the source must retain the established,
    # source-specific diagnostic (and, importantly, is never modified).
    output = Path(os.path.abspath(os.path.expanduser(output_value)))
    if _analyze_paths_alias(source_path, output):
        raise AnalyzeCliError("output aliases the source file: %s" % output)
    if _analyze_path_has_symlink(output_value):
        raise AnalyzeCliError("output parent must be a real directory (symlink)")
    if os.path.lexists(output):
        raise AnalyzeCliError("analysis output already exists: %s" % output)

    parent = output.parent
    if _analyze_parent_has_symlink(parent):
        raise AnalyzeCliError("output parent must be a real directory: %s" % parent)
    parent_existed = os.path.lexists(parent)
    if parent_existed and not parent.is_dir():
        raise AnalyzeCliError("output parent must be a real directory: %s" % parent)
    try:
        if not parent_existed:
            parent.mkdir(parents=True, mode=0o700)
        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        directory_fd = os.open(str(parent), directory_flags)
    except OSError as exc:
        raise AnalyzeCliError(
            "analysis output parent cannot be prepared: %s (%s)"
            % (parent, type(exc).__name__)
        ) from exc

    temporary = ""
    descriptor = -1
    try:
        if parent_existed:
            mode = stat.S_IMODE(os.fstat(directory_fd).st_mode)
            if mode != 0o700:
                raise AnalyzeCliError(
                    "existing analysis output parent mode must be 0700: %s"
                    % parent
                )
        else:
            os.fchmod(directory_fd, 0o700)
        encoded = (
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        descriptor, temporary = tempfile.mkstemp(
            prefix=".%s-" % (output.name or "analysis"),
            suffix=".tmp",
            dir=str(parent),
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError as exc:
            raise AnalyzeCliError(
                "analysis output already exists: %s" % output
            ) from exc
        os.unlink(temporary)
        temporary = ""
        os.fsync(directory_fd)
    except AnalyzeCliError:
        raise
    except OSError as exc:
        raise AnalyzeCliError(
            "analysis output cannot be published: %s (%s)"
            % (output, type(exc).__name__)
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass
        os.close(directory_fd)
    return str(output)


def _runtime_audit_id(runtime_identity):
    if isinstance(runtime_identity, Mapping):
        value = runtime_identity.get("audit_id")
    else:
        value = getattr(runtime_identity, "audit_id", None)
    if isinstance(value, str) and _ANALYZE_AUDIT_ID.fullmatch(value):
        return value
    return None


def _render_qualified_analysis(payload, output_path=None):
    apis = payload["qualified_apis"]
    rendered = ["── NumPilot 静态迁移分析 ──"]
    rendered.append("[源码] %s" % payload["source_path"])
    rendered.append("[SHA-256] %s" % payload["source_sha256"])
    rendered.append(
        "[导入] %s" % (", ".join(payload["imports"]) or "—")
    )
    rendered.append(
        "[调用点] %d · qualified APIs=%s"
        % (len(payload["calls"]), ", ".join(apis) or "—")
    )
    rendered.append(
        "[判定] %s · runtime=%s · identity=%s"
        % (
            payload["decision"],
            payload["runtime_identity_status"],
            payload["runtime_identity_sha256"] or "unavailable",
        )
    )
    rendered.append("[下一步] %s" % payload["next_action"])
    if payload["issues"]:
        rendered.append("[阻断] %d 项；完整原因请查看 JSON" % len(payload["issues"]))
    rendered.append(
        "[副作用] actual_remote_call=false · execution_performed=false"
    )
    if output_path is not None:
        rendered.append("[回执] %s" % output_path)
    return "\n".join(rendered)


def _cmd_analyze(args):
    """Run the inert, runtime-qualified admission analysis for one script."""

    from numpilot.backends.npu import NpuBackend
    from numpilot.qualified_analysis import (
        QualifiedAnalysisError,
        analyze_qualified_script,
    )

    try:
        source, source_path = _read_analyze_source(args.file)
        with terminal_status(
            "读取当前 AsNumpy / NPU 资格身份",
            enabled=not args.json,
        ) as progress:
            try:
                runtime_identity = NpuBackend().runtime_identity()
            except Exception:
                # A probe failure is an audit-required fact, not a reason to execute
                # the source or silently switch to a less strict runtime.
                runtime_identity = {
                    "status": "audit_required",
                    "identity_sha256": None,
                }
                progress.warning()
        analysis = analyze_qualified_script(
            source,
            source_path,
            runtime_identity=runtime_identity,
        )
        payload = analysis.to_dict()
        payload["producer"] = "numpilot-qualified-analysis-cli-v1"
        payload["runtime_audit_id"] = _runtime_audit_id(runtime_identity)
        payload["qualified_apis"] = sorted({
            call["canonical_key"] for call in payload["calls"]
        })
        output_path = None
        if args.out is not None:
            output_path = _write_analyze_receipt(
                args.out, source_path, payload
            )
    except (AnalyzeCliError, QualifiedAnalysisError) as exc:
        print("[analyze] %s" % exc, file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ))
    else:
        print(_render_qualified_analysis(payload, output_path=output_path))
    if args.require_direct and payload["decision"] != "DIRECT":
        return 1
    return 0


def _cmd_verify_script(args):
    """Explicitly execute and compare one published ordinary migration."""

    from numpilot.script_verification import (
        ScriptVerificationError,
        verify_script,
    )

    try:
        with terminal_status(
            "在隔离进程中执行原脚本与迁移产物",
            enabled=not args.json,
        ) as progress:
            receipt = verify_script(
                args.file,
                args.migration_report,
                args.outputs,
                args.out,
                migrated_path=args.migrated,
                timeout_seconds=args.timeout,
                max_output_bytes=args.max_output_bytes,
                source_python_executable=args.source_python,
                migrated_python_executable=args.migrated_python,
            )
            if receipt.status != "passed":
                progress.error()
    except ScriptVerificationError as exc:
        print(
            "[普通脚本验证] %s: %s" % (exc.code, exc),
            file=sys.stderr,
        )
        receipt = getattr(exc, "receipt", None)
        if receipt is None:
            print(
                "[执行边界] execution_not_admitted_or_receipt_not_published",
                file=sys.stderr,
            )
            return 2
        payload = receipt.to_dict()
        if args.json:
            print(json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            ))
        else:
            print("[回执] %s" % receipt.path)
        return 1

    payload = receipt.to_dict()
    if args.json:
        print(json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ))
    else:
        print("── NumPilot ordinary script 数值验证 ──")
        print("[结果] status=%s" % receipt.status)
        print("[输出合同] %s" % ", ".join(args.outputs))
        print(
            "[执行] subprocess_isolation=true · os_sandbox=false · "
            "explicit_execution=true"
        )
        print(
            "[证据边界] numerical_only=true · strict_npu_verified=false · "
            "performance_valid=false"
        )
        print("[回执] %s" % receipt.path)
    return 0 if receipt.status == "passed" else 1


def _cmd_benchmark_script(args):
    """Benchmark one strict ordinary migration in isolated CPU/NPU workers."""

    from numpilot.script_benchmark import (
        ScriptBenchmarkError,
        benchmark_script,
    )

    try:
        with terminal_status(
            "执行单核 CPU / strict-NPU 设备驻留评测",
            enabled=not args.json,
        ) as progress:
            receipt = benchmark_script(
                args.file,
                args.migration_report,
                args.outputs,
                args.out,
                migrated_path=args.migrated,
                marker=args.marker,
                core=args.core,
                warmup=args.warmup,
                samples=args.samples,
                timeout_seconds=args.timeout,
                max_output_bytes=args.max_output_bytes,
                source_python_executable=args.source_python,
                migrated_python_executable=args.migrated_python,
            )
            if receipt.status != "passed":
                progress.error()
    except ScriptBenchmarkError as exc:
        print(
            "[普通脚本性能] %s: %s" % (exc.code, exc),
            file=sys.stderr,
        )
        receipt = getattr(exc, "receipt", None)
        if receipt is None:
            print(
                "[性能边界] benchmark_not_admitted_or_receipt_not_published",
                file=sys.stderr,
            )
            return 2
        payload = receipt.to_dict()
        if args.json:
            print(json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            ))
        else:
            print("[失败回执] %s" % receipt.path)
            print("[性能声明] performance_valid=false · speedup=null")
        return 1

    payload = receipt.to_dict()
    if args.json:
        print(json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ))
    else:
        performance = payload["performance"]
        print("── NumPilot ordinary script strict-NPU 性能验证 ──")
        print("[结果] status=%s · performance_valid=true" % receipt.status)
        print(
            "[稳态] CPU %.3f ms · NPU %.3f ms · median %.2fx · conservative %.2fx"
            % (
                performance["cpu_median_ms"],
                performance["npu_median_ms"],
                performance["median_speedup"],
                performance["conservative_cpu_p10_over_npu_p90"],
            )
        )
        print(
            "[口径] device-resident marker suffix；不含冷启动、输入准备和最终 D2H"
        )
        print("[回执] %s" % receipt.path)
    return 0


def _workflow_log_tail(path, *, lines=8):
    """Return a small, display-safe tail of a private workflow log."""

    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    # Workflow stderr is produced by a child process (and may ultimately
    # contain model/user-controlled bytes).  Do not pass terminal control
    # sequences, carriage returns, bidi marks, or other format characters
    # through the human-readable CLI.  Replacing rather than dropping them
    # preserves enough shape for diagnosis while making the output safe to
    # paste into a terminal/log viewer.  JSON mode keeps the original bounded
    # log file and hashes; this sanitization is presentation-only.
    safe_chars = []
    for character in text:
        if character in {"\n", "\t"} or (
            character.isprintable()
            and not unicodedata.category(character).startswith("C")
        ):
            safe_chars.append(character)
        else:
            safe_chars.append("�")
    values = "".join(safe_chars).splitlines()
    if len(values) > lines:
        values = values[-lines:]
    return "\n".join(values)


def _cmd_run(args):
    """Run the product-facing ordinary chain with one user command."""

    from numpilot.workflow import WorkflowError, run_ordinary_workflow

    try:
        # ``None`` means "choose the private default"; an explicitly empty
        # value is still a caller-provided path and must be rejected by the
        # workflow admission contract rather than silently changing roots.
        if args.out is not None:
            if isinstance(args.out, str) and not args.out:
                raise ValueError("empty output path")
            out_dir = Path(args.out).expanduser()
        else:
            root = default_demo_dir()
            # ``run_ordinary_workflow`` deliberately admits only a non-existing
            # output path.  Generate a collision-resistant name here and let the
            # workflow create both the parent and boundary atomically; pre-creating
            # either one would make the normal one-command path look like an unsafe
            # retry and would turn a default-directory error into an uncaught
            # exception before WorkflowError handling is installed.
            out_dir = root / ("workflow-" + uuid.uuid4().hex)
    except (OSError, RuntimeError, TypeError, ValueError):
        # Path expansion/default-root resolution happens before the workflow
        # can translate filesystem failures into WorkflowError.  Keep this
        # CLI boundary stable and deliberately omit the raw path/exception,
        # which may contain user or environment details.
        print(
            "[一键流水线] invalid_path: workflow output path is invalid",
            file=sys.stderr,
        )
        return 2

    try:
        with terminal_status(
            "执行一键流水线：分析 → 模型迁移 → 数值验证"
            + (" → strict-NPU 性能" if args.benchmark else ""),
            enabled=not args.json,
        ) as progress:
            result = run_ordinary_workflow(
                args.file,
                out_dir,
                backend=args.backend,
                fallback_policy=args.fallback_policy,
                target_soc=args.target_soc,
                cann_version=args.cann_version,
                outputs=args.outputs or ("result",),
                benchmark=bool(args.benchmark),
                warmup=args.warmup,
                samples=args.samples,
                core=args.core,
                skip_env_check=bool(args.skip_env_check),
                timeout_seconds=args.timeout,
                python_executable=args.python,
            )
            if result["status"] != "passed":
                progress.error()
    except WorkflowError as exc:
        print("[一键流水线] %s: %s" % (exc.code, exc), file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ))
    else:
        print("── NumPilot 一键普通脚本流水线 ──")
        print("[源码] %s" % result["source_path"])
        print("[运行目录] %s" % out_dir.absolute())
        print(
            "[策略] backend=%s · engine=llm · fallback=%s · benchmark=%s"
            % (result["backend"], result["fallback_policy"], result["benchmark_requested"])
        )
        for stage in result["stages"]:
            marker = {"passed": "✓", "failed": "✗", "blocked": "✗", "timeout": "!"}.get(
                stage["status"], "?"
            )
            print(
                "[%s] %s · status=%s · %.2fs"
                % (marker, stage["name"], stage["status"], stage["elapsed_seconds"])
            )
            if stage["status"] not in {"passed"}:
                tail = _workflow_log_tail(stage["stderr_path"])
                if tail:
                    print("    stderr:")
                    for line in tail.splitlines():
                        print("      " + line)
                print("    日志=%s" % stage["stderr_path"])
        claims = result["claims"]
        print(
            "[事实] model_generated=%s · numerically_verified=%s · "
            "strict_npu_performance_verified=%s"
            % (
                claims["model_generated"],
                claims["numerically_verified"],
                claims["strict_npu_performance_verified"],
            )
        )
        print("[下一步] %s" % result["next_action"])
        print("[回执] %s" % result["workflow_report"])
    return 0 if result["status"] == "passed" else (1 if result["status"] == "blocked" else 2)


def _cmd_replay(args):
    """Render a read-only summary of a hash-bound workflow directory."""

    from numpilot.workflow import WorkflowError, replay_workflow

    try:
        result = replay_workflow(args.run_dir)
    except WorkflowError as exc:
        print("[replay] %s: %s" % (exc.code, exc), file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True))
    else:
        integrity = result["receipt_integrity"]
        print("── NumPilot workflow replay（只读）──")
        print("[运行目录] %s" % result["run_dir"])
        print("[状态] workflow=%s · receipt_integrity=%s" % (
            result["workflow_status"], integrity["status"],
        ))
        for stage in result["stages"]:
            print("[%s] %s" % (stage["status"], stage["name"]))
        print("[执行] execution_performed=false · actual_remote_call=false")
        print("[回执] %s" % result["workflow_report"])
        if integrity["errors"]:
            for error in integrity["errors"]:
                print("[完整性错误] %s" % error, file=sys.stderr)
    return 0 if result["receipt_integrity"]["status"] == "passed" else 1


# ---------------------------------------------------------------------------
# Interactive ordinary-script entry point
# ---------------------------------------------------------------------------
#
# The shell workflow wrapper deliberately stays thin: it validates the
# repository Skill, activates the user's private runtime, and then delegates
# here.  Keeping the conversation and receipt choreography in the framework
# means an arbitrary source path gets the same stages as a future CodeArts or
# API caller, rather than a second, demo-only implementation in bash.


def _interactive_private_json(path, payload, *, replace=False):
    """Write one bounded private JSON file without exposing model text."""

    target = Path(path)
    if not target.is_absolute() or target.is_symlink() or target.exists() and not replace:
        raise ValueError("interactive receipt path is not write-once")
    parent_existed = target.parent.exists()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise ValueError("interactive receipt parent is invalid")
    if os.name == "posix":
        if parent_existed and stat.S_IMODE(target.parent.stat().st_mode) & 0o077:
            raise ValueError("interactive receipt parent must be private")
        if not parent_existed:
            os.chmod(target.parent, 0o700)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(encoded) > 512 * 1024:
        raise ValueError("interactive receipt is oversized")
    fd = -1
    temporary = ""
    try:
        fd, temporary = tempfile.mkstemp(
            prefix="." + target.name + "-", suffix=".tmp", dir=str(target.parent)
        )
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
            os.unlink(temporary)
        temporary = ""
        return target
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def _interactive_read_json(path, *, maximum=8 * 1024 * 1024):
    """Read a framework receipt defensively for a human summary."""

    candidate = Path(path)
    try:
        if candidate.is_symlink() or not candidate.is_file():
            return None
        payload = candidate.read_bytes()
        if len(payload) > maximum:
            return None
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError, ValueError):
        return None
    return value if isinstance(value, Mapping) else None


def _interactive_heartbeat(stop, label, *, interval=5.0):
    """Return a daemon thread which makes a long remote/stage wait visible."""

    started = time.monotonic()

    def beat():
        while not stop.wait(interval):
            elapsed = int(time.monotonic() - started)
            flow_print(
                "[进行中] %s · 已等待 %ss（仍在等待真实回执，未假定成功）"
                % (label, elapsed),
                tone="progress",
                flush=True,
            )

    thread = threading.Thread(target=beat, name="numpilot-interactive-heartbeat", daemon=True)
    thread.start()
    return thread


def _interactive_display_text(value, *, limit=512):
    """Bound source-derived text before it reaches a human terminal."""

    try:
        text = str(value)
    except Exception:
        text = "?"
    safe = []
    for character in text[: max(1, int(limit))]:
        if character in {"\n", "\t"} or (
            character.isprintable()
            and not unicodedata.category(character).startswith("C")
        ):
            safe.append(character)
        else:
            safe.append("�")
    return "".join(safe)


def _interactive_source_arg(args):
    """Resolve a positional/option source path, prompting only on a TTY."""

    positional = getattr(args, "file", None)
    option = getattr(args, "source_option", None)
    if positional and option:
        raise ValueError("source path must be provided once")
    value = positional or option
    if value:
        return value
    if not sys.stdin.isatty():
        raise ValueError("非交互环境必须显式提供 SOURCE 或 --source")
    try:
        flow_print(
            "请输入待迁移 Python 代码的绝对路径：",
            tone="prompt",
            end="",
            flush=True,
        )
        value = input().strip()
    except (EOFError, KeyboardInterrupt) as exc:
        raise ValueError("未提供待迁移源码路径") from exc
    if not value:
        raise ValueError("未提供待迁移源码路径")
    return value


def _interactive_runtime_analysis(snapshot, args):
    """Run the same inert qualification/scanner used by ordinary workflow."""

    from numpilot.backends.npu import NpuBackend
    from numpilot.qualified_analysis import analyze_qualified_script
    from numpilot.script_migration import classify_script, scan_numpy_script

    backend_name = getattr(args, "backend", "npu") or "npu"
    backend, _profiles = _qualified_runtime_inputs(args, backend_name)
    runtime_identity = None
    try:
        provider = getattr(backend, "runtime_identity", None)
        runtime_identity = provider() if callable(provider) else None
    except Exception:
        runtime_identity = {"status": "audit_required", "identity_sha256": None}
    try:
        qualified = analyze_qualified_script(
            snapshot.text,
            snapshot.path,
            runtime_identity=runtime_identity,
        )
        qualified_payload = qualified.to_dict()
    except Exception as exc:
        # Keep the local scanner useful even when a platform identity probe is
        # unavailable.  The ordinary workflow will still enforce its own
        # qualification gate before execution.
        qualified = None
        qualified_payload = {
            "analysis_mode": "STATIC_AST_ONLY",
            "source_path": snapshot.path,
            "source_sha256": snapshot.sha256,
            "decision": "CAPABILITY_AUDIT_REQUIRED",
            "runtime_identity_status": "audit_required",
            "runtime_identity_sha256": None,
            "issues": ["qualified analysis unavailable: %s" % type(exc).__name__],
            "calls": [],
            "imports": [],
            "required_rewrites": [],
            "next_action": "complete the capability/runtime audit before migration",
            "actual_remote_call": False,
            "execution_performed": False,
        }
    qualified_payload["qualified_apis"] = sorted({
        str(item.get("canonical_key"))
        for item in qualified_payload.get("calls", [])
        if isinstance(item, Mapping) and item.get("canonical_key")
    })

    scan = scan_numpy_script(
        snapshot.text,
        snapshot.path,
        qualified_analysis=qualified,
    )
    classification = classify_script(
        scan,
        backend,
        qualified_analysis=qualified,
    )
    return qualified_payload, scan, classification


def _interactive_static_source_admission(
    snapshot, qualified_payload, scan, registered_add_one
):
    """Admit only a clean scan or the exact, safely lowered add-one gap."""

    from numpilot.add_one_gap import (
        AddOneGapError,
        validate_add_one_source_admission,
    )

    raw_qualified_issues = qualified_payload.get("issues", ())
    qualified_issues = (
        tuple(str(item) for item in raw_qualified_issues)
        if isinstance(raw_qualified_issues, (list, tuple))
        else ("qualified analysis returned malformed issues",)
    )
    raw_scan_issues = getattr(scan, "issues", ())
    scan_issues = (
        tuple(str(item) for item in raw_scan_issues)
        if isinstance(raw_scan_issues, (list, tuple))
        else ("ordinary scanner returned malformed issues",)
    )
    scanner_ok = getattr(scan, "ok", False) is True
    if not qualified_issues and scanner_ok and not scan_issues:
        return {
            "ok": True,
            "route": "ordinary_scanner",
            "issues": [],
        }
    if (
        not qualified_issues
        and not scanner_ok
        and scan_issues
        and registered_add_one
    ):
        try:
            validate_add_one_source_admission(
                source_path=snapshot.path,
                source=snapshot.text,
                calls=registered_add_one,
                scanner_issues=scan_issues,
            )
        except AddOneGapError as exc:
            admission_issue = str(exc)
        else:
            return {
                "ok": True,
                "route": "registered_add_one_exact_lowering",
                "issues": [],
                "waived_scanner_issues": list(scan_issues),
            }
    else:
        admission_issue = "static scanner did not pass"
    return {
        "ok": False,
        "route": "blocked",
        "issues": list(dict.fromkeys(
            qualified_issues + scan_issues + (admission_issue,)
        )),
    }


def _interactive_render_analysis(snapshot, qualified_payload, scan, classification):
    """Render bounded source/API facts, including concrete missing operators."""

    import ast

    from numpilot.operator_specs import collect_static_operator_specs
    from numpilot.add_one_gap import add_one_gap_summary, detect_add_one_calls

    flow_print("\n── NumPilot 迁移前分析（只读，不执行源码）──", tone="title")
    flow_print(
        "[源码] %s · sha256=%s · %d bytes · %d lines"
        % (_interactive_display_text(snapshot.path), snapshot.sha256, snapshot.bytes, snapshot.lines),
        tone="artifact",
    )
    flow_print(
        "[扫描] imports=%d · NumPy 调用/引用=%d · scanner_ok=%s"
        % (
            len(qualified_payload.get("imports", [])),
            len(getattr(scan, "uses", ()) or ()),
            "yes" if getattr(scan, "ok", False) else "no",
        )
    )
    flow_print(
        "[资格] decision=%s · runtime=%s · identity=%s"
        % (
            _interactive_display_text(qualified_payload.get("decision", "unknown")),
            _interactive_display_text(qualified_payload.get("runtime_identity_status", "unknown")),
            _interactive_display_text(qualified_payload.get("runtime_identity_sha256") or "unavailable"),
        )
    )
    decisions = tuple(getattr(classification, "decisions", ()) or ())
    if not decisions:
        flow_print("[API] 未识别到可迁移的 np.* 调用点；后续会由框架门禁给出具体结论。")
    else:
        for decision in decisions:
            callsites = tuple(getattr(decision, "callsites", ()) or ())
            lines = ",".join(str(getattr(item, "lineno", "?")) for item in callsites) or "?"
            status = _interactive_display_text(getattr(decision, "status", "unknown"))
            api = _interactive_display_text(getattr(decision, "qualified_api", "numpy.?"))
            provenance = _interactive_display_text(getattr(decision, "provenance", "unknown"))
            flow_print(
                "[API] %-28s line=%-8s status=%-24s provenance=%s"
                % (api, lines, status, provenance)
            )
    # OperatorSpec is a declarative source contract.  Parse it as AST data so
    # the workflow explicitly connects an observed custom gap to the operator
    # shape that CANNBot will be asked to draft, without importing the source.
    try:
        static_specs = collect_static_operator_specs(ast.parse(snapshot.text, filename=snapshot.path))
    except (SyntaxError, TypeError, ValueError):
        static_specs = None
    if static_specs is not None:
        for oracle, spec in sorted(static_specs.specs.items()):
            signature = ", ".join(
                "%s positional + %s keyword"
                % (
                    item.get("positional_args", "?"),
                    ",".join(str(key) for key in item.get("keyword_args", ())) or "none",
                )
                for item in spec.signature_variants
            )
            flow_print(
                "[自定义缺口] operator.%s -> %s · oracle=%s · route=%s"
                % (
                    _interactive_display_text(spec.name),
                    _interactive_display_text(spec.target_api),
                    _interactive_display_text(oracle),
                    _interactive_display_text(spec.route),
                )
            )
            flow_print(
                "  待补算子契约：dtype=%s · shape=%s · signature=%s · 官方等价性=待证明；"
                "仅允许生成 draft，不会自动激活。"
                % (
                    _interactive_display_text(", ".join(spec.dtype_rules)),
                    _interactive_display_text(", ".join(spec.shape_rules)),
                    _interactive_display_text(signature),
                )
            )
        for oracle, reason in sorted(static_specs.errors.items()):
            flow_print(
                "[自定义缺口] operator.%s 声明无效：%s"
                % (_interactive_display_text(oracle), _interactive_display_text(reason)),
                tone="error",
            )
    registered_add_one = detect_add_one_calls(snapshot.text)
    registered_lines = {item.lineno for item in registered_add_one}
    gaps = []
    for decision in decisions:
        available = getattr(decision, "available", None)
        kind = getattr(decision, "kind", "")
        if kind == "call" and available is False:
            qualified_api = str(getattr(decision, "qualified_api", ""))
            callsites = tuple(getattr(decision, "callsites", ()) or ())
            if (
                qualified_api == "numpy.add"
                and callsites
                and all(getattr(item, "lineno", None) in registered_lines for item in callsites)
            ):
                continue
            gaps.append(decision)
    if not gaps:
        # A qualified-analysis GAP_REQUIRED call can be present even when the
        # legacy scanner did not classify it as an ordinary-script decision.
        for call in qualified_payload.get("calls", []):
            if isinstance(call, Mapping) and call.get("decision") == "GAP_REQUIRED":
                source_api = str(call.get("source_api") or call.get("canonical_key") or "")
                if source_api == "numpy.add" and call.get("line") in registered_lines:
                    continue
                gaps.append(call)
    if registered_add_one:
        registered = add_one_gap_summary(registered_add_one)
        gaps.append(registered)
        lines = ",".join(str(item.lineno) for item in registered_add_one)
        flow_print(
            "[缺口] numpy.add(array, 1.0) @ line %s：当前 AsNumpy 的 add 仅接收两个已证明数组；"
            "缺少 asnumpy.add_one(x)。" % lines,
            tone="warning",
        )
        flow_print(
            "      待补语义：float32 contiguous，输出 y=x+1.0、shape 不变；"
            "候选 Ascend C 实现使用逐元素 Adds。",
            tone="warning",
        )
        flow_print(
            "      功能优先接入：CANNBot 先生成未验证草稿；NumPilot 再以"
            " asnumpy.ones_like + asnumpy.add 做独立真机资格验证。",
            tone="warning",
        )
    ordinary_gaps = [
        item
        for item in gaps
        if not (isinstance(item, Mapping) and item.get("kind") == "registered_signature_gap")
    ]
    if ordinary_gaps:
        flow_print(
            "[缺口] 另发现 %d 个尚无已注册接入路径的 API：" % len(ordinary_gaps),
            tone="warning",
        )
        for item in ordinary_gaps:
            if isinstance(item, Mapping):
                api = _interactive_display_text(item.get("source_api", "numpy.?"))
                line = item.get("line", "?")
                facts = item.get("facts") if isinstance(item.get("facts"), Mapping) else {}
                reason = _interactive_display_text(item.get("reason_code", "API_UNAVAILABLE"))
                sig = "%s positional + %s keyword" % (
                    facts.get("positional_args", "?"),
                    ",".join(str(x) for x in facts.get("keyword_args", ())) or "none",
                )
            else:
                api = _interactive_display_text(getattr(item, "qualified_api", "numpy.?"))
                line = getattr((getattr(item, "callsites", ()) or (None,))[0], "lineno", "?")
                reason = _interactive_display_text(getattr(item, "reason_code", "API_UNAVAILABLE"))
                sig = "%s positional + %s keyword" % (
                    getattr((getattr(item, "callsites", ()) or (None,))[0], "positional_args", "?"),
                    ",".join(str(x) for x in getattr((getattr(item, "callsites", ()) or (None,))[0], "keyword_args", ())) or "none",
                )
            api_name = _interactive_display_text(api.removeprefix("numpy."))
            flow_print(
                "    - %s @ line %s：AsNumpy 缺少 asnumpy.%s（%s）"
                % (api, line, api_name, reason),
                tone="warning",
            )
            flow_print(
                "      待补算子：asnumpy.%s；先绑定语义/签名（%s），再走官方算子解析；"
                "无官方匹配才允许 CANNBot draft。" % (api_name, sig),
                tone="warning",
            )
    elif not registered_add_one:
        flow_print("[缺口] 当前静态扫描未发现缺口；不会凭空生成算子。", tone="success")
    issues = tuple(getattr(scan, "issues", ()) or ())
    if issues:
        flow_print(
            "[扫描提示] %s"
            % _interactive_display_text("；".join(str(item) for item in issues[:4])),
            tone="warning",
        )
        if len(issues) > 4:
            flow_print(
                "              … 其余 %d 项写入预检回执" % (len(issues) - 4),
                tone="warning",
            )
    return gaps


def _interactive_gap_request(snapshot, classification, args, session_root):
    """Persist a canonical ordinary-script gap request, if one is possible."""

    from numpilot.script_migration import build_gap_request

    unsupported = tuple(getattr(classification, "unsupported", ()) or ())
    if not unsupported:
        return None, ()
    try:
        payload = build_gap_request(
            classification,
            unsupported,
            target_soc=args.target_soc,
            cann_version=args.cann_version,
        )
    except Exception as exc:
        flow_print(
            "[缺口回执] 无法生成规范 GapSpec：%s" % type(exc).__name__,
            tone="error",
        )
        return None, unsupported
    path = _interactive_private_json(session_root / "gap-request.json", payload)
    flow_print("[缺口回执] 已绑定源哈希/调用点/目标环境：%s" % path, tone="artifact")
    return path, unsupported


def _interactive_cannbot_route(snapshot, gap_request, args, session_root):
    """Run the framework's official-first gap route after confirmation.

    Ordinary missing APIs use ``resolve-gaps → complete-gaps``.  The tiny
    explicit ``add_one`` OperatorSpec used in the example is routed through the
    existing framework-owned demo provider, because a full 17-file Ascend C
    project would obscure the same official Skill/readiness boundary.  Both
    paths remain draft-only and are reported separately from migration.
    """

    if not getattr(args, "cannbot_gap_draft", False):
        flow_print(
            "[CANNBot] 未启用；如确认后需要补算子，请加 --cannbot-gap-draft。",
            tone="warning",
        )
        return {
            "status": "not_requested",
            "actual_remote_call": False,
            "claim_boundary": {"interface_available": False, "activated": False},
        }
    if not getattr(args, "accept_cannbot_license", False):
        flow_print(
            "[CANNBot] 未收到本次许可证确认；停止，未读取 CANNBot Skill、未调用模型。",
            tone="warning",
        )
        return {
            "status": "license_required",
            "actual_remote_call": False,
            "claim_boundary": {"interface_available": False, "activated": False},
        }
    # The tiny OperatorSpec example fixture is a first-class framework
    # route too.  Its canonical GapSpec is discovered by ``resolve-gaps``
    # itself, rather than by the ordinary NumPy-call hand-off, so it must be
    # admitted before the generic ``gap_request is None`` guard.  Detect it
    # from literal decorator syntax, not a substring in a comment/string.
    is_add_one = False
    try:
        import ast

        from numpilot.operator_specs import collect_static_operator_specs

        static_specs = collect_static_operator_specs(
            ast.parse(snapshot.text, filename=snapshot.path)
        )
        is_add_one = any(spec.name == "add_one" for spec in static_specs.specs.values())
    except (SyntaxError, TypeError, ValueError):
        is_add_one = False
    if gap_request is None and not is_add_one:
        flow_print(
            "[CANNBot] 当前源没有可绑定的普通脚本 GapSpec，跳过算子生成。",
            tone="warning",
        )
        return {
            "status": "no_gap",
            "actual_remote_call": False,
            "claim_boundary": {"interface_available": False, "activated": False},
        }

    cann_version = str(args.cann_version or "").strip()
    cann_root = str(getattr(args, "cann_root", "") or os.getenv("ASCEND_HOME_PATH", ""))
    if cann_version.lower() in {"", "unknown", "auto"} and cann_root:
        match = re.search(r"/cann-([^/]+)$", cann_root.rstrip("/"))
        if match:
            cann_version = match.group(1)
    if cann_version.lower() in {"", "unknown", "auto"}:
        flow_print(
            "[CANNBot] CANN 版本未知；停止在官方解析门禁，未调用模型。",
            tone="error",
        )
        return {
            "status": "cann_version_required",
            "actual_remote_call": False,
            "claim_boundary": {"interface_available": False, "activated": False},
        }

    target_soc = str(args.target_soc or "unknown")
    if target_soc.lower() in {"", "unknown", "auto"}:
        target_soc = os.getenv("ASCEND_SOC_VERSION", "Ascend910B4")
    resolution = session_root / "official-resolution.json"
    flow_print("\n── CANNBot-Skill 算子补全流程（框架内 draft-only）──", tone="title")
    flow_print(
        "[CANNBot 1/3] 官方解析：检查已安装 CANN/ops-nn；不联网、不调用模型",
        tone="phase",
    )
    command = [
        sys.executable, "-m", "numpilot", "resolve-gaps", snapshot.path,
        "--out", str(resolution), "--target-soc", target_soc,
        "--cann-version", cann_version, "--allow-custom-operator-generation",
    ]
    if not is_add_one:
        command[5:5] = ["--gap-request", str(gap_request)]
    if cann_root:
        command.extend(["--cann-root", cann_root])
    ops_root = os.getenv("NUMPILOT_OPS_NN_ROOT", "")
    if ops_root:
        command.extend(["--ops-nn-root", ops_root, "--accept-cann-osl"])
    flow_print("[框架命令] %s" % shlex.join(command), tone="artifact")
    try:
        completed = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parents[1]),
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            check=False,
            timeout=min(float(getattr(args, "timeout", 3600.0)), 3600.0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        flow_print("[CANNBot 1/3] 失败：%s" % type(exc).__name__, tone="error")
        return {
            "status": "resolution_failed",
            "actual_remote_call": False,
            "claim_boundary": {"interface_available": False, "activated": False},
        }
    if completed.returncode != 0 or not resolution.is_file():
        flow_print("[CANNBot] 官方解析未通过；fail-closed，不进入模型阶段。", tone="error")
        return {
            "status": "resolution_failed",
            "actual_remote_call": False,
            "claim_boundary": {"interface_available": False, "activated": False},
        }
    resolution_payload = _interactive_read_json(resolution)
    if not isinstance(resolution_payload, Mapping) or not resolution_payload.get(
        "model_kernel_generation_allowed"
    ):
        flow_print(
            "[CANNBot] 官方候选尚未明确 clean no_match；不调用模型。",
            tone="warning",
        )
        return {
            "status": "official_match_or_ambiguous",
            "actual_remote_call": False,
            "claim_boundary": {"interface_available": False, "activated": False},
        }
    flow_print(
        "[CANNBot 2/3] 官方门禁通过：status=%s · resolution=%s"
        % (resolution_payload.get("status", "?"), resolution),
        tone="success",
    )

    # The explicit add_one fixture uses the compact, receipt-bound provider so
    # an interactive run remains practical.  It is still launched by this core
    # command and all outputs stay in the same private session tree.
    if is_add_one:
        skill_root = str(
            getattr(args, "cannbot_skill_root", "")
            or os.getenv("NUMPILOT_CANNBOT_SKILLS", "")
            or ""
        )
        if not skill_root or skill_root == "auto":
            try:
                skill_root = str(_load_cannbot_api().default_skill_checkout())
            except Exception:
                skill_root = ""
        if not skill_root:
            flow_print(
                "[CANNBot 2/3] 找不到固定 Skill checkout；不调用模型。",
                tone="error",
            )
            return {
                "status": "skill_required",
                "actual_remote_call": False,
                "claim_boundary": {"interface_available": False, "activated": False},
            }
        try:
            from numpilot.migration_skill import load_migration_skill

            requested_skill = getattr(args, "skill_path", None)
            requested_skill_path = (
                Path(requested_skill).expanduser().absolute()
                if requested_skill
                else None
            )
            migration_skill = load_migration_skill(
                requested_skill_path.parent
                if requested_skill_path is not None
                else None
            )
            if (
                requested_skill_path is not None
                and requested_skill_path != migration_skill.skill_path.absolute()
            ):
                raise ValueError(
                    "--skill-path must name SKILL.md in a complete Skill bundle"
                )
        except (OSError, RuntimeError, ValueError) as exc:
            flow_print(
                "[CANNBot 2/3] NumPilot Skill 发布包不可用：%s；不调用模型。"
                % type(exc).__name__,
                tone="error",
            )
            return {
                "status": "numpilot_skill_required",
                "actual_remote_call": False,
                "claim_boundary": {"interface_available": False, "activated": False},
            }
        output = session_root / "cannbot-output"
        command = [
            sys.executable,
            "-m", "numpilot.demo_llm_operator_generation",
            "--out", str(output),
            "--artifact-store", str(session_root / "operator-artifacts"),
            "--llm-config", str(getattr(args, "llm_config", None) or os.getenv("NUMPILOT_LLM_CONFIG_FILE", "")),
            "--skill-path", str(migration_skill.skill_path),
            "--skill-sha256", migration_skill.files_sha256["SKILL.md"],
            "--skill-bundle-sha256", migration_skill.bundle_sha256,
            "--gap-source", snapshot.path,
            "--official-resolution", str(resolution),
            "--cannbot-skill-root", skill_root,
            "--show-cannbot-skill", getattr(args, "show_cannbot_skill", "full"),
            "--accept-cannbot-license",
            "--max-tokens", str(getattr(args, "cannbot_max_tokens", 4096)),
            "--timeout", str(int(float(getattr(args, "cannbot_timeout", 300)))),
        ]
        # Do not display an empty --llm-config value; the child can use its
        # already-loaded environment in that case.
        if not command[command.index("--llm-config") + 1]:
            index = command.index("--llm-config")
            del command[index:index + 2]
        flow_print(
            "[CANNBot 3/3] 读取已校验官方上下文并调用真实配置模型；显示心跳/完整公开上下文",
            tone="phase",
        )
        flow_print("[框架命令] %s" % shlex.join(command), tone="artifact")
        try:
            completed = subprocess.run(
                command,
                cwd=str(Path(__file__).resolve().parents[1]),
                env=os.environ.copy(),
                stdin=subprocess.DEVNULL,
                check=False,
                timeout=min(float(getattr(args, "cannbot_timeout", 300)), 3600.0),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            flow_print("[CANNBot 3/3] 失败：%s" % type(exc).__name__, tone="error")
            return {
                "status": "generation_failed",
                "actual_remote_call": True,
                "claim_boundary": {"interface_available": False, "activated": False},
            }
        report = output / "operator-generation-report.json"
        report_payload = _interactive_read_json(report)
        status = report_payload.get("status") if isinstance(report_payload, Mapping) else "failed"
        flow_print(
            "[CANNBot] 生成阶段结束：status=%s · report=%s" % (status, report),
            tone="warning" if status == "generated_unverified" else "error",
        )
        return {
            "status": status,
            "actual_remote_call": bool(
                isinstance(report_payload, Mapping)
                and report_payload.get("actual_remote_call") is True
            ),
            "report": str(report) if report.is_file() else None,
            "operator_artifact_ref": (
                report_payload.get("operator_artifact_ref")
                if isinstance(report_payload, Mapping)
                else None
            ),
            "generated_file": (
                str(report.parent / str(report_payload.get("generated_file")))
                if isinstance(report_payload, Mapping)
                and isinstance(report_payload.get("generated_file"), str)
                else None
            ),
            "claim_boundary": {
                "generated": status == "generated_unverified",
                "compiled": False,
                "numerically_verified": False,
                "activated": False,
                "interface_available": False,
            },
        }

    # Generic ordinary-script gaps use the canonical complete-gaps pipeline.
    output = session_root / "operator-gaps"
    command = [
        sys.executable, "-m", "numpilot", "complete-gaps", snapshot.path,
        "--gap-request", str(gap_request), "--out", str(output),
        "--target-soc", target_soc, "--cann-version", cann_version,
        "--provider", "llm", "--max-attempts", "1",
        "--llm-timeout", str(
            max(10, min(int(float(getattr(args, "cannbot_timeout", 300))), 3600))
        ),
        "--official-resolution-report", str(resolution),
        "--allow-custom-operator-generation",
        "--show-cannbot-skill", getattr(args, "show_cannbot_skill", "full"),
    ]
    if getattr(args, "llm_config", None):
        # The Python interpreter options must precede ``-m``; NumPilot's
        # global option belongs after the module name (or after the
        # subcommand).  Append it here so ``python -m numpilot`` remains a
        # valid argv on every supported Python version.
        command.extend(["--llm-config", str(args.llm_config)])
    if getattr(args, "cannbot_skill_root", None):
        command.extend(["--skill-root", str(args.cannbot_skill_root)])
    flow_print(
        "[CANNBot 3/3] 进入 complete-gaps：分阶段读取 Skill、真实模型生成；阶段心跳会持续显示",
        tone="phase",
    )
    flow_print("[框架命令] %s" % shlex.join(command), tone="artifact")
    try:
        completed = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parents[1]),
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            check=False,
            timeout=min(float(getattr(args, "cannbot_timeout", 300)), 3600.0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        flow_print("[CANNBot 3/3] 失败：%s" % type(exc).__name__, tone="error")
        return {
            "status": "generation_failed",
            "actual_remote_call": True,
            "claim_boundary": {"interface_available": False, "activated": False},
        }
    report = output / "gap-report.json"
    flow_print(
        "[CANNBot] complete-gaps rc=%s · report=%s" % (completed.returncode, report),
        tone="success" if completed.returncode == 0 else "error",
    )
    report_payload = _interactive_read_json(report)
    actual_remote = False
    if isinstance(report_payload, Mapping):
        count = report_payload.get("current_run_actual_remote_call_count")
        if isinstance(count, int) and not isinstance(count, bool) and count > 0:
            actual_remote = True
        drafts = report_payload.get("drafts")
        if isinstance(drafts, list):
            actual_remote = actual_remote or any(
                isinstance(item, Mapping)
                and item.get("current_run_actual_remote_call") is True
                for item in drafts
            )
    report_status = (
        str(report_payload.get("status", "failed"))
        if isinstance(report_payload, Mapping)
        else "failed"
    )
    claim_boundary = (
        dict(report_payload.get("claim_boundary"))
        if isinstance(report_payload, Mapping)
        and isinstance(report_payload.get("claim_boundary"), Mapping)
        else {"interface_available": False, "activated": False}
    )
    # A successful process only proves that a receipt was published.  It does
    # not upgrade a generated draft into an integrated interface.
    claim_boundary["interface_available"] = claim_boundary.get("interface_available") is True
    return {
        "status": report_status if completed.returncode == 0 else "failed",
        "actual_remote_call": actual_remote,
        "report": str(report) if report.is_file() else None,
        "claim_boundary": claim_boundary,
    }


def _interactive_final_summary(result, *, session_root, interpretation, gap_result):
    """Render final facts from workflow/verification/benchmark receipts only."""

    def _number(value):
        """Convert receipt numbers without allowing a malformed receipt to abort."""

        try:
            if isinstance(value, bool):
                return None
            converted = float(value)
            return converted if converted == converted else None
        except (TypeError, ValueError, OverflowError):
            return None

    run_dir = Path(result.get("workflow_report", "")).parent if result.get("workflow_report") else None
    flow_print("\n── NumPilot 迁移结果（以框架回执为准）──", tone="title")
    workflow_status = result.get("status", "unknown")
    flow_print(
        "[状态] workflow=%s · stages=%s/%s" % (
            workflow_status,
            result.get("completed_stage_count", 0), result.get("stage_count", 0),
        ),
        tone=(
            "success" if workflow_status == "passed"
            else "error" if workflow_status in {"failed", "blocked", "timeout"}
            else "warning"
        ),
    )
    for stage in result.get("stages", ()):
        if not isinstance(stage, Mapping):
            flow_print("[?] invalid-stage · receipt entry is not an object", tone="warning")
            continue
        stage_status = stage.get("status")
        marker = {"passed": "✓", "failed": "✗", "blocked": "!", "timeout": "!"}.get(
            stage_status, "?"
        )
        elapsed = _number(stage.get("elapsed_seconds", 0.0))
        flow_print(
            "[%s] %s · %s · %s"
            % (marker, stage.get("name", "stage"), stage_status or "unknown",
               "%.2fs" % elapsed if elapsed is not None else "elapsed=?"),
            tone=(
                "success" if stage_status == "passed"
                else "error" if stage_status == "failed"
                else "warning"
            ),
        )
    migrated = run_dir / "migration" / "migrated.py" if run_dir else None
    if migrated is not None and migrated.is_file() and not migrated.is_symlink():
        flow_print("[迁移后代码] %s" % migrated, tone="artifact")
    else:
        flow_print("[迁移后代码] 尚未生成（迁移阶段未通过）", tone="warning")
    verification = _interactive_read_json(run_dir / "script-verification-report.json") if run_dir else None
    if isinstance(verification, Mapping):
        verification_status = verification.get("status", "unknown")
        flow_print(
            "[运行验证] status=%s · numerical_consistency=%s" % (
                verification_status,
                verification_status == "passed",
            ),
            tone="success" if verification_status == "passed" else "error",
        )
        comparisons = verification.get("comparisons")
        if isinstance(comparisons, Mapping):
            for name, value in list(comparisons.items())[:8]:
                if isinstance(value, Mapping):
                    flow_print(
                        "    %s: elements=%s max_abs_diff=%s max_rel_diff=%s"
                        % (name, value.get("n_elements", "?"), value.get("max_abs_diff", "?"), value.get("max_rel_diff", "?")),
                        tone="metric",
                    )
    else:
        flow_print("[运行验证] not_run（没有可接受的 migration report）", tone="warning")
    benchmark = _interactive_read_json(run_dir / "script-benchmark-report.json") if run_dir else None
    performance = benchmark.get("performance") if isinstance(benchmark, Mapping) else None
    scale_rendered = False

    if isinstance(performance, Mapping) and performance.get("performance_valid") is True:
        speed = performance.get("median_speedup")
        conservative = performance.get("conservative_cpu_p10_over_npu_p90")
        ci = performance.get("median_speedup_bootstrap_95ci")
        cpu_ms = _number(performance.get("cpu_median_ms"))
        npu_ms = _number(performance.get("npu_median_ms"))
        speed_num = _number(speed)
        conservative_num = _number(conservative)
        if None in {cpu_ms, npu_ms, speed_num, conservative_num}:
            flow_print(
                "[性能] performance_valid=true 但数值字段不完整；不输出加速比。",
                tone="warning",
            )
        else:
            flow_print(
                "[性能] resident CPU median=%.3f ms · NPU median=%.3f ms · speedup=%.2fx · conservative=%.2fx · CI=%s"
                % (cpu_ms, npu_ms, speed_num, conservative_num, ci),
                tone="metric",
            )
        workers = benchmark.get("workers") if isinstance(benchmark, Mapping) else {}
        npu_worker = workers.get("npu") if isinstance(workers, Mapping) else None
        npu_outputs = npu_worker.get("outputs", []) if isinstance(npu_worker, Mapping) else []
        output = npu_outputs[0] if isinstance(npu_outputs, list) and npu_outputs else {}
        elements = output.get("elements") if isinstance(output, Mapping) else None
        data_bytes = output.get("data_bytes") if isinstance(output, Mapping) else None
        shape = output.get("shape") if isinstance(output, Mapping) else None
        dtype = output.get("dtype") if isinstance(output, Mapping) else None
        data_bytes_num = _number(data_bytes)
        mib = data_bytes_num / (1024 * 1024) if data_bytes_num is not None else None
        flow_print(
            "[规模] shape=%s · dtype=%s · elements=%s · data=%s bytes%s"
            % (shape, dtype, elements, data_bytes, " (%.2f MiB)" % mib if mib is not None else ""),
            tone="metric",
        )
        scale_rendered = True
        if speed_num is None:
            flow_print("[口径] 性能数值不完整；不判断当前规模是否有加速优势。", tone="warning")
        elif speed_num >= 1.0:
            flow_print(
                "[口径] 当前规模/当前设备的 steady device-resident marker 区间体现优势；不外推冷启动或端到端。",
                tone="success",
            )
        else:
            flow_print("[口径] 当前规模未体现优势；不外推到其他规模或端到端。", tone="warning")
    else:
        flow_print(
            "[性能] performance_valid=false · 未形成可宣称的加速比（benchmark 未运行或门禁未通过）",
            tone="warning",
        )
    # Verification is intentionally independent from the performance gate.  A
    # A no-benchmark run should still tell the developer what data size was
    # actually checked, while making it explicit that shape/dtype/bytes and any
    # speedup conclusion require a valid benchmark receipt.
    if not scale_rendered:
        fallback_elements = None
        if isinstance(benchmark, Mapping):
            workers = benchmark.get("workers")
            npu_worker = workers.get("npu") if isinstance(workers, Mapping) else None
            npu_outputs = npu_worker.get("outputs", []) if isinstance(npu_worker, Mapping) else []
            if isinstance(npu_outputs, list) and npu_outputs and isinstance(npu_outputs[0], Mapping):
                fallback_elements = npu_outputs[0].get("elements")
        if fallback_elements is None and isinstance(verification, Mapping):
            comparisons = verification.get("comparisons")
            if isinstance(comparisons, Mapping):
                for value in comparisons.values():
                    if isinstance(value, Mapping) and value.get("n_elements") is not None:
                        fallback_elements = value.get("n_elements")
                        break
        if fallback_elements is not None:
            flow_print(
                "[规模] elements=%s · shape/dtype/bytes=unavailable（benchmark 未运行或门禁未通过）"
                % _interactive_display_text(fallback_elements),
                tone="warning",
            )
        else:
            flow_print("[规模] unavailable（benchmark 未运行或回执未提供规模字段）", tone="warning")
    if isinstance(gap_result, Mapping) and gap_result.get("status") not in {None, "not_requested", "no_gap"}:
        gap_status = gap_result.get("status")
        flow_print(
            "[算子补全] status=%s · actual_remote_call=%s · report=%s" % (
                gap_status, gap_result.get("actual_remote_call"), gap_result.get("report", "—")
            ),
            tone=(
                "success" if gap_status == "qualified_composite_interface"
                else "warning" if gap_status == "generated_unverified"
                else "error"
            ),
        )
    flow_print(
        "[解读边界] 模型解读 status=%s · unverified=true · receipt=%s" % (
            interpretation.status, interpretation.report_path or "—"
        ),
        tone="warning",
    )


def _interactive_new_session_root(args):
    """Create the single private output tree for one interactive session."""

    configured = getattr(args, "out", None)
    if configured:
        target = Path(configured).expanduser()
        if not target.is_absolute():
            raise ValueError("--out 必须是全新的绝对目录")
        target = target.absolute()
    else:
        target = (default_demo_dir() / ("interactive-" + uuid.uuid4().hex)).absolute()
    if os.path.lexists(target):
        raise ValueError("会话输出目录已存在；请使用全新的绝对目录")
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise ValueError("会话输出目录的父目录不可用")
    target.mkdir(mode=0o700)
    os.chmod(target, 0o700)
    return target


def _interactive_report_interpretation(interpretation):
    if interpretation is None:
        return {}
    return {
        "status": getattr(interpretation, "status", "unknown"),
        "report": getattr(interpretation, "report_path", None),
        "configured_model": getattr(interpretation, "model", None),
        "provider": getattr(interpretation, "provider", None),
        "actual_remote_call": getattr(interpretation, "actual_remote_call", False) is True,
        "unverified": True,
    }


def _interactive_publish_reports(
    *,
    session_root,
    snapshot,
    status,
    api_analysis=None,
    interpretation=None,
    gap_result=None,
    workflow=None,
):
    """Publish the final JSON/Markdown navigation reports and show their paths."""

    from numpilot.interactive_report import build_session_report, write_session_reports

    report = build_session_report(
        session_root=Path(session_root),
        source={
            "path": snapshot.path,
            "sha256": snapshot.sha256,
            "bytes": snapshot.bytes,
            "lines": snapshot.lines,
        },
        status=status,
        api_analysis=api_analysis or {},
        interpretation=_interactive_report_interpretation(interpretation),
        gap_result=gap_result or {},
        workflow=workflow,
    )
    json_path, markdown_path = write_session_reports(Path(session_root), report)
    flow_print("[完整报告] Markdown：%s" % markdown_path, tone="artifact")
    flow_print("[机器回执] JSON：%s" % json_path, tone="artifact")
    return json_path, markdown_path


def _interactive_confirm_add_one_source_execution(
    snapshot, normalized_snapshot, args, session_root
):
    """Obtain a confirmation dedicated to executing two hash-bound scripts."""

    outputs = tuple(getattr(args, "outputs", None) or ("result",))
    flow_print(
        "\n[源码执行确认] 下一步将分别在隔离子进程中执行原始用户源码与"
        " add_one 规范化源码，并只比较输出 %s。" % ",".join(outputs),
        tone="warning",
    )
    flow_print(
        "                 原始=%s · sha256=%s"
        % (_interactive_display_text(snapshot.path), snapshot.sha256),
        tone="artifact",
    )
    flow_print(
        "                 规范化=%s · sha256=%s"
        % (
            _interactive_display_text(normalized_snapshot.path),
            normalized_snapshot.sha256,
        ),
        tone="artifact",
    )
    flow_print(
        "                 子进程隔离不是 OS 安全沙箱；源码自身的文件、网络或"
        "其他副作用仍可能发生。",
        tone="warning",
    )
    if getattr(args, "execute_source", False) is True:
        confirmed = True
        method = "cli_execute_source"
        flow_print(
            "[源码执行确认] --execute-source 已明确授权现在执行上述两个哈希绑定脚本。",
            tone="warning",
        )
    elif sys.stdin.isatty():
        try:
            flow_print(
                "仅输入 execute 明确授权执行，其他输入取消 [execute/N]：",
                tone="prompt",
                end="",
                flush=True,
            )
            answer = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        confirmed = answer == "execute"
        method = "tty_exact_phrase"
    else:
        confirmed = False
        method = "non_tty_without_execute_source"
        flow_print(
            "[源码执行确认] 通用 --yes 不授权执行源码；非交互环境还必须显式提供"
            " --execute-source。不会执行用户源码。",
            stream=sys.stderr,
            tone="error",
        )
    _interactive_private_json(
        Path(session_root) / "source-execution-confirmation.json",
        {
            "status": "accepted" if confirmed else "declined",
            "method": method,
            "source_path": snapshot.path,
            "source_sha256": snapshot.sha256,
            "normalized_source_path": normalized_snapshot.path,
            "normalized_source_sha256": normalized_snapshot.sha256,
            "outputs": list(outputs),
            "subprocess_isolation": True,
            "os_sandbox": False,
        },
    )
    return confirmed


def _interactive_complete_registered_add_one(snapshot, calls, args, session_root):
    """Generate a CANNBot draft, then qualify a separate run-scoped interface."""

    from numpilot.add_one_gap import (
        AddOneLoweringVerificationError,
        lower_add_one_calls,
        materialize_add_one_operator_spec,
        verify_add_one_lowering,
    )
    from numpilot.interactive_migration import (
        InteractiveMigrationError,
        read_source_snapshot,
    )
    from numpilot.runtime_extensions import (
        RuntimeExtensionError,
        activate_qualified_add_one_extension,
        materialize_add_one_extension,
        qualify_add_one_extension,
    )

    flow_print(
        "\n[算子接入 1/5] 固化 add_one 语义、调用点与原始源码哈希",
        tone="operator",
    )
    try:
        operator_spec, relation = materialize_add_one_operator_spec(
            original_source_path=snapshot.path,
            original_source=snapshot.text,
            calls=calls,
            out_dir=session_root / "add-one-contract",
            target_soc=args.target_soc,
            cann_version=args.cann_version,
        )
        operator_snapshot = read_source_snapshot(operator_spec)
    except Exception as exc:
        flow_print("[算子接入] 契约固化失败：%s" % type(exc).__name__, tone="error")
        return {
            "status": "operator_contract_failed",
            "route": "registered_add_one_composite_npu",
            "actual_remote_call": False,
            "claim_boundary": {"interface_available": False, "activated": False},
        }, None

    flow_print(
        "[算子接入 2/5] 官方优先解析 → 读取完整 CANNBot Skill → 配置模型生成",
        tone="operator",
    )
    generated = _interactive_cannbot_route(
        operator_snapshot, None, args, session_root
    )
    base = {
        "route": "registered_add_one_composite_npu",
        "interface": "asnumpy.add_one",
        "operator_spec": str(operator_spec),
        "relation_report": str(relation),
        "report": generated.get("report"),
        "generated_file": generated.get("generated_file"),
        "actual_remote_call": generated.get("actual_remote_call") is True,
        "upstream_modified": False,
        "cannbot_draft_activated": False,
    }
    if generated.get("status") != "generated_unverified":
        flow_print(
            "[算子接入] CANNBot 未形成合格的未验证草稿；后续接入和迁移停止。",
            tone="error",
        )
        base.update(
            status=generated.get("status", "generation_failed"),
            claim_boundary={"interface_available": False, "activated": False},
        )
        return base, None

    report_path = generated.get("report")
    draft_path = generated.get("generated_file")
    flow_print(
        "[算子接入 3/5] 草稿已生成：%s\n"
        "                 边界：generated_unverified；未编译、未测试、未激活。"
        % (draft_path or "unavailable"),
        tone="warning",
    )
    try:
        extension = materialize_add_one_extension(
            report_path,
            draft_path,
            session_root / "runtime-extension",
        )
        flow_print(
            "[算子接入 4/5] 资格验证 run-scoped composite："
            "asnumpy.ones_like → asnumpy.add（真实 NPU 子进程）",
            tone="operator",
        )
        stop = threading.Event()
        heartbeat = _interactive_heartbeat(stop, "add_one 真实 NPU 资格验证", interval=5.0)
        try:
            qualification = qualify_add_one_extension(
                extension,
                session_root / "runtime-extension-qualification",
                python_executable=getattr(args, "python", None),
                timeout_seconds=min(float(getattr(args, "timeout", 3600.0)), 3600.0),
            )
        finally:
            stop.set()
            heartbeat.join(timeout=1.0)
    except (RuntimeExtensionError, OSError, ValueError, TypeError) as exc:
        flow_print(
            "[算子接入] 组合接口资格验证失败：%s" % type(exc).__name__,
            tone="error",
        )
        base.update(
            status="qualification_failed",
            extension_manifest=(extension.manifest_path if "extension" in locals() else None),
            claim_boundary={"generated": True, "interface_available": False, "activated": False},
        )
        return base, None
    base.update(
        extension_manifest=extension.manifest_path,
        qualification_report=qualification.report_path,
    )
    if not qualification.interface_available:
        flow_print("[算子接入] NPU 资格回执未开放接口；后续迁移停止。", tone="error")
        base.update(
            status=qualification.status,
            claim_boundary={"generated": True, "interface_available": False, "activated": False},
        )
        return base, None

    try:
        import asnumpy

        activate_qualified_add_one_extension(qualification.report_path, asnumpy)
    except (RuntimeExtensionError, ImportError) as exc:
        flow_print("[算子接入] run-scoped 激活失败：%s" % type(exc).__name__, tone="error")
        base.update(
            status="activation_failed",
            claim_boundary={"generated": True, "qualified": True, "interface_available": False, "activated": False},
        )
        return base, None
    flow_print(
        "[算子接入 5/5] 本次 NumPilot 运行上下文新增 asnumpy.add_one(x)："
        "float32 y=x+1.0；scope=run，provenance=composite_npu。",
        tone="success",
    )
    flow_print(
        "                 上游 AsNumpy 未修改；CANNBot Ascend C 草稿仍未编译、未激活。",
        tone="warning",
    )
    try:
        normalized, lowering = lower_add_one_calls(
            source_path=snapshot.path,
            source=snapshot.text,
            calls=calls,
            out_dir=session_root / "add-one-lowering",
        )
    except Exception as exc:
        flow_print("[算子接入] 源码 lowering 失败：%s" % type(exc).__name__, tone="error")
        base.update(
            status="lowering_failed",
            claim_boundary={"generated": True, "qualified": True, "activated": True, "interface_available": True},
        )
        return base, None
    try:
        normalized_snapshot = read_source_snapshot(normalized)
        execution_confirmed = _interactive_confirm_add_one_source_execution(
            snapshot,
            normalized_snapshot,
            args,
            session_root,
        )
    except (InteractiveMigrationError, OSError, ValueError) as exc:
        flow_print(
            "[源码执行确认] 回执失败：%s；不会执行用户源码。"
            % type(exc).__name__,
            tone="error",
        )
        base.update(
            status="source_execution_confirmation_failed",
            lowered_source=str(normalized),
            lowering_report=str(lowering),
            claim_boundary={
                "generated": True,
                "qualified": True,
                "activated": True,
                "interface_available": True,
                "integration_verified": False,
            },
        )
        return base, None
    if not execution_confirmed:
        flow_print(
            "[源码执行确认] 已取消；原始与规范化源码均未执行。",
            tone="warning",
        )
        base.update(
            status="source_execution_declined",
            lowered_source=str(normalized),
            lowering_report=str(lowering),
            claim_boundary={
                "generated": True,
                "qualified": True,
                "activated": True,
                "interface_available": True,
                "integration_verified": False,
            },
        )
        return base, None
    flow_print(
        "[源码等价门] 已获得紧邻执行确认；现在对两个哈希绑定脚本做独立输出对拍",
        tone="phase",
    )
    try:
        lowering_verification = verify_add_one_lowering(
            snapshot.path,
            normalized,
            lowering,
            args.outputs or ("result",),
            session_root / "add-one-lowering" / "add-one-lowering-verification.json",
            user_confirmed=execution_confirmed,
            python_executable=getattr(args, "python", None),
            timeout_seconds=min(float(getattr(args, "timeout", 3600.0)), 3600.0),
        )
    except AddOneLoweringVerificationError as exc:
        receipt = exc.receipt.path if exc.receipt is not None else None
        flow_print(
            "[源码等价门] 失败：%s · receipt=%s" % (exc.code, receipt or "unavailable"),
            tone="error",
        )
        base.update(
            status="lowering_verification_failed",
            lowered_source=str(normalized),
            lowering_report=str(lowering),
            lowering_verification_report=receipt,
            claim_boundary={
                "generated": True,
                "qualified": True,
                "activated": True,
                "interface_available": True,
                "integration_verified": False,
            },
        )
        return base, None
    flow_print(
        "[源码等价门] constituent lowering passed · outputs=%s · receipt=%s"
        % (",".join(args.outputs or ("result",)), lowering_verification.path),
        tone="success",
    )
    flow_print(
        "                 已验证 numpy.ones_like + numpy.add 的等价 lowering；"
        "迁移程序未直接调用 asnumpy.add_one，接口集成未验证。",
        tone="warning",
    )
    base.update(
        status="qualified_composite_interface",
        provenance="composite_npu",
        scope="run",
        lowered_source=str(normalized),
        lowering_report=str(lowering),
        lowering_verification_report=lowering_verification.path,
        claim_boundary={
            "generated": True,
            "compiled": False,
            "unit_tested": False,
            "numerically_verified": True,
            "integration_verified": False,
            "qualified": True,
            "activated": True,
            "rescanned": False,
            "interface_available": True,
        },
    )
    return base, normalized_snapshot


def _cmd_interactive_migrate(args):
    """Run the reusable terminal analysis→gap→migration→evidence workflow."""

    from numpilot.interactive_migration import (
        InteractiveMigrationError,
        failure_report_payload,
        interpret_source_with_configured_model,
        read_source_snapshot,
        write_interpretation_report,
    )
    from numpilot.workflow import WorkflowError, run_ordinary_workflow

    try:
        source_arg = _interactive_source_arg(args)
        snapshot = read_source_snapshot(source_arg)
        session_root = _interactive_new_session_root(args)
    except (ValueError, InteractiveMigrationError, OSError) as exc:
        flow_print("[交互入口] %s" % exc, stream=sys.stderr, tone="error")
        return 2

    flow_print("\n── NumPilot 交互迁移入口 ──", tone="title")
    flow_print("[会话目录] %s" % session_root, tone="artifact")
    flow_print(
        "[阶段 1/5] 已读取稳定源码；完整源文件只在本机静态分析/后续门禁中使用。",
        tone="phase",
    )
    qualified_payload = {}
    gaps = []
    analysis_summary = {}
    static_admission = {"ok": False, "route": "not_run", "issues": []}
    try:
        qualified_payload, scan, classification = _interactive_runtime_analysis(snapshot, args)
        from numpilot.add_one_gap import detect_add_one_calls

        registered_add_one = detect_add_one_calls(snapshot.text)
        static_admission = _interactive_static_source_admission(
            snapshot,
            qualified_payload,
            scan,
            registered_add_one,
        )
        gaps = _interactive_render_analysis(snapshot, qualified_payload, scan, classification)
        analysis_summary = {
            "decision": qualified_payload.get("decision"),
            "runtime_identity_status": qualified_payload.get("runtime_identity_status"),
            "qualified_apis": list(qualified_payload.get("qualified_apis", ())),
            "numpy_use_count": len(getattr(scan, "uses", ()) or ()),
            "gap_count": len(gaps),
            "registered_add_one_calls": [item.to_dict() for item in registered_add_one],
            "static_source_admission": static_admission,
        }
        _interactive_private_json(
            session_root / "preflight-analysis.json",
            {"source": {"path": snapshot.path, "sha256": snapshot.sha256, "bytes": snapshot.bytes, "lines": snapshot.lines},
             "qualified": qualified_payload,
             "scanner": {"ok": bool(getattr(scan, "ok", False)), "issues": list(getattr(scan, "issues", ()) or ())},
             "static_source_admission": static_admission,
             "gap_count": len(gaps),
             "registered_add_one_calls": [item.to_dict() for item in registered_add_one]},
        )
    except Exception as exc:
        flow_print(
            "[分析] 失败关闭：%s" % type(exc).__name__,
            stream=sys.stderr,
            tone="error",
        )
        try:
            _interactive_publish_reports(
                session_root=session_root,
                snapshot=snapshot,
                status="analysis_failed",
                api_analysis=analysis_summary,
            )
        except Exception:
            pass
        flow_print("[会话回执] %s" % session_root, stream=sys.stderr, tone="artifact")
        return 2

    if static_admission.get("ok") is not True:
        reasons = tuple(static_admission.get("issues") or ())
        flow_print(
            "[静态准入] 未通过；不会读取模型配置、生成算子或执行用户源码。%s"
            % (
                " 原因：" + _interactive_display_text("；".join(reasons[:3]))
                if reasons
                else ""
            ),
            stream=sys.stderr,
            tone="error",
        )
        try:
            _interactive_publish_reports(
                session_root=session_root,
                snapshot=snapshot,
                status="static_admission_failed",
                api_analysis=analysis_summary,
            )
        except Exception:
            pass
        flow_print("[会话回执] %s" % session_root, stream=sys.stderr, tone="artifact")
        return 2

    # Build a canonical ordinary-script request before the remote call.  This
    # is local-only and lets the model see bounded, deterministic facts.
    target_soc = str(getattr(args, "target_soc", None) or "").strip()
    if target_soc.lower() in {"", "unknown", "auto"}:
        target_soc = os.getenv("ASCEND_SOC_VERSION", "Ascend910B4")
    cann_version = str(getattr(args, "cann_version", None) or "").strip()
    cann_root = str(getattr(args, "cann_root", "") or os.getenv("ASCEND_HOME_PATH", ""))
    if cann_version.lower() in {"", "unknown", "auto"} and cann_root:
        match = re.search(r"/cann-([^/]+)$", cann_root.rstrip("/"))
        if match:
            cann_version = match.group(1)
    if cann_version.lower() in {"", "unknown", "auto"}:
        cann_version = os.getenv("CANN_VERSION", "unknown")
    args.target_soc, args.cann_version = target_soc, cann_version
    gap_request, _unsupported = _interactive_gap_request(snapshot, classification, args, session_root)

    # Reject an ambiguous non-interactive invocation before reading the private
    # model configuration or making any remote request.  The shell wrapper
    # performs the same gate, but keeping it in the framework entry point makes
    # direct ``python -m numpilot interactive-migrate`` calls equally safe.
    if getattr(args, "non_interactive", False) and not getattr(args, "yes", False):
        flow_print(
            "[确认] --non-interactive 必须与 --yes 一起使用；未读取模型配置、未创建 workflow。",
            stream=sys.stderr,
            tone="error",
        )
        try:
            _interactive_publish_reports(
                session_root=session_root,
                snapshot=snapshot,
                status="confirmation_required",
                api_analysis=analysis_summary,
            )
        except Exception:
            pass
        flow_print("[会话回执] %s" % session_root, stream=sys.stderr, tone="artifact")
        return 2
    if not getattr(args, "yes", False) and not sys.stdin.isatty():
        flow_print(
            "[确认] 非交互环境必须显式 --yes；未读取模型配置、未创建 workflow。",
            stream=sys.stderr,
            tone="error",
        )
        try:
            _interactive_publish_reports(
                session_root=session_root,
                snapshot=snapshot,
                status="confirmation_required",
                api_analysis=analysis_summary,
            )
        except Exception:
            pass
        flow_print("[会话回执] %s" % session_root, stream=sys.stderr, tone="artifact")
        return 2

    interpretation = None
    try:
        status = _load_llm_or_report()
        if status is None:
            try:
                _interactive_publish_reports(
                    session_root=session_root,
                    snapshot=snapshot,
                    status="llm_not_configured",
                    api_analysis=analysis_summary,
                )
            except Exception:
                pass
            flow_print("[会话回执] %s" % session_root, stream=sys.stderr, tone="artifact")
            return 4
        flow_print(
            "\n[阶段 2/5] 模型正在阅读源码并整理迁移前解读（仅终端/manual lane，未执行源码）",
            tone="phase",
        )
        stop = threading.Event()
        heartbeat = _interactive_heartbeat(stop, "配置模型解读源码", interval=5.0)
        try:
            interpretation = interpret_source_with_configured_model(
                snapshot,
                config_path=getattr(args, "llm_config", None) or os.getenv("NUMPILOT_LLM_CONFIG_FILE"),
                static_analysis=qualified_payload,
                max_tokens=int(getattr(args, "interpret_max_tokens", 1800)),
            )
        finally:
            stop.set()
            heartbeat.join(timeout=1.0)
        interpretation_path = write_interpretation_report(
            session_root / "interpretation-report.json", interpretation
        )
        interpretation = interpretation.__class__(**{
            **interpretation.__dict__, "report_path": str(interpretation_path)
        })
    except InteractiveMigrationError as exc:
        flow_print(
            "[模型解读] 失败：%s；未进入迁移/执行阶段。" % exc,
            stream=sys.stderr,
            tone="error",
        )
        failure_path = None
        try:
            failure_path = _interactive_private_json(
                session_root / "interpretation-failure.json",
                failure_report_payload(snapshot, exc),
            )
        except Exception:
            pass
        flow_print(
            "[模型回执] actual_remote_call=%s · failure_receipt=%s"
            % (exc.actual_remote_call, failure_path or "unavailable"),
            stream=sys.stderr,
            tone="error",
        )
        try:
            _interactive_publish_reports(
                session_root=session_root,
                snapshot=snapshot,
                status="interpretation_failed",
                api_analysis=analysis_summary,
            )
        except Exception:
            pass
        flow_print("[会话回执] %s" % session_root, stream=sys.stderr, tone="artifact")
        return 4
    flow_print(
        "\n[模型解读 · 未验证建议] model=%s provider=%s elapsed=%sms" % (
            interpretation.model, interpretation.provider, interpretation.elapsed_ms
        ),
        tone="warning",
    )
    print(interpretation.display_text)
    if interpretation.source_truncated_for_prompt:
        flow_print("[模型输入边界] 完整源码 hash 已绑定；模型提示仅发送了首尾片段。", tone="warning")
    flow_print(
        "[模型回执] actual_remote_call=%s · receipt metadata 已写入私有目录（正文不落盘）"
        % interpretation.actual_remote_call,
        tone="model",
    )

    has_gap = bool(gaps)
    if getattr(args, "yes", False):
        answer = "y"
        flow_print("\n[确认] --yes/--non-interactive：开始迁移。", tone="warning")
    elif getattr(args, "non_interactive", False):
        flow_print(
            "[确认] --non-interactive 必须与 --yes 一起使用；未创建 workflow 运行目录。",
            stream=sys.stderr,
            tone="error",
        )
        return 2
    elif not sys.stdin.isatty():
        flow_print(
            "[确认] 非交互环境必须显式 --yes；未创建 workflow 运行目录。",
            stream=sys.stderr,
            tone="error",
        )
        return 2
    else:
        try:
            if has_gap and getattr(args, "cannbot_gap_draft", False):
                prompt = (
                    "\n是否开始迁移？输入 y 同时确认本次 CANNBot/CANN OSL 使用，"
                    "其他键取消 [y/N]："
                )
            else:
                prompt = "\n是否开始迁移？输入 y 开始，其他键取消 [y/N]："
            flow_print(prompt, tone="prompt", end="", flush=True)
            answer = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
    if answer not in {"y", "yes"}:
        _interactive_private_json(
            session_root / "confirmation.json",
            {"status": "declined", "source_path": snapshot.path, "source_sha256": snapshot.sha256,
             "interpretation_report": interpretation.report_path, "workflow_created": False},
        )
        flow_print("[确认] 已取消；未创建 workflow 运行目录、未执行源码、未开始迁移。", tone="warning")
        try:
            _interactive_publish_reports(
                session_root=session_root,
                snapshot=snapshot,
                status="declined",
                api_analysis=analysis_summary,
                interpretation=interpretation,
            )
        except Exception:
            pass
        flow_print("[会话回执] %s" % session_root, tone="artifact")
        return 0

    # In the human terminal the same explicit `y` authorizes the disclosed
    # per-invocation license.  Automated `--yes` remains separate and must also
    # supply --accept-cannbot-license so scripts cannot opt in accidentally.
    if (
        has_gap
        and getattr(args, "cannbot_gap_draft", False)
        and not getattr(args, "yes", False)
        and sys.stdin.isatty()
    ):
        args.accept_cannbot_license = True
    if (
        has_gap
        and getattr(args, "cannbot_gap_draft", False)
        and not getattr(args, "accept_cannbot_license", False)
    ):
        flow_print(
            "[确认] 自动化缺口补全还需要 --accept-cannbot-license；未调用 CANNBot。",
            stream=sys.stderr,
            tone="error",
        )
        gap_result = {
            "status": "license_required",
            "actual_remote_call": False,
            "claim_boundary": {"interface_available": False, "activated": False},
        }
        _interactive_publish_reports(
            session_root=session_root,
            snapshot=snapshot,
            status="license_required",
            api_analysis=analysis_summary,
            interpretation=interpretation,
            gap_result=gap_result,
        )
        return 2
    _interactive_private_json(
        session_root / "confirmation.json",
        {"status": "accepted", "source_path": snapshot.path, "source_sha256": snapshot.sha256,
         "interpretation_report": interpretation.report_path, "workflow_authorized": True,
         "workflow_created": False},
    )

    flow_print("\n[阶段 3/5] 已确认 y；进入框架迁移主链。", tone="phase")
    registered_gap = [
        item
        for item in gaps
        if isinstance(item, Mapping) and item.get("kind") == "registered_signature_gap"
    ]
    ordinary_gaps = [item for item in gaps if item not in registered_gap]
    workflow_snapshot = snapshot
    gap_result = {
        "status": "no_gap",
        "actual_remote_call": False,
        "claim_boundary": {"interface_available": False, "activated": False},
    }

    if ordinary_gaps:
        gap_result = _interactive_cannbot_route(snapshot, gap_request, args, session_root)
        boundary = gap_result.get("claim_boundary")
        interface_available = isinstance(boundary, Mapping) and boundary.get("interface_available") is True
        if not interface_available:
            flow_print(
                "[迁移门禁] 缺口仅形成草稿或尚未完成接入/重扫；"
                "不会继续迁移并假装接口已存在。",
                tone="error",
            )
            _interactive_publish_reports(
                session_root=session_root,
                snapshot=snapshot,
                status="gap_not_integrated",
                api_analysis=analysis_summary,
                interpretation=interpretation,
                gap_result=gap_result,
            )
            flow_print("[会话回执] %s" % session_root, tone="artifact")
            return 1
    elif registered_gap:
        if not getattr(args, "cannbot_gap_draft", False):
            gap_result = {
                "status": "cannbot_not_enabled",
                "route": "registered_add_one_composite_npu",
                "actual_remote_call": False,
                "claim_boundary": {"interface_available": False, "activated": False},
            }
            flow_print(
                "[迁移门禁] add_one 缺口需要 CANNBot 路由；当前未启用，停止。",
                tone="error",
            )
            _interactive_publish_reports(
                session_root=session_root,
                snapshot=snapshot,
                status="gap_not_integrated",
                api_analysis=analysis_summary,
                interpretation=interpretation,
                gap_result=gap_result,
            )
            return 1
        gap_result, workflow_snapshot = _interactive_complete_registered_add_one(
            snapshot,
            registered_add_one,
            args,
            session_root,
        )
        boundary = gap_result.get("claim_boundary")
        if workflow_snapshot is None or not (
            isinstance(boundary, Mapping) and boundary.get("interface_available") is True
        ):
            flow_print(
                "[迁移门禁] add_one 接口没有通过真机资格门禁；整体迁移停止。",
                tone="error",
            )
            _interactive_publish_reports(
                session_root=session_root,
                snapshot=snapshot,
                status="gap_not_integrated",
                api_analysis=analysis_summary,
                interpretation=interpretation,
                gap_result=gap_result,
            )
            flow_print("[会话回执] %s" % session_root, tone="artifact")
            return 1

    try:
        workflow_out = session_root / "workflow"
        _interactive_private_json(
            session_root / "workflow-start.json",
            {
                "status": "authorized",
                "original_source_path": snapshot.path,
                "original_source_sha256": snapshot.sha256,
                "workflow_source_path": workflow_snapshot.path,
                "workflow_source_sha256": workflow_snapshot.sha256,
                "workflow_output": str(workflow_out),
            },
        )
        flow_print(
            "\n[阶段 4/5] 分析 → configured-model 迁移 → 独立运行验证"
            + (" → strict-NPU benchmark" if args.benchmark else ""),
            tone="phase",
        )
        stop = threading.Event()
        heartbeat = _interactive_heartbeat(stop, "NumPilot 迁移/验证流水线", interval=8.0)
        try:
            result = run_ordinary_workflow(
                workflow_snapshot.path,
                workflow_out,
                backend=args.backend,
                fallback_policy=args.fallback_policy,
                target_soc=args.target_soc,
                cann_version=args.cann_version,
                outputs=args.outputs or ("result",),
                benchmark=bool(args.benchmark),
                warmup=args.warmup,
                samples=args.samples,
                core=args.core,
                skip_env_check=bool(args.skip_env_check),
                timeout_seconds=args.timeout,
                python_executable=args.python,
            )
        finally:
            stop.set()
            heartbeat.join(timeout=1.0)
    except WorkflowError as exc:
        flow_print(
            "[迁移流水线] %s: %s" % (exc.code, exc),
            stream=sys.stderr,
            tone="error",
        )
        try:
            _interactive_publish_reports(
                session_root=session_root,
                snapshot=snapshot,
                status="workflow_failed",
                api_analysis=analysis_summary,
                interpretation=interpretation,
                gap_result=gap_result,
            )
        except Exception:
            pass
        flow_print("[会话回执] %s" % session_root, stream=sys.stderr, tone="artifact")
        return 2
    boundary = gap_result.get("claim_boundary")
    if isinstance(boundary, dict) and result.get("stages"):
        boundary["rescanned"] = any(
            isinstance(stage, Mapping)
            and stage.get("name") == "01-analyze"
            and stage.get("status") == "passed"
            for stage in result.get("stages", ())
        )
    flow_print("\n[阶段 5/5] 读取真实回执并汇总（不根据模型口述编造结果）", tone="phase")
    _interactive_final_summary(result, session_root=session_root, interpretation=interpretation, gap_result=gap_result)
    try:
        _interactive_publish_reports(
            session_root=session_root,
            snapshot=snapshot,
            status=str(result.get("status", "unknown")),
            api_analysis=analysis_summary,
            interpretation=interpretation,
            gap_result=gap_result,
            workflow=result,
        )
    except Exception as exc:
        flow_print(
            "[完整报告] 发布失败：%s" % type(exc).__name__,
            stream=sys.stderr,
            tone="error",
        )
        flow_print("[会话回执] %s" % session_root, stream=sys.stderr, tone="artifact")
        return 2
    flow_print("[会话回执] %s" % session_root, tone="artifact")
    if result.get("status") == "passed":
        return 0
    return 1 if result.get("status") == "blocked" else 2


def _cmd_doctor(args):
    backend_name = _resolve_backend_name(args)
    machine = bool(getattr(args, "json", False))
    with terminal_status(
        "检查 Python、CANN、NPU 与 AsNumpy",
        enabled=not machine,
    ) as progress:
        rep = Env.check(backend=backend_name)
        if rep.errors:
            progress.error()
        elif rep.warnings:
            progress.warning()
    if machine:
        print(json.dumps(
            rep.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ))
        return 1 if rep.errors else 0
    if not render_doctor(rep):
        print(rep.render())
    return 1 if rep.errors else 0


def _cmd_paths(args):
    paths = path_summary()
    if args.get:
        print(paths[args.get])
    elif args.json:
        print(json.dumps(paths, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        if not render_paths(paths):
            print("NumPilot 路径（本命令不会创建目录）:")
            for name, path in paths.items():
                print(f"  {name:14} {path}")
    return 0


def _bootstrap_arguments(args):
    """Map the public CLI to the deliberately small bootstrap API."""

    return {
        "asnumpy_component": (
            Path(args.asnumpy_component).expanduser()
            if args.asnumpy_component is not None
            else None
        ),
        "asnumpy_wheel": (
            Path(args.asnumpy_wheel).expanduser()
            if args.asnumpy_wheel is not None
            else None
        ),
        "expected_wheel_sha256": args.asnumpy_wheel_sha256,
        "build_asnumpy": bool(args.build_asnumpy),
        "use_cannbot": bool(args.with_cannbot),
        "accept_cannbot_license": bool(args.accept_cannbot_license),
        "asnumpy_cache": (
            Path(args.asnumpy_cache).expanduser()
            if args.asnumpy_cache is not None
            else None
        ),
        "cannbot_cache": (
            Path(args.cannbot_cache).expanduser()
            if args.cannbot_cache is not None
            else None
        ),
    }


def _render_bootstrap_plan(plan):
    request = plan["request"]
    print("[bootstrap] mode=read-only-plan · 未联网 · 未修改环境")
    print(
        "[环境] Python=%s (ok=%s) · venv=%s · CANN=%s · OPP=%s · NPU=%s"
        % (
            plan["python"]["version"],
            plan["python"]["ok"],
            plan["venv"]["active"],
            plan["cann"]["toolkit_directory"],
            plan["cann"]["opp_directory"],
            plan["npu_smi"]["available"],
        )
    )
    print(
        "[选择] AsNumpy=%s · CANNBot=%s · license-accepted=%s"
        % (
            request["asnumpy_mode"] or "未选择",
            request["use_cannbot"],
            request["cannbot_license_accepted"],
        )
    )
    for action in plan["planned_actions"]:
        print(
            "[计划] %s · network=%s"
            % (action["action"], str(bool(action["network"])).lower())
        )
    for issue in plan["issues"]:
        print("[待处理] %s" % issue)
    print("[提示] 使用 --json 查看固定仓库身份、工具版本和完整计划。")


def _cmd_bootstrap(args):
    """Inspect locally by default; mutate only after an explicit ``--apply``."""

    from numpilot.bootstrap import (
        BootstrapError,
        apply_bootstrap,
        build_bootstrap_plan,
        verify_bootstrap_installation,
    )

    values = _bootstrap_arguments(args)
    if args.verify:
        if any(
            (
                values["asnumpy_wheel"] is not None,
                values["expected_wheel_sha256"] is not None,
                values["build_asnumpy"],
                values["asnumpy_component"] is not None,
                values["asnumpy_cache"] is not None,
            )
        ):
            print(
                "[bootstrap] --verify 不接受 AsNumpy 安装/构建参数；"
                "它只复验当前解释器。",
                file=sys.stderr,
            )
            return 2
        try:
            result = verify_bootstrap_installation(
                use_cannbot=values["use_cannbot"],
                accept_cannbot_license=values["accept_cannbot_license"],
                cannbot_cache=values["cannbot_cache"],
            )
        except BootstrapError as exc:
            print("[bootstrap] verify 失败: %s" % exc, file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            smoke = result["asnumpy"]
            print(
                "[AsNumpy] 安装验证通过 · ap.ones(shape=%s, dtype=%s)"
                % (tuple(smoke["shape"]), smoke["dtype"])
            )
            cannbot = result["cannbot"]
            if cannbot["status"] == "passed":
                print("[CANNBot] %s" % cannbot["question"])
                for item in cannbot["skills"]:
                    print("  - %s：%s" % (item["phase"], item["name"]))
            else:
                print("[CANNBot] 未请求；增加 --with-cannbot 可复验本地固定 bundle。")
            print("[bootstrap] verification network_requested_by_numpilot=false")
        return 0
    if not args.apply:
        plan = build_bootstrap_plan(**values)
        if args.json:
            print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            _render_bootstrap_plan(plan)
        return 0

    receipt = Path(args.receipt).expanduser()
    try:
        result = apply_bootstrap(receipt_path=receipt, **values)
    except BootstrapError as exc:
        print("[bootstrap] apply 失败: %s" % exc, file=sys.stderr)
        print("[bootstrap] 失败 receipt: %s" % receipt, file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("[bootstrap] status=ready · receipt=%s" % receipt)
        print("[bootstrap] AsNumpy wheel SHA-256=%s" % result["wheel"]["sha256"])
        if args.with_cannbot:
            print("[bootstrap] CANNBot 固定版本与许可证回执已验证。")
    return 0


def _cmd_check_suite(args):
    """Statically admit a proposed migration suite without executing it."""

    from numpilot.suite_check import (
        SuiteCheckError,
        check_suite,
        public_suite_report,
        write_suite_report,
    )

    try:
        report = check_suite(
            args.paths,
            target_soc=args.target_soc,
            cann_version=args.cann_version,
            expected_direct=args.expect_direct,
            expected_single_gap=args.expect_single_gap,
        )
        if args.out:
            write_suite_report(args.out, report)
    except SuiteCheckError as exc:
        print("[测试集准入] 失败: %s" % exc, file=sys.stderr)
        return 2

    # Keep stdout a strict JSON stream when requested.  This is useful for
    # CodeArts/CI jobs that consume the admission receipt directly.  The
    # optional ``--out`` publication still happens above, so callers can both
    # capture a pipe-safe payload and retain the private, mode-0600 receipt.
    if getattr(args, "json", False):
        public_report = public_suite_report(report)
        print(json.dumps(
            public_report,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ))
        return 0 if report["status"] == "pass" else 1

    counts = report["counts"]
    print(
        "[测试集准入] status=%s total=%d direct=%d single_gap=%d blocked=%d"
        % (
            report["status"],
            len(report["cases"]),
            counts["direct_migration_candidate"],
            counts["single_gap_candidate"],
            counts["blocked"],
        )
    )
    for case in report["cases"]:
        suffix = ""
        if case["category"] == "single_gap_candidate":
            suffix = " gap=" + case["generatable_gaps"][0]["qualified_api"]
        print(
            "[%s] %s%s"
            % (case["category"], case["path"], suffix)
        )
        for error in case.get("errors", ()):
            print("  - %s" % error, file=sys.stderr)
    for error in report["expectation_errors"]:
        print("[测试集期望] %s" % error, file=sys.stderr)
    if args.out:
        print("[测试集准入] report=%s" % Path(args.out).absolute())
    print(
        "[测试集准入] execution_performed=no runtime_capability_checked=no；"
        "通过仅表示静态候选合格，真机仍须精确 wheel provenance 与数值门禁。"
    )
    return 0 if report["status"] == "pass" else 1


def _load_llm_or_report():
    """Load framework-managed credentials; never prompt from non-wizard commands."""
    try:
        status = load_llm_config()
    except LlmConfigError as exc:
        print(f"[LLM] 本机配置无效: {exc}", file=sys.stderr)
        print(
            "[LLM] 请在仓库根目录执行 install -m 600 llm.example.json llm.json "
            "后填写，或运行交互配置/configure-llm --init。",
            file=sys.stderr,
        )
        return None
    if not status.configured:
        print(f"[LLM] {status.summary()}", file=sys.stderr)
        print(
            "[LLM] 请在仓库根目录执行 install -m 600 llm.example.json llm.json "
            "后填写，或运行交互配置/configure-llm --init。",
            file=sys.stderr,
        )
        return None
    return status


def _llm_proxy_status_or_report():
    """Return a redacted local proxy snapshot or print an actionable error."""

    try:
        proxy_status = llm_proxy_route_status()
    except LlmConfigError as exc:
        print("[LLM proxy] 配置无效: %s" % exc, file=sys.stderr)
        return None
    if (
        proxy_status["policy"] == "required"
        and proxy_status["route"] != "enforced"
    ):
        if proxy_status["bypassed"]:
            reason = "模型 endpoint 命中 NO_PROXY/no_proxy，已禁止绕过代理"
        elif not proxy_status["matched"]:
            reason = "endpoint 协议没有匹配的 HTTP_PROXY/HTTPS_PROXY"
        else:
            reason = "代理路由无法被强制证明"
        print("[LLM proxy] 门禁失败: %s" % reason, file=sys.stderr)
        return None
    return proxy_status


def _render_llm_proxy_status(proxy_status):
    """Render only non-sensitive route facts."""

    return (
        "[LLM proxy] policy=%s scheme=%s matched=%s bypass=%s route=%s"
        % (
            proxy_status["policy"],
            proxy_status["scheme"],
            "yes" if proxy_status["matched"] else "no",
            "yes" if proxy_status["bypassed"] else "no",
            proxy_status["route"],
        )
    )


def _llm_proxy_for_remote_call(stage):
    """Validate and display the redacted route before a model workflow."""

    proxy_status = _llm_proxy_status_or_report()
    if proxy_status is None:
        print(
            "[%s] fail · 代理路由门禁未通过，模型调用未开始" % stage,
            file=sys.stderr,
        )
        return None
    print(_render_llm_proxy_status(proxy_status))
    return proxy_status


_LLM_READY_MARKER = "NUMPILOT_LLM_READY"
_LLM_PROBE_SYSTEM = (
    "You are a connectivity probe. Return one short acknowledgement."
)
_LLM_PROBE_QUESTION = "Reply with exactly the ASCII token NUMPILOT_LLM_READY and nothing else. Do not use XML tags, markup, or code fences."


def _run_llm_readiness_gate(status, *, require_marker):
    """Make and verify one small remote call before operator generation."""

    try:
        identity = require_model_status(status)
    except ModelContractError as exc:
        raise LlmResponseError("readiness_gate_validation_failed", {}) from exc
    content, receipt = LlmEngine._chat_with_receipt(
        _LLM_PROBE_SYSTEM,
        _LLM_PROBE_QUESTION,
    )
    try:
        validated_receipt = validate_configured_model_receipt(
            receipt,
            content,
            expected_model=identity.model,
            expected_provider=identity.provider,
            expected_protocol=identity.protocol,
            response_model_aliases=identity.response_model_aliases,
        )
    except ModelContractError as exc:
        raise LlmResponseError(
            "readiness_gate_validation_failed", receipt or {}
        ) from exc
    marker_present = (
        isinstance(content, str) and _LLM_READY_MARKER in content
    )
    if not isinstance(content, str) or not content.strip() or (
        require_marker and not marker_present
    ):
        raise LlmResponseError("readiness_gate_validation_failed", receipt or {})
    return {
        "schema_version": 2,
        "gate": "llm_readiness",
        "status": "pass",
        "actual_remote_call": True,
        "question_sha256": hashlib.sha256(
            _LLM_PROBE_QUESTION.encode("utf-8")
        ).hexdigest(),
        "answer_marker": _LLM_READY_MARKER,
        "answer_marker_present": marker_present,
        "receipt": validated_receipt,
    }


def _run_cannbot_skill_gate(bundle, required_files=None):
    """Prove that the external CANNBot workflow files were read."""

    cannbot = _load_cannbot_api()
    required = tuple(
        required_files or cannbot.CANNBOT_GENERATION_SKILL_FILES
    )
    if not required or len(set(required)) != len(required):
        raise CannbotSkillError(
            "CANNBot gate has an invalid required file set"
        )
    manifest = bundle.manifest()
    manifest_files = manifest.get("files") or {}
    verified_hashes = {}
    for relative in required:
        content = bundle.files.get(relative)
        if not isinstance(content, str) or not content.strip():
            raise CannbotSkillError(
                "required CANNBot workflow file is empty or unreadable: %s"
                % relative
            )
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if manifest_files.get(relative) != content_hash:
            raise CannbotSkillError(
                "required CANNBot workflow file hash mismatch: %s" % relative
            )
        verified_hashes[relative] = content_hash
    rendered = bundle.prompt_context(required)
    if (
        bundle.bundle_sha256 not in rendered
        or any(bundle.files[path] not in rendered for path in required)
    ):
        raise CannbotSkillError(
            "required CANNBot workflow was not embedded in prompt context"
        )
    return {
        "schema_version": 1,
        "gate": "cannbot_skill_readiness",
        "status": "pass",
        "repository": manifest.get("repository"),
        "commit": bundle.commit,
        "skill_name": bundle.skill_name,
        "primary_file": required[0],
        "primary_file_sha256": verified_hashes[required[0]],
        "required_skill_files": list(required),
        "required_skill_files_sha256": verified_hashes,
        "bundle_sha256": bundle.bundle_sha256,
        "verified_file_count": len(bundle.files),
        "prompt_context_sha256": hashlib.sha256(
            rendered.encode("utf-8")
        ).hexdigest(),
        "content_embedded_for_remote_model": True,
    }


def _run_asnumpy_skill_gate(bundle, required_files=None):
    """Prove that NumPilot's repository-owned integration skill was read."""

    required = tuple(required_files or ASNUMPY_INTEGRATION_SKILL_FILES)
    if not required or len(set(required)) != len(required):
        raise AsnumpyIntegrationSkillError(
            "AsNumpy integration gate has an invalid required file set"
        )
    manifest = bundle.manifest()
    manifest_files = manifest.get("files") or {}
    verified_hashes = {}
    for relative in required:
        content = bundle.files.get(relative)
        if not isinstance(content, str) or not content.strip():
            raise AsnumpyIntegrationSkillError(
                "required AsNumpy integration skill file is unreadable: %s"
                % relative
            )
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if manifest_files.get(relative) != digest:
            raise AsnumpyIntegrationSkillError(
                "required AsNumpy integration skill hash mismatch: %s" % relative
            )
        verified_hashes[relative] = digest
    rendered = bundle.prompt_context(required)
    if (
        bundle.bundle_sha256 not in rendered
        or any(bundle.files[path] not in rendered for path in required)
    ):
        raise AsnumpyIntegrationSkillError(
            "AsNumpy integration skill was not embedded in prompt context"
        )
    return {
        "schema_version": 1,
        "gate": "asnumpy_integration_skill_readiness",
        "status": "pass",
        "source": manifest.get("source"),
        "skill_name": bundle.skill_name,
        "required_skill_files": list(required),
        "required_skill_files_sha256": verified_hashes,
        "bundle_sha256": bundle.bundle_sha256,
        "verified_file_count": len(bundle.files),
        "prompt_context_sha256": hashlib.sha256(
            rendered.encode("utf-8")
        ).hexdigest(),
        "content_embedded_for_remote_model": True,
    }


def _write_generation_preflight(
    out_dir,
    llm_gate,
    skill_gate,
    timeout_seconds,
    operator_goal=None,
    adapter_context_plan=None,
):
    """Atomically save both successful gates without secrets or response text."""

    root = Path(out_dir).absolute()
    if os.path.lexists(root) and (root.is_symlink() or not root.is_dir()):
        raise OSError("generation output is not a real directory")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    skill_gate_name = skill_gate.get("gate")
    if skill_gate_name not in (
        "cannbot_skill_readiness",
        "asnumpy_integration_skill_readiness",
    ):
        raise OSError("generation skill gate identity is invalid")
    preflight = {
            "schema_version": 1,
            "producer": "numpilot-generation-preflight-v1",
            "status": "pass",
            "required_order": ["llm_readiness", skill_gate_name],
            "client_timeout_seconds": timeout_seconds,
            "gates": {
                "llm_readiness": llm_gate,
                skill_gate_name: skill_gate,
            },
        }
    if operator_goal is not None:
        preflight["operator_goal"] = operator_goal
        preflight["performance_optimization"] = (
            "deferred_until_accuracy_and_user_confirmation"
            if operator_goal == "performance"
            else "skipped_accuracy_first"
        )
    if adapter_context_plan is not None:
        # Keep the persisted preflight on the same schema validator as the
        # provider.  A second, narrower schema check here previously rejected
        # the reviewed official schema-2 dependency plan after readiness had
        # already made a network call.
        from numpilot.gaps.provider import (
            DraftGenerationError,
            _validate_adapter_context_plan,
        )

        try:
            normalized_context_plan = _validate_adapter_context_plan(
                adapter_context_plan
            )
        except DraftGenerationError as exc:
            raise OSError("AsNumpy adapter context plan is invalid") from exc
        preflight["asnumpy_adapter_context_plan"] = normalized_context_plan
    payload = (
        json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    fd, temporary = tempfile.mkstemp(
        prefix=".generation-preflight-", suffix=".tmp", dir=str(root)
    )
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, root / "generation-preflight.json")
        temporary = ""
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def _create_output_stage(out_dir, label):
    """Create a private sibling stage without claiming the final output path."""

    final = Path(out_dir).absolute()
    if os.path.lexists(final):
        raise OSError("%s output already exists" % label)
    final.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "-", final.name)[:80] or "output"
    return Path(tempfile.mkdtemp(
        prefix=".%s-%s-stage-" % (safe_name, label),
        dir=str(final.parent),
    ))


def _publish_output_stage(stage, out_dir, label):
    """Publish validated staged entries without replacing any existing path."""

    stage = Path(stage)
    final = Path(out_dir).absolute()
    # mkdir is the atomic no-replace reservation.  We intentionally do not use
    # os.replace here: on POSIX it can replace an existing empty user directory.
    os.mkdir(final, mode=0o700)
    try:
        for source in sorted(stage.iterdir(), key=lambda item: item.name):
            os.rename(source, final / source.name)
        os.rmdir(stage)
    except Exception as exc:
        # The directory was exclusively created by this process, but another
        # process may have written into it since then.  Preserve it rather than
        # recursively deleting data we cannot prove we own.
        raise OSError(
            "%s publication was interrupted; retained partial output at %s"
            % (label, final)
        ) from exc


def _resolve_operator_goal(args):
    """Choose the implemented accuracy lane or the reserved performance lane."""

    goal = getattr(args, "operator_goal", None)
    if goal is None and sys.stdin.isatty() and sys.stdout.isatty():
        print("[算子目标] 请选择补全目标：")
        print("  1) 精度正确（推荐）：通过编译、真机运行和数值精度后即可使用")
        print("  2) 高性能（尚未开放）：当前会安全拒绝且不会联网或写输出")
        answer = input("输入 1 或 2 [默认 1]: ").strip()
        goal = "performance" if answer == "2" else "accuracy"
    elif goal is None:
        goal = "accuracy"
        print(
            "[算子目标] 非交互运行，默认 accuracy：先保证精度正确，"
            "跳过性能优化。"
        )
    if goal == "performance":
        print(
            "[算子目标] performance 尚未开放：当前版本不会启动模型、"
            "读取 Skill 或写入输出。",
            file=sys.stderr,
        )
    else:
        print(
            "[算子目标] accuracy：本轮以语义、编译、真机运行和数值精度正确为准；"
            "不执行性能优化。"
        )
    return goal


def _cmd_configure_llm(args):
    """Framework-owned setup; token is hidden or read from stdin, never argv."""
    if args.init:
        conflicting = (
            any(value is not None for value in (
                args.base_url, args.model, args.provider, args.protocol,
                args.response_model_alias,
            ))
            or args.token_stdin
        )
        if conflicting:
            print(
                "[LLM] --init 只生成空白模板，不能与 --base-url、--model、"
                "--provider、--protocol、--response-model-alias 或 "
                "--token-stdin 同时使用。",
                file=sys.stderr,
            )
            return 2
        try:
            path, created = init_llm_config_template()
        except LlmConfigError as exc:
            print(f"[LLM] 模板生成失败: {exc}", file=sys.stderr)
            return 2
        if created:
            print(f"LLM 配置模板已生成: {path} (权限 0600)")
            print("请打开该 JSON 文件，填写 base_url、auth_token 和 model。")
            print("源码 checkout 也可直接使用仓库根模板: "
                  "install -m 600 llm.example.json llm.json")
        else:
            print(f"LLM 配置目标已存在，未覆盖: {path}")
        print("填写后运行: python -m numpilot configure-llm --check")
        print("确认真实调用: python -m numpilot configure-llm --probe")
        return 0

    if args.check:
        status = _load_llm_or_report()
        if status is None:
            return 1
        proxy_status = _llm_proxy_status_or_report()
        if proxy_status is None:
            return 1
        print(status.summary())
        print(f"配置路径: {status.path}")
        print(_render_llm_proxy_status(proxy_status))
        return 0

    if args.probe:
        status = _load_llm_or_report()
        if status is None:
            print(
                "[LLM probe] 失败: 本地配置未通过，未发起远端请求。",
                file=sys.stderr,
            )
            return 1
        proxy_status = _llm_proxy_status_or_report()
        if proxy_status is None:
            print(
                "[LLM probe] 失败: 本地代理门禁未通过，未发起远端请求。",
                file=sys.stderr,
            )
            return 1
        try:
            with terminal_status(
                "连接已配置模型并校验真实响应",
            ):
                gate = _run_llm_readiness_gate(status, require_marker=True)
        except LlmResponseError as exc:
            if "readiness_gate_validation_failed" in str(exc):
                print(
                    "[LLM probe] 失败: 远端响应与本地回执校验不一致，"
                    "不能证明真实模型调用成功。",
                    file=sys.stderr,
                )
            else:
                print(
                    "[LLM probe] 失败: 未成功获得可验证的远端模型响应: %s"
                    % exc,
                    file=sys.stderr,
                )
            return 4
        except LlmConfigError as exc:
            print(
                "[LLM probe] 失败: 本地配置或代理门禁无效: %s" % exc,
                file=sys.stderr,
            )
            return 1
        except (LlmHttpError, LlmTransportError) as exc:
            print(
                "[LLM probe] 失败: 未成功获得可验证的远端模型响应: %s"
                % exc,
                file=sys.stderr,
            )
            return 4
        except Exception as exc:
            print(
                "[LLM probe] 失败: 本地请求准备或未知错误: %s"
                % type(exc).__name__,
                file=sys.stderr,
            )
            return 4
        receipt = gate["receipt"]
        usage = receipt.get("usage") or {}
        print(
            "[LLM probe] 成功: actual_remote_call=yes http_status=%s "
            "tls_verification=%s"
            % (receipt.get("http_status"), receipt.get("tls_verification"))
        )
        print(
            "[LLM probe] requested_model=%s response_model=%s "
            "total_tokens=%s content_sha256=%s"
            % (
                receipt.get("requested_model"),
                receipt.get("response_model") or "unknown",
                usage.get("total_tokens", "unknown"),
                receipt["content_sha256"],
            )
        )
        print(_render_llm_proxy_status(proxy_status))
        return 0

    token = None
    if args.token_stdin:
        if not args.base_url or not args.model:
            print("[LLM] --token-stdin 必须同时提供 --base-url 和 --model", file=sys.stderr)
            return 2
        token = sys.stdin.readline().rstrip("\r\n")
    elif not (sys.stdin.isatty() and sys.stdout.isatty()):
        print(
            "[LLM] 当前不是交互终端，拒绝等待凭据输入。CI 请使用 NUMPILOT_LLM_* "
            "环境变量，或用 install -m 600 llm.example.json llm.json 后手填，"
            "或运行 configure-llm --init。",
            file=sys.stderr,
        )
        return 2

    try:
        status = configure_llm_interactive(
            base_url=args.base_url,
            model=args.model,
            provider=args.provider,
            protocol=args.protocol,
            response_model_aliases=(
                tuple(args.response_model_alias)
                if args.response_model_alias is not None
                else None
            ),
            auth_token=token,
        )
    except (EOFError, KeyboardInterrupt):
        print("\n[LLM] 已取消，未写入配置。", file=sys.stderr)
        return 130
    except LlmConfigError as exc:
        print(f"[LLM] 配置失败: {exc}", file=sys.stderr)
        return 2
    print(status.summary())
    print(f"配置已安全保存: {status.path} (权限 0600)")
    return 0


def _cmd_check_llm(args):
    """Send one bounded message through the configured production transport."""

    from numpilot.llm_check import run_llm_check_command

    return run_llm_check_command(config_path=getattr(args, "llm_config", None))


def _cmd_setup_skills(args):
    """Fetch or verify the pinned official CANNBot registry-invoke skill."""

    cannbot = _load_cannbot_api()
    skill_root = args.root or "auto"
    if (
        skill_root == "auto"
        and not args.check
        and not args.accept_cannbot_license
    ):
        print(
            "[CANNBot] 获取官方 Skill 前必须显式接受 CANN Open Software "
            "License Agreement v2.0；请审阅许可证后重试并传入 "
            "--accept-cannbot-license。",
            file=sys.stderr,
        )
        return 2
    bundle = resolve_skill_bundle(skill_root, allow_fetch=not args.check)
    action = "verified" if args.check or args.root else "ready"
    print(
        "[CANNBot] %s skill=%s commit=%s"
        % (action, bundle.skill_name, bundle.commit)
    )
    print(
        "[CANNBot] files=%d bytes=%d bundle_sha256=%s"
        % (len(bundle.files), bundle.total_bytes, bundle.bundle_sha256)
    )
    print("[CANNBot] checkout=%s" % bundle.root)
    return 0


def _cmd_recover_gaps(args):
    """Ingest receipts, materialize a verified candidate, or inspect state."""

    # Keep the ordinary direct lane independent from optional gap tooling.
    # Importing ``numpilot.gaps`` transitively loads provider/CANNBot modules,
    # so the recovery stack must remain behind this explicit command boundary.
    from numpilot.gaps.recovery import RecoveryError
    from numpilot.gaps.recovery_ingest import (
        RecoveryIngestError,
        ingest_official_recovery,
        recovery_status,
    )

    try:
        if args.recovery_command == "ingest-official":
            result = ingest_official_recovery(
                args.source,
                args.migration_report,
                args.gap_request,
                args.official_resolution_report,
                args.state_dir,
                run_id=args.run_id,
                profile=args.authorization_profile,
            )
            if args.json:
                print(json.dumps(
                    result.to_dict(),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                ))
            else:
                status = result.recovery
                action = "新建" if result.created else "续接"
                print(
                    "[缺口恢复] %s状态成功 · 阶段=%s · 本次追加=%d · "
                    "总修订=%d · 下一步=%s"
                    % (
                        action,
                        status.state.phase.value,
                        result.appended_revisions,
                        status.history_length,
                        status.next_action,
                    )
                )
            return 0

        if args.recovery_command == "materialize-candidates":
            from numpilot.gaps.candidate_materializer import (
                CandidateMaterializationError,
                materialize_official_binary_candidates,
            )

            try:
                result = materialize_official_binary_candidates(
                    args.state_dir,
                    args.out,
                )
            except CandidateMaterializationError as exc:
                print(
                    "[候选物化] %s: %s" % (exc.code, exc),
                    file=sys.stderr,
                )
                return 2
            if args.json:
                print(json.dumps(
                    result.to_dict(),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                ))
            else:
                print(
                    "[候选物化] 官方二进制候选=%d · 阶段=%s · "
                    "本次追加=%d · 模型调用=no · 激活=no"
                    % (
                        len(result.candidates),
                        result.recovery.state.phase.value,
                        result.appended_revisions,
                    )
                )
                print("[候选物化] 输出=%s" % result.output_dir)
            return 0

        if args.recovery_command == "ingest-codex-adapter":
            from numpilot.gaps.adapter_recovery import (
                AdapterRecoveryError,
                ingest_codex_adapter_candidate,
            )

            try:
                result = ingest_codex_adapter_candidate(
                    args.state_dir,
                    args.candidate,
                    args.adapter_output,
                )
            except AdapterRecoveryError as exc:
                print(
                    "[Codex候选导入] %s: %s" % (exc.code, exc),
                    file=sys.stderr,
                )
                return 2
            if args.json:
                print(json.dumps(
                    result.to_dict(),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                ))
            else:
                print(
                    "[Codex候选导入] gap=%s · 阶段=%s · 本次追加=%d · "
                    "producer_model=no · network=no · build/execution/activation=no"
                    % (
                        result.gap_id,
                        result.recovery.state.phase.value,
                        result.appended_revisions,
                    )
                )
                print("[Codex候选导入] 输出=%s" % result.adapter_output_root)
            return 0

        if args.recovery_command == "ingest-adapter":
            from numpilot.gaps.adapter_recovery import (
                AdapterRecoveryError,
                ingest_official_adapter_candidate,
            )

            try:
                result = ingest_official_adapter_candidate(
                    args.state_dir,
                    args.candidate,
                    args.adapter_output,
                )
            except AdapterRecoveryError as exc:
                print(
                    "[接入候选导入] %s: %s" % (exc.code, exc),
                    file=sys.stderr,
                )
                return 2
            if args.json:
                print(json.dumps(
                    result.to_dict(),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                ))
            else:
                print(
                    "[接入候选导入] gap=%s · 阶段=%s · 本次追加=%d · "
                    "producer_model=yes · build/execution/activation=no"
                    % (
                        result.gap_id,
                        result.recovery.state.phase.value,
                        result.appended_revisions,
                    )
                )
                print("[接入候选导入] 输出=%s" % result.adapter_output_root)
            return 0

        if args.recovery_command == "ingest-adapter-repair":
            from numpilot.gaps.adapter_recovery import (
                AdapterRecoveryError,
                ingest_repaired_adapter_candidate,
            )

            try:
                result = ingest_repaired_adapter_candidate(
                    args.state_dir,
                    args.candidate,
                    args.adapter_output,
                )
            except AdapterRecoveryError as exc:
                print(
                    "[接入候选修复导入] %s: %s" % (exc.code, exc),
                    file=sys.stderr,
                )
                return 2
            if args.json:
                print(json.dumps(
                    result.to_dict(),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                ))
            else:
                print(
                    "[接入候选修复导入] gap=%s · 阶段=%s · 修复轮=%d · "
                    "本次追加=%d · producer_model=yes · build/execution=no"
                    % (
                        result.gap_id,
                        result.recovery.state.phase.value,
                        result.repairs_used,
                        result.appended_revisions,
                    )
                )
                print("[接入候选修复导入] 输出=%s" % result.adapter_output_root)
            return 0

        if args.recovery_command == "preflight-integrate-candidate":
            from numpilot.gaps.integration import (
                IntegrationError,
                preflight_official_adapter_integration,
            )

            try:
                result = preflight_official_adapter_integration(
                    args.state_dir,
                    args.candidate,
                    args.adapter_output,
                    args.asnumpy_reference,
                    python_executable=args.python,
                )
            except IntegrationError as exc:
                print("[集成预检] %s: %s" % (exc.code, exc), file=sys.stderr)
                return 2
            payload = result.to_dict()
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True))
            else:
                print(
                    "[集成预检] gap=%s · status=%s · code=%s · profile=%s · next=%s"
                    % (
                        result.gap_id,
                        result.status,
                        result.code,
                        result.generation_mode,
                        result.next_action,
                    )
                )
                if result.missing_policy:
                    print(
                        "[集成预检] missing_policy=%s" % ",".join(result.missing_policy),
                        file=sys.stderr,
                    )
                print(
                    "[集成预检] model=no · build=no · import=no · execution=no · activation=no"
                )
            return 0 if result.status == "ready" else 2

        if args.recovery_command == "integrate-candidate":
            from numpilot.gaps.integration import (
                IntegrationError,
                run_official_adapter_integration,
            )

            try:
                result = run_official_adapter_integration(
                    args.state_dir,
                    args.candidate,
                    args.adapter_output,
                    args.asnumpy_reference,
                    args.out,
                    python_executable=args.python,
                    timeout_seconds=args.timeout,
                )
            except IntegrationError as exc:
                print("[隔离集成] %s: %s" % (exc.code, exc), file=sys.stderr)
                return 2
            if args.json:
                print(json.dumps(
                    result.to_dict(),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                ))
            else:
                print(
                    "[隔离集成] gap=%s · status=%s · 阶段=%s · "
                    "本次追加=%d · model=no · activation=no"
                    % (
                        result.gap_id,
                        result.status,
                        result.recovery.state.phase.value,
                        result.appended_revisions,
                    )
                )
                print("[隔离集成] 事务=%s" % result.transaction_ref.locator)
                if result.diagnostic_ref is not None:
                    print("[隔离集成] 诊断=%s" % result.diagnostic_ref.locator)
                if result.manual_review_required:
                    print(
                        "[隔离集成] MANUAL_REVIEW_REQUIRED",
                        file=sys.stderr,
                    )
            return 0

        status = recovery_status(args.state_dir)
        if args.json:
            print(json.dumps(
                status.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            ))
        else:
            print(
                "[缺口恢复] 阶段=%s · 结果=%s · 总修订=%d · 下一步=%s"
                % (
                    status.state.phase.value,
                    status.state.outcome.value,
                    status.history_length,
                    status.next_action,
                )
            )
        return 0
    except RecoveryIngestError as exc:
        print(
            "[缺口恢复导入] %s: %s" % (exc.code, exc),
            file=sys.stderr,
        )
        return 2
    except RecoveryError as exc:
        print("[缺口恢复] %s" % exc, file=sys.stderr)
        return 2


def _cmd_prepare_official_candidate(args):
    """Publish a hash-bound installed ACLNN candidate hand-off only."""

    from numpilot.gaps.official_candidate_handoff import (
        OfficialCandidateHandoffError,
        prepare_official_candidate_handoff,
    )

    try:
        handoff = prepare_official_candidate_handoff(
            args.file, args.official_resolution_report, args.out
        )
    except OfficialCandidateHandoffError as exc:
        flow_print("[官方候选 hand-off] 失败: %s" % exc, stream=sys.stderr, tone="error")
        flow_print("[模型调用] actual_remote_call=no · 未读取 Skill、未构建、未激活")
        return 2
    if args.json:
        print(json.dumps(handoff.to_dict(), ensure_ascii=False, allow_nan=False, sort_keys=True))
    else:
        flow_print("[官方候选 hand-off] status=candidate_adapter_required · model_call=no · activation=no")
        flow_print("[官方候选 hand-off] receipt=%s id=%s" % (args.out, handoff.handoff_id))
    return 0


def _cmd_preflight_official_candidate(args):
    """Recheck an official candidate hand-off without model/build side effects."""
    from numpilot.gaps.official_candidate_preflight import (
        OfficialCandidatePreflightError,
        preflight_official_candidate,
    )
    try:
        receipt = preflight_official_candidate(
            args.handoff,
            args.file,
            args.official_resolution_report,
            args.out,
        )
    except OfficialCandidatePreflightError as exc:
        print("[官方候选 preflight] 失败: %s" % exc, file=sys.stderr)
        print("[模型调用] actual_remote_call=no · build=no · activation=no", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(receipt, ensure_ascii=False, allow_nan=False, sort_keys=True))
    else:
        print("[官方候选 preflight] status=%s · model_call=no · build=no · activation=no" % receipt["status"])
        print("[官方候选 preflight] receipt=%s" % args.out)
    return 0

def _cmd_prepare_official_adapter_input(args):
    """Prepare an inert official adapter input from a validated preflight."""
    from numpilot.gaps.official_candidate_preflight import (
        OfficialCandidatePreflightError,
        prepare_official_adapter_input_from_preflight,
    )
    try:
        prepared = prepare_official_adapter_input_from_preflight(
            args.handoff, args.preflight, args.file, args.official_resolution_report
        )
        payload = prepared.to_dict()
        output = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(output) or ".", mode=0o700, exist_ok=True)
        encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if fd >= 0:
                os.close(fd)
    except (OfficialCandidatePreflightError, OSError, TypeError, ValueError) as exc:
        print("[官方 adapter input] 失败: %s" % exc, file=sys.stderr)
        print("[模型调用] actual_remote_call=no · build=no · activation=no", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True))
    else:
        print("[官方 adapter input] status=inert · strategy=%s · model_call=no · activation=no" % payload["operator_manifest"]["strategy"])
        print("[官方 adapter input] receipt=%s" % args.out)
    return 0


def _cmd_validate_official_adapter_context(args):
    """Validate the official adapter model gate without invoking it."""
    from numpilot.gaps.official_adapter_context_gate import (
        OfficialAdapterContextGateError,
        validate_official_adapter_context,
    )
    try:
        payload = validate_official_adapter_context(
            args.bound,
            args.handoff,
            args.preflight,
            args.source,
            args.official_resolution_report,
        )
        output = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(output) or ".", mode=0o700, exist_ok=True)
        encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if fd >= 0:
                os.close(fd)
    except (OfficialAdapterContextGateError, OSError, TypeError, ValueError) as exc:
        print("[官方 adapter gate] 失败: %s" % exc, file=sys.stderr)
        print("[模型调用] actual_remote_call=no · build=no · activation=no", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True))
    else:
        print("[官方 adapter gate] status=%s · model_call=no · activation=no" % payload["status"])
        print("[官方 adapter gate] receipt=%s" % args.out)
    return 0


def _cmd_materialize_official_candidate_recovery(args):
    """Advance an official candidate hand-off to CANDIDATE_READY only."""
    from numpilot.gaps.official_candidate_recovery import (
        OfficialCandidateRecoveryError,
        materialize_official_candidate_recovery,
    )
    try:
        result = materialize_official_candidate_recovery(
            args.handoff,
            args.preflight,
            args.source,
            args.official_resolution_report,
            args.context_gate,
            args.state_dir,
            runtime_backend=args.runtime_backend,
            runtime_audit_id=args.runtime_audit_id,
            runtime_identity_sha256=args.runtime_identity_sha256,
            run_id=args.run_id,
            profile=args.profile,
            output_dir=args.output_dir,
        )
    except (OfficialCandidateRecoveryError, OSError, TypeError, ValueError) as exc:
        print("[官方候选 recovery] 失败: %s" % exc, file=sys.stderr)
        print("[模型调用] actual_remote_call=no · build=no · activation=no", file=sys.stderr)
        return 2
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True))
    else:
        print("[官方候选 recovery] status=%s · phase=%s · appended=%d" % (
            payload["status"], payload["recovery"]["state"]["phase"], payload["appended_revisions"]
        ))
        print("[官方候选 recovery] output=%s · model_call=no · activation=no" % payload["output_dir"])
    return 0


def _cmd_resolve_gaps(args):
    """Resolve canonical gaps locally before selecting any model workflow."""

    from numpilot.gaps.official_resolution import (
        OfficialResolutionError,
        resolve_official_operators,
        write_official_resolution_report,
    )
    from numpilot.gaps.pipeline import (
        GapCompletionError,
        load_script_gap_request,
        prepare_gap_completion,
    )

    requested_specs = None
    requested_blocked = ()
    try:
        if args.gap_request:
            selection = load_script_gap_request(
                args.gap_request,
                args.file,
                target_soc=args.target_soc,
                cann_version=args.cann_version,
            )
            requested_specs = selection.specs
            requested_blocked = selection.blocked
        preparation = prepare_gap_completion(
            args.file,
            target_soc=args.target_soc,
            cann_version=args.cann_version,
            requested_specs=requested_specs,
            requested_blocked=requested_blocked,
        )
        report = resolve_official_operators(
            preparation,
            cann_root=args.cann_root,
            ops_nn_root=args.ops_nn_root,
            accept_cann_osl=args.accept_cann_osl,
            allow_custom_operator_generation=getattr(
                args, "allow_custom_operator_generation", False
            ),
        )
        report.validate_preparation(preparation)
        write_official_resolution_report(report, args.out)
    except (GapCompletionError, OfficialResolutionError) as exc:
        flow_print("[官方算子解析] 失败: %s" % exc, stream=sys.stderr, tone="error")
        flow_print(
            "[模型调用] actual_remote_call=no · 未读取 Skill、未构建、未安装、未激活",
            stream=sys.stderr,
            tone="warning",
        )
        return 2

    flow_print(
        "[官方算子解析] status=%s decisions=%d model_kernel_allowed=%s"
        % (
            report.status,
            len(report.decisions),
            "yes" if report.model_kernel_generation_allowed else "no",
        ),
        tone="error" if report.status == "blocked" else "success",
    )
    flow_print(
        "[官方算子解析] receipt=%s id=%s" % (args.out, report.resolution_id),
        tone="artifact",
    )
    for decision in report.decisions:
        flow_print(
            "[官方路由] %s -> %s · installed=%s · ops-nn=%s"
            % (
                decision.qualified_api,
                decision.route,
                decision.installed_cann.get("status", "?"),
                (
                    decision.ops_nn_source.get("status", "?")
                    if decision.ops_nn_source is not None else "not_read"
                ),
            ),
            tone="info",
        )
    flow_print(
        "[解析边界] discovery_only=yes semantic_qualified=no build=no "
        "system_opp_write=no activation=no network=no model_call=no",
        tone="warning",
    )
    return 1 if report.status == "blocked" else 0


def _cmd_complete_gaps(args):
    """Prepare gap specs or ask the configured LLM for an unverified draft."""
    from numpilot.gaps.pipeline import (
        GapCompletionError,
        GapCompletionPipeline,
        load_script_gap_request,
        prepare_gap_completion,
    )
    from numpilot.gaps.provider import (
        DraftGenerationError,
        LlmDraftProvider,
    )

    requested_specs = None
    requested_blocked = ()
    if args.gap_request:
        try:
            selection = load_script_gap_request(
                args.gap_request,
                args.file,
                target_soc=args.target_soc,
                cann_version=args.cann_version,
            )
        except GapCompletionError as exc:
            print("[缺口请求] 本地校验失败: %s" % exc, file=sys.stderr)
            return 2
        requested_specs = selection.specs
        requested_blocked = selection.blocked
        print(
            "[缺口请求] 已绑定普通脚本 GapSpec=%d blocked=%d；尚未调用模型"
            % (len(requested_specs), len(requested_blocked))
        )

    try:
        preparation = prepare_gap_completion(
            args.file,
            target_soc=args.target_soc,
            cann_version=args.cann_version,
            requested_specs=requested_specs,
            requested_blocked=requested_blocked,
        )
    except GapCompletionError as exc:
        print("[算子补全] 本地预分析失败: %s" % exc, file=sys.stderr)
        print(
            "[算子生成] actual_remote_generation_call=no · 未读取 Skill",
            file=sys.stderr,
        )
        return 2

    llm_enabled = args.provider == "llm" and bool(preparation.specs)
    injected_enabled = args.provider == "injected" and bool(preparation.specs)
    provider_enabled = llm_enabled or injected_enabled
    operator_goal = None
    if provider_enabled:
        # Reject an unimplemented local mode before asking the user to produce
        # any official-resolution receipt.  This remains before model, Skill,
        # network, and output access.
        operator_goal = _resolve_operator_goal(args)
        if operator_goal == "performance":
            return 2

    official_report = None
    if args.official_resolution_report:
        from numpilot.gaps.official_resolution import (
            OfficialResolutionError,
            load_official_resolution_report,
            resolve_official_operators,
        )

        try:
            official_report = load_official_resolution_report(
                args.official_resolution_report
            )
            official_report.validate_preparation(preparation)
            receipt_custom_routes = [
                decision for decision in official_report.decisions
                if decision.route == "custom_operator_generation_allowed"
            ]
            if receipt_custom_routes and not getattr(
                args, "allow_custom_operator_generation", False
            ):
                print(
                    "[官方算子门禁] 收据包含 custom OperatorSpec 路由；必须显式提供 "
                    "--allow-custom-operator-generation，默认不调用模型",
                    file=sys.stderr,
                )
                print(
                    "[算子生成] actual_remote_generation_call=no · 未读取 Skill",
                    file=sys.stderr,
                )
                return 3
            current_report = resolve_official_operators(
                preparation,
                cann_root=official_report.cann_root,
                ops_nn_root=official_report.ops_nn_root,
                accept_cann_osl=official_report.cann_osl_accepted,
                allow_custom_operator_generation=getattr(
                    args, "allow_custom_operator_generation", False
                ),
            )
            current_report.validate_preparation(preparation)
            if current_report.resolution_id != official_report.resolution_id:
                raise OfficialResolutionError(
                    "official-resolution evidence changed after the receipt was written"
                )
            official_report = current_report
        except OfficialResolutionError as exc:
            print("[官方算子门禁] 本地收据校验失败: %s" % exc, file=sys.stderr)
            print(
                "[算子生成] actual_remote_generation_call=no · 未读取 Skill",
                file=sys.stderr,
            )
            return 2
        print(
            "[官方算子门禁] receipt=%s status=%s id=%s fresh_recheck=yes"
            % (
                args.official_resolution_report,
                official_report.status,
                official_report.resolution_id,
            )
        )
        if preparation.specs and not official_report.model_kernel_generation_allowed:
            for decision in official_report.decisions:
                print(
                    "[官方路由] %s -> %s"
                    % (decision.qualified_api, decision.route),
                    file=sys.stderr,
                )
            print(
                "[官方算子门禁] 未取得全部 Gap 的双重 clean no_match；"
                "不得调用模型手写 kernel",
                file=sys.stderr,
            )
            print(
                "[算子生成] actual_remote_generation_call=no · 未读取 Skill",
                file=sys.stderr,
            )
            return 3
    elif args.provider == "llm" and preparation.specs:
        print(
            "[官方算子门禁] fail · llm 新 kernel 必须提供 "
            "--official-resolution-report；请先运行 resolve-gaps，证明 "
            "installed CANN/ops-nn 均为 clean no_match。",
            file=sys.stderr,
        )
        print(
            "[算子生成] actual_remote_generation_call=no · 未读取 Skill",
            file=sys.stderr,
        )
        return 3

    configuration_status = None
    llm_gate = None
    timeout_seconds = None
    if injected_enabled:
        response_path = Path(args.response_file).expanduser() if args.response_file else None
        if response_path is None:
            print(
                "[注入模型门禁] fail · --provider injected 必须提供 --response-file；"
                "不会读取 llm.json，也不会发起网络请求",
                file=sys.stderr,
            )
            return 2
        if response_path.is_symlink() or not response_path.is_file():
            print(
                "[注入模型门禁] fail · response-file 必须是本地普通文件",
                file=sys.stderr,
            )
            return 2
        try:
            response_bytes = response_path.read_bytes()
        except OSError as exc:
            print(
                "[注入模型门禁] fail · 无法读取 response-file: %s" % type(exc).__name__,
                file=sys.stderr,
            )
            return 2
        if not response_bytes or len(response_bytes) > 3 * 1024 * 1024:
            print(
                "[注入模型门禁] fail · response-file 为空或超过 3 MiB",
                file=sys.stderr,
            )
            return 2
        try:
            injected_response = response_bytes.decode("utf-8")
        except UnicodeDecodeError:
            print(
                "[注入模型门禁] fail · response-file 必须是 UTF-8 文本",
                file=sys.stderr,
            )
            return 2
        print(
            "[注入模型门禁] pass · local response=%s · bytes=%d · "
            "actual_remote_generation_call=no" % (response_path, len(response_bytes))
        )

    if llm_enabled:
        if str(args.cann_version).strip().lower() in {"", "unknown", "auto"}:
            print(
                "[生成门禁 0/2] CANN 版本必须明确，已在联网前拦截",
                file=sys.stderr,
            )
            return 2
        raw_timeout = (
            str(args.llm_timeout)
            if args.llm_timeout is not None
            else os.getenv("NUMPILOT_LLM_TIMEOUT", "120")
        )
        try:
            timeout_seconds = validate_llm_timeout(raw_timeout)
        except LlmConfigError as exc:
            print("[生成门禁 1/2] %s" % exc, file=sys.stderr)
            return 2
        os.environ["NUMPILOT_LLM_TIMEOUT"] = str(timeout_seconds)
        configuration_status = _load_llm_or_report()
        if configuration_status is None:
            print("[生成门禁 1/2] fail · LLM 本地配置不可用", file=sys.stderr)
            return 4
        if _llm_proxy_for_remote_call("生成门禁 1/2") is None:
            return 4
        try:
            require_model_status(configuration_status)
        except ModelContractError as exc:
            print("[生成门禁 1/2] fail · %s" % exc, file=sys.stderr)
            return 4
        try:
            llm_gate = _run_llm_readiness_gate(
                configuration_status, require_marker=True
            )
        except (
            LlmHttpError,
            LlmTransportError,
            LlmResponseError,
        ) as exc:
            print(
                "[生成门禁 1/2] fail · 简单问答没有取得可验证响应: %s" % exc,
                file=sys.stderr,
            )
            print(
                "[算子生成] actual_remote_generation_call=no · 未读取 Skill，"
                "未开始算子生成",
                file=sys.stderr,
            )
            return 4
        except Exception as exc:
            print(
                "[生成门禁 1/2] fail · 简单问答本地错误: %s"
                % type(exc).__name__,
                file=sys.stderr,
            )
            return 4
        probe_receipt = llm_gate["receipt"]
        print(
            "[生成门禁 1/2] pass · LLM 简单问答 HTTP %s · model=%s · "
            "timeout=%ss"
            % (
                probe_receipt.get("http_status"),
                probe_receipt.get("response_model"),
                timeout_seconds,
            )
        )

    # ``prepare`` is deliberately a local-only operation: it serializes the
    # canonical GapSpec and must never read or fetch an external Skill.  The
    # LLM lane consumes only a checkout that an explicit bootstrap/setup step
    # has already pinned and cached; gap completion itself has no hidden
    # network side effect or implicit CANNBot licence acceptance.
    bundle = None
    skill_gate = None
    if llm_enabled:
        cannbot = _load_cannbot_api()
        bundle = resolve_skill_bundle(args.skill_root, allow_fetch=False)
        skill_gate = _run_cannbot_skill_gate(
            bundle,
            required_files=cannbot.CANNBOT_GENERATION_SKILL_FILES,
        )
        print(
            "[生成门禁 2/2] pass · 已读取 CANNBot Skill · commit=%s · "
            "files=%s · bundle_sha256=%s"
            % (bundle.commit, len(bundle.files), bundle.bundle_sha256)
        )
        display = getattr(args, "show_cannbot_skill", "none")
        if display in {"summary", "full"}:
            selected = tuple(cannbot.CANNBOT_GENERATION_SKILL_FILES)
            print(
                "[CANNBot Skill] 本次请求实际嵌入的官方生成上下文：%d 个文件 · %d bytes"
                % (
                    len(selected),
                    len(bundle.prompt_context(selected).encode("utf-8")),
                )
            )
            for relative in selected:
                print(
                    "[CANNBot Skill] file=%s sha256=%s"
                    % (relative, bundle.file_sha256.get(relative, "?"))
                )
            if display == "full":
                print("[CANNBot Skill] ===== embedded official context begin =====")
                print(bundle.prompt_context(selected), end="")
                if not bundle.prompt_context(selected).endswith("\n"):
                    print()
                print("[CANNBot Skill] ===== embedded official context end =====")
            else:
                print("[CANNBot Skill] 正文展示为 summary；可用 --show-cannbot-skill full 展开。")
        try:
            _write_generation_preflight(
                args.out,
                llm_gate,
                skill_gate,
                timeout_seconds,
                operator_goal=operator_goal,
            )
        except OSError as exc:
            print(
                "[生成门禁] 无法落盘门禁收据: %s" % type(exc).__name__,
                file=sys.stderr,
            )
            return 2
    provider = None
    if llm_enabled:
        print(
            "[CANNBot] 使用已缓存官方 skill=%s commit=%s bundle_sha256=%s"
            % (bundle.skill_name, bundle.commit, bundle.bundle_sha256)
        )
        provider = LlmDraftProvider(
            max_attempts=args.max_attempts,
            configuration_status=configuration_status,
            skill_bundle=bundle,
            staged_direct=True,
            operator_goal=operator_goal,
        )
    elif injected_enabled:
        # Codex/other host agents can submit one locally generated bundle without
        # exposing credentials or allowing the CLI to silently call llm.json.
        # The normal NumPilot validator, artifact store and fail-closed gates still
        # own the candidate; the response is explicitly recorded as an injected
        # test-double rather than a remote model result.
        provider = LlmDraftProvider(
            chat=lambda _system, _user: injected_response,
            max_attempts=args.max_attempts,
            provider_label="codex-self",
            model_name="codex-local",
            operator_goal=operator_goal,
        )
        print(
            "[注入模型] 使用本地候选响应；provider=codex-self model=codex-local "
            "· 不读取 llm.json · 不联网 · 仍执行完整本地合同校验"
        )

    pipeline = GapCompletionPipeline(
        provider=provider,
        skill_root=bundle.root if bundle is not None else "",
        asnumpy_reference=args.asnumpy_reference,
        skill_bundle=bundle,
    )
    progress_stop = threading.Event() if provider is not None else None
    progress_thread = None
    if progress_stop is not None:
        progress_thread = _interactive_heartbeat(
            progress_stop,
            "CANNBot/配置模型分阶段生成算子文件",
            interval=5.0,
        )
    result = None
    try:
        result = pipeline.run(
            args.file,
            args.out,
            target_soc=args.target_soc,
            cann_version=args.cann_version,
            requested_specs=requested_specs,
            requested_blocked=requested_blocked,
            preparation=preparation,
            continue_on_error=getattr(args, "continue_on_error", False),
        )
    except GapCompletionError as exc:
        print(f"[算子补全] 失败: {exc}", file=sys.stderr)
        if args.provider == "llm":
            print(
                "[算子生成] verified_new_remote_artifact=no；本次没有获得经远端回执"
                "绑定且通过本地校验的新模型产物（请求可能未发出、远端失败，或响应"
                "校验失败），并且未回退到 AST。",
                file=sys.stderr,
            )
        return 2
    finally:
        if progress_stop is not None:
            progress_stop.set()
        if progress_thread is not None:
            progress_thread.join(timeout=1.0)
    if provider is not None and result.drafts:
        report_path = os.path.join(result.out_dir, "gap-report.json")
        if os.path.islink(report_path) or not os.path.isfile(report_path):
            raise GapCompletionError("gap report is not a regular file")
        with open(report_path, "rb") as handle:
            report_payload = handle.read(8 * 1024 * 1024 + 1)
        if len(report_payload) > 8 * 1024 * 1024:
            raise GapCompletionError("gap report exceeds the size limit")
        report = json.loads(report_payload.decode("utf-8"))
        actions = {
            draft.gap_id: provider.action_for(draft)
            for draft in result.drafts
        }
        report_drafts = report.get("drafts")
        if not isinstance(report_drafts, list):
            raise GapCompletionError("gap report draft list is invalid")
        seen = set()
        for draft_data in report_drafts:
            if not isinstance(draft_data, dict):
                raise GapCompletionError("gap report draft entry is invalid")
            gap_id = draft_data.get("gap_id")
            if gap_id not in actions:
                raise GapCompletionError("gap report draft identity mismatch")
            action = actions[gap_id]
            draft_data["current_run_action"] = action
            draft_data["current_run_actual_remote_call"] = (
                action == "fresh_remote"
            )
            seen.add(gap_id)
        if seen != set(actions):
            raise GapCompletionError("gap report is missing a generated draft")
        report["current_run_actions"] = dict(sorted(actions.items()))
        report["current_run_actual_remote_call_count"] = sum(
            action == "fresh_remote" for action in actions.values()
        )
        payload = (
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        fd = -1
        temp_name = ""
        try:
            fd, temp_name = tempfile.mkstemp(
                prefix=".gap-report-", suffix=".tmp", dir=result.out_dir
            )
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, report_path)
            temp_name = ""
            directory_fd = os.open(result.out_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if fd >= 0:
                os.close(fd)
            if temp_name:
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass
    print(f"[算子补全] status={result.status}")
    print(f"[算子补全] report={os.path.join(result.out_dir, 'gap-report.json')}")
    for spec in result.specs:
        workspace = os.path.join(result.out_dir, "gaps", spec.gap_id)
        print(f"[缺口] {spec.qualified_api} -> {workspace}")
    for item in result.blocked:
        print(
            "[阻断] {name}: {reason}".format(
                name=item.get("qualified_api") or item.get("name", "?"),
                reason=item.get("reason", "unknown"),
            ),
            file=sys.stderr,
        )
    for item in result.generation_failures:
        # Failure records are intentionally sanitized by the pipeline; never
        # print an exception string or a model response here.
        detail = item.get("detail", "generation failed")
        status = item.get("http_status")
        suffix = " HTTP %s" % status if status is not None else ""
        remote = item.get("actual_remote_call")
        remote_text = "unknown" if remote is None else str(remote).lower()
        print(
            "[生成失败] {api}: {detail}{suffix} · actual_remote_call={remote}".format(
                api=item.get("qualified_api", "?"),
                detail=detail,
                suffix=suffix,
                remote=remote_text,
            ),
            file=sys.stderr,
        )
    if result.drafts:
        print(
            "[安全门禁] 已生成 Ascend C 草稿，但尚未编译、单测、数值校验或激活；"
            "当前不会注册到 AsNumpy。"
        )
        for draft in result.drafts:
            evidence = draft.generation
            hashes = evidence.get("response_sha256") or []
            action = provider.action_for(draft) if provider is not None else "unknown"
            print(
                "[模型证据] action={action} mode={mode} provider={provider} model={model} "
                "attempts={attempts} fallback={fallback} response_sha256={digest}".format(
                    action=action,
                    mode=evidence.get("mode", "?"),
                    provider=evidence.get("provider", "?"),
                    model=evidence.get("model", "?"),
                    attempts=evidence.get("attempt_count", "?"),
                    fallback=evidence.get("fallback", "?"),
                    digest=hashes[-1] if hashes else "?",
                )
            )
            if action == "fresh_remote" and evidence.get("mode") == "remote_llm":
                receipts = evidence.get("remote_receipts") or []
                statuses = ",".join(
                    str(receipt.get("http_status", "?")) for receipt in receipts
                ) or "?"
                print(
                    "[算子生成] actual_remote_generation_call=yes "
                    "verified_new_remote_artifact=yes receipts={count} "
                    "http_statuses={statuses} content_hash_bound=yes".format(
                        count=len(receipts),
                        statuses=statuses,
                    )
                )
            elif action == "reused_existing":
                print(
                    "[算子生成] actual_remote_generation_call=no "
                    "action=reused_existing；"
                    "本次只复验并复用已有产物，没有重新请求大模型。"
                )
            else:
                print(
                    "[算子生成] actual_remote_generation_call=no action={action}；"
                    "本次没有发起真实远端模型调用。".format(action=action)
                )
    elif provider is not None:
        print(
            "[算子生成] actual_remote_generation_call=no；"
            "本次没有可生成的合格缺口，"
            "因此没有请求大模型。"
        )
    if result.generation_failures:
        print(
            "[算子生成] 本次有 %d 个缺口未生成；已保留其 GapSpec/请求文件，"
            "未生成可激活产物。" % len(result.generation_failures),
            file=sys.stderr,
        )
        return 2
    return 1 if result.blocked else 0


def _cmd_repair_draft(args):
    """Repair selected signed-draft files into a new unverified candidate."""

    from pathlib import Path

    from numpilot.gaps.models import GapSpec
    from numpilot.gaps.provider import (
        DraftGenerationError,
        DraftRepairError,
        LlmDraftProvider,
    )

    def read_bounded(path, label, limit):
        target = Path(path)
        if target.is_symlink() or not target.is_file():
            raise DraftGenerationError("%s must be a regular file" % label)
        payload = target.read_bytes()
        if len(payload) > limit:
            raise DraftGenerationError("%s exceeds the size limit" % label)
        return payload.decode("utf-8")

    raw_timeout = (
        str(args.llm_timeout)
        if args.llm_timeout is not None
        else os.getenv("NUMPILOT_LLM_TIMEOUT", "120")
    )
    try:
        timeout_seconds = validate_llm_timeout(raw_timeout)
    except LlmConfigError as exc:
        print("[修复门禁 1/2] %s" % exc, file=sys.stderr)
        return 2
    os.environ["NUMPILOT_LLM_TIMEOUT"] = str(timeout_seconds)
    configuration_status = _load_llm_or_report()
    if configuration_status is None:
        print("[修复门禁 1/2] fail · LLM 本地配置不可用", file=sys.stderr)
        return 4
    if _llm_proxy_for_remote_call("修复门禁 1/2") is None:
        return 4
    try:
        llm_gate = _run_llm_readiness_gate(
            configuration_status, require_marker=True
        )
    except (LlmHttpError, LlmTransportError, LlmResponseError) as exc:
        print(
            "[修复门禁 1/2] fail · 简单问答没有取得可验证响应: %s" % exc,
            file=sys.stderr,
        )
        print(
            "[算子修复] actual_remote_repair_call=no · 未读取 Skill，"
            "未开始模型修复",
            file=sys.stderr,
        )
        return 4
    probe_receipt = llm_gate["receipt"]
    print(
        "[修复门禁 1/2] pass · LLM 简单问答 HTTP %s · model=%s · "
        "timeout=%ss"
        % (
            probe_receipt.get("http_status"),
            probe_receipt.get("response_model") or configuration_status.model,
            timeout_seconds,
        )
    )

    # Repair is code generation too: it may not read the external Skill until
    # the model-readiness call above has passed, and it may not contact the
    # model for repair until both gates have passed.
    cannbot = _load_cannbot_api()
    bundle = resolve_skill_bundle(
        args.skill_root,
        allow_fetch=not args.offline_skills,
    )
    repair_skill_files = {
        "compile": cannbot.CANNBOT_REPAIR_SKILL_FILES,
        "precision": cannbot.CANNBOT_PRECISION_SKILL_FILES,
        "runtime": cannbot.CANNBOT_RUNTIME_SKILL_FILES,
    }[args.repair_profile]
    skill_gate = _run_cannbot_skill_gate(
        bundle, required_files=repair_skill_files
    )
    print(
        "[修复门禁 2/2] pass · 已读取 CANNBot Skill · commit=%s · "
        "files=%s · bundle_sha256=%s"
        % (bundle.commit, len(bundle.files), bundle.bundle_sha256)
    )
    preflight_dir = Path(str(Path(args.out).absolute()) + ".preflight")
    try:
        _write_generation_preflight(
            preflight_dir, llm_gate, skill_gate, timeout_seconds
        )
    except OSError as exc:
        print(
            "[修复门禁] 无法落盘门禁收据: %s" % type(exc).__name__,
            file=sys.stderr,
        )
        return 2
    try:
        spec_data = json.loads(read_bounded(
            args.spec, "gap spec", 2 * 1024 * 1024
        ))
        if not isinstance(spec_data, dict):
            raise DraftGenerationError("gap spec must be a JSON object")
        spec = GapSpec.from_dict(spec_data)
        request = read_bounded(
            args.generation_request,
            "generation request",
            2 * 1024 * 1024,
        )
        diagnostic = read_bounded(
            args.diagnostic_file,
            "repair diagnostic",
            64 * 1024,
        )
        provider = LlmDraftProvider(
            max_attempts=args.max_attempts,
            configuration_status=configuration_status,
            skill_bundle=bundle,
            staged_direct=True,
            repair_skill_files=repair_skill_files,
        )
        report = provider.repair_to_candidate(
            spec,
            Path(args.draft),
            Path(args.out),
            paths=tuple(args.path),
            diagnostic=diagnostic,
            generation_request=request,
            base_candidate_dir=(
                Path(args.base_candidate) if args.base_candidate else None
            ),
        )
    except DraftRepairError as exc:
        print("[LLM repair] 失败: %s" % exc, file=sys.stderr)
        print(
            "[LLM调用] actual_remote_call=%s verified_new_remote_repair=no "
            "successful_http_receipts=%d；远端修复响应未通过本地校验。"
            % ("yes" if exc.actual_remote_call else "no", exc.receipt_count),
            file=sys.stderr,
        )
        return 2
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError,
            DraftGenerationError) as exc:
        print("[LLM repair] 失败: %s" % exc, file=sys.stderr)
        print(
            "[LLM调用] verified_new_remote_repair=no；未生成可发布修复副本。",
            file=sys.stderr,
        )
        return 2

    print(
        "[LLM repair] status=%s source_artifact_id=%s out=%s"
        % (report["status"], report["source_artifact_id"], args.out)
    )
    print("[修复门禁] receipt=%s" % (preflight_dir / "generation-preflight.json"))
    print(
        "[LLM调用] actual_remote_call=yes verified_new_remote_repair=yes "
        "attempts=%d receipts=%d content_hash_bound=yes"
        % (report["attempt_count"], len(report["remote_receipts"]))
    )
    return 0


def _cmd_seal_repair(args):
    """Seal a verified repair chain without making another model call."""

    from pathlib import Path

    from numpilot.gaps.models import GapSpec
    from numpilot.gaps.provider import DraftGenerationError, LlmDraftProvider

    try:
        spec_path = Path(args.spec)
        if spec_path.is_symlink() or not spec_path.is_file():
            raise DraftGenerationError("gap spec must be a regular file")
        payload = spec_path.read_bytes()
        if len(payload) > 2 * 1024 * 1024:
            raise DraftGenerationError("gap spec exceeds the size limit")
        data = json.loads(payload.decode("utf-8"))
        if not isinstance(data, dict):
            raise DraftGenerationError("gap spec must be a JSON object")
        # Sealing is deliberately offline.  The original repair report is
        # bound to a verified CANNBot bundle, so reloading that same pinned
        # bundle is part of verification but can never trigger a fetch here.
        cannbot = _load_cannbot_api()
        bundle = resolve_skill_bundle(args.skill_root, allow_fetch=False)
        draft = LlmDraftProvider(skill_bundle=bundle).seal_repair_candidate(
            GapSpec.from_dict(data),
            Path(args.draft),
            Path(args.candidate),
            Path(args.out),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError,
            CannbotSkillError, DraftGenerationError) as exc:
        print("[seal repair] 失败: %s" % exc, file=sys.stderr)
        print(
            "[LLM调用] actual_remote_call=no sealed_artifact=no",
            file=sys.stderr,
        )
        return 2
    print(
        "[seal repair] status=%s artifact_id=%s files=%d out=%s"
        % (draft.status, draft.artifact_id, len(draft.files), draft.workspace)
    )
    print(
        "[LLM调用] actual_remote_call=no action=sealed_repair_chain "
        "prior_receipts=%d"
        % len(draft.generation["remote_receipts"])
    )
    return 0



def _cmd_validate_asnumpy_adapter_handoff(args):
    """Validate a local Codex response package without reading llm.json."""

    from numpilot.gaps.adapter_handoff import load_adapter_handoff
    from numpilot.gaps.provider import (
        DraftGenerationError,
        DraftRepairError,
        LlmDraftProvider,
        _injected_generation_receipt,
        _validate_adapter_context_plan,
    )
    from numpilot.asnumpy_skill import load_asnumpy_integration_skill

    handoff_root = Path(args.handoff_dir).absolute()
    response_root = Path(args.response_dir).absolute()
    out_root = Path(args.out).absolute()
    if response_root.is_symlink() or not response_root.is_dir():
        print("[Codex交接] VALIDATE 失败：response-dir 必须是普通目录", file=sys.stderr)
        return 2
    if os.path.lexists(out_root):
        print("[Codex交接] VALIDATE 失败：输出目录必须不存在", file=sys.stderr)
        return 2
    try:
        handoff = load_adapter_handoff(handoff_root)
        manifest = handoff["manifest"]
        plan = _validate_adapter_context_plan(handoff["adapter_context_plan"])
        if tuple(plan["expected_overlay_paths"]) != handoff["expected_overlay_paths"]:
            raise DraftGenerationError("handoff context plan/path mismatch")
        if not isinstance(handoff["operator_artifact"], dict):
            raise DraftGenerationError("handoff operator artifact is invalid")
        if not isinstance(handoff["adapter_contract"], str):
            raise DraftGenerationError("handoff adapter contract is invalid")
        bundle = load_asnumpy_integration_skill(
            Path(args.asnumpy_skill_root) if args.asnumpy_skill_root else None
        )
        if bundle.bundle_sha256 != manifest.get("skill_bundle_sha256"):
            raise DraftGenerationError("handoff Skill bundle hash mismatch")
        _run_asnumpy_skill_gate(bundle, required_files=tuple(handoff["skill_files"]))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, DraftGenerationError) as exc:
        print("[Codex交接] VALIDATE 本地 handoff 校验失败：%s" % exc, file=sys.stderr)
        return 2

    def chat(system_prompt, user_prompt, **_options):
        match = re.search(r"TARGET PATH: `([^`]+)`", user_prompt)
        if match is None:
            raise DraftGenerationError("Codex response prompt has no target path")
        relative = match.group(1)
        if relative not in handoff["expected_overlay_paths"]:
            raise DraftGenerationError("Codex response target is not in handoff")
        response_path = response_root / (relative + ".response")
        try:
            response_path.relative_to(response_root)
        except ValueError as exc:
            raise DraftGenerationError("Codex response path escapes response-dir") from exc
        if response_path.is_symlink() or not response_path.is_file():
            raise DraftGenerationError("missing Codex response: %s" % relative)
        payload = response_path.read_bytes()
        if not payload or len(payload) > 2 * 1024 * 1024:
            raise DraftGenerationError("Codex response exceeds the size limit: %s" % relative)
        response = payload.decode("utf-8")
        generation = {
            "mode": "injected_test_double",
            "provider": "codex-self",
            "model": "codex-local",
            "protocol": "openai-chat-completions",
            "response_model_aliases": [],
        }
        return response, _injected_generation_receipt(
            system_prompt, user_prompt, response, generation
        )

    try:
        out_root.mkdir(parents=True, mode=0o700)
        provider = LlmDraftProvider(
            chat=chat,
            max_attempts=args.max_attempts,
            provider_label="codex-self",
            model_name="codex-local",
            skill_bundle=bundle,
            staged_direct=True,
        )
        report = provider.generate_asnumpy_adapter_candidate(
            out_root,
            operator_artifact=handoff["operator_artifact"],
            adapter_contract=handoff["adapter_contract"],
            asnumpy_context=handoff["context"],
            expected_paths=handoff["expected_overlay_paths"],
            adapter_context_plan=plan,
            skill_files=tuple(handoff["skill_files"]),
        )
    except DraftRepairError as exc:
        print("[Codex交接] VALIDATE 失败：%s" % exc, file=sys.stderr)
        print(
            "[Codex交接] provider=codex-self actual_remote_call=%s receipts=%d"
            % ("yes" if exc.actual_remote_call else "no", exc.receipt_count),
            file=sys.stderr,
        )
        # A missing or malformed local response can fail before a receipt is
        # created. Persist the failure receipt whenever the bounded attempt
        # chain actually ran, so operators can distinguish an empty response
        # directory from a preflight failure and retry deterministically.
        if exc.receipts or exc.failures or exc.request_count:
            failure_path = out_root / "asnumpy-adapter-validation-failure.json"
            try:
                out_root.mkdir(parents=True, mode=0o700, exist_ok=True)
                failure_payload = {
                    "schema_version": 1,
                    "status": "validation_failed",
                    "provider": "codex-self",
                    "model": "codex-local",
                    "generation_mode": "injected_test_double",
                    "actual_remote_call": False,
                    "attempt_count": exc.request_count,
                    "receipt_count": exc.receipt_count,
                    "response_sha256": list(exc.response_hashes),
                    "local_receipts": list(exc.receipts),
                    "expected_response_paths": [
                        relative + ".response"
                        for relative in handoff["expected_overlay_paths"]
                    ],
                    "response_dir": str(response_root),
                    "missing_response_paths": [
                        relative + ".response"
                        for relative in handoff["expected_overlay_paths"]
                        if (
                            (response_root / (relative + ".response")).is_symlink()
                            or not (response_root / (relative + ".response")).is_file()
                        )
                    ],
                    "failures": list(exc.failures),
                    "verified_new_adapter": False,
                    "gates": {
                        "generated": False, "built": False, "imported": False,
                        "executed": False, "qualified": False, "activated": False,
                    },
                }
                failure_path.write_text(
                    json.dumps(failure_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                os.chmod(failure_path, 0o600)
                print("[Codex交接] 失败回执=%s" % failure_path, file=sys.stderr)
            except (OSError, TypeError, ValueError) as receipt_error:
                print("[Codex交接] 失败回执写入失败：%s" % type(receipt_error).__name__, file=sys.stderr)
        return 2
    except (OSError, UnicodeError, ValueError, DraftGenerationError) as exc:
        print("[Codex交接] VALIDATE 失败：%s" % exc, file=sys.stderr)
        return 2
    print(
        "[Codex交接] WAITING_FOR_AGENT → VALIDATE → GENERATED_UNVERIFIED · "
        "provider=codex-self actual_remote_call=no · artifact=%s · out=%s"
        % (report["adapter_artifact_id"], out_root / "candidate")
    )
    print(
        "[Codex交接] 后续仍需 build/import/NumPy oracle/NPU/performance/qualification；"
        "当前不会自动激活"
    )
    return 0

def _cmd_generate_asnumpy_adapter(args):
    """Generate an architecture-aware AsNumpy overlay with the remote model."""

    from numpilot.asnumpy_adapter_context import (
        plan_asnumpy_adapter_context,
        plan_official_aclnn_adapter_context,
    )
    from numpilot.gaps.models import GapSpec
    from numpilot.gaps.official_adapter import (
        OfficialAdapterInputError,
        bind_official_adapter_context,
        load_operator_candidate_artifact,
        prepare_official_candidate_adapter_input,
    )
    from numpilot.gaps.official_resolution import (
        OfficialResolutionError,
        load_official_resolution_report,
    )
    from numpilot.gaps.official_candidate_handoff import (
        OfficialCandidateHandoffError,
        load_official_candidate_handoff,
    )
    from numpilot.gaps.adapter_recovery import (
        AdapterRecoveryError,
        ingest_official_adapter_candidate,
    )
    from numpilot.gaps.adapter_handoff import write_adapter_handoff
    from numpilot.gaps.pipeline import (
        GapCompletionError,
        load_script_gap_request,
        prepare_gap_completion,
    )
    from numpilot.gaps.provider import (
        DraftGenerationError,
        DraftRepairError,
        LlmDraftProvider,
        _load_existing_draft,
        _validate_adapter_context_plan,
    )
    from numpilot.gaps.recovery import (
        RecoveryError,
        RecoveryPhase,
        load_recovery_status,
    )

    def read_bounded(path, label, limit):
        target = Path(path)
        if target.is_symlink() or not target.is_file():
            raise DraftGenerationError("%s must be a regular file" % label)
        payload = target.read_bytes()
        if len(payload) > limit:
            raise DraftGenerationError("%s exceeds the size limit" % label)
        return payload.decode("utf-8")

    out_root = Path(args.out).absolute()
    if os.path.lexists(out_root):
        print("[AsNumpy 接入] 输出目录必须不存在", file=sys.stderr)
        return 2

    # Finish every inert, local validation before the readiness probe.  A bad
    # GapSpec, signed artifact, adapter contract, or AsNumpy layout must never
    # cause even a connectivity call to the configured model.
    try:
        if args.candidate:
            if args.spec or args.artifact or not args.recovery_state_dir:
                raise DraftGenerationError(
                    "official candidate mode requires --candidate and "
                    "--recovery-state-dir only"
                )
            recovery = load_recovery_status(args.recovery_state_dir)
            if recovery.state.phase is not RecoveryPhase.CANDIDATE_READY:
                raise DraftGenerationError(
                    "official adapter requires a CANDIDATE_READY recovery state"
                )
            if not (
                recovery.state.policy.model_allowed
                and recovery.state.policy.network_allowed
            ):
                raise DraftGenerationError(
                    "recovery authorization does not allow configured-model access"
                )
            candidate_path = Path(args.candidate).absolute()
            candidate = load_operator_candidate_artifact(candidate_path)
            gap_matches = tuple(
                gap for gap in recovery.state.gaps
                if gap.gap_id == candidate.gap_id
            )
            if len(gap_matches) != 1:
                raise DraftGenerationError(
                    "operator candidate gap is absent from recovery state"
                )
            candidate_refs = tuple(
                ref for ref in gap_matches[0].refs
                if ref.name == "operator_candidate"
            )
            candidate_payload = read_bounded(
                candidate_path, "operator candidate", 2 * 1024 * 1024
            ).encode("utf-8")
            if (
                len(candidate_refs) != 1
                or Path(candidate_refs[0].locator).absolute() != candidate_path
                or hashlib.sha256(candidate_payload).hexdigest()
                != candidate_refs[0].sha256
            ):
                raise DraftGenerationError(
                    "operator candidate differs from its recovery receipt"
                )

            if args.official_context_gate:
                gate_path = Path(args.official_context_gate).absolute()
                gate_payload = json.loads(read_bounded(
                    gate_path, "official context gate", 8 * 1024 * 1024
                ))
                gate_refs = tuple(
                    ref for ref in recovery.state.refs
                    if ref.name == "official_context_gate"
                )
                if len(gate_refs) != 1 or Path(gate_refs[0].locator).absolute() != gate_path:
                    raise DraftGenerationError(
                        "official context gate is not bound to recovery state"
                    )
                if hashlib.sha256(gate_path.read_bytes()).hexdigest() != gate_refs[0].sha256:
                    raise DraftGenerationError("official context gate content drift")
                if (
                    gate_payload.get("schema") != "numpilot.official-adapter-context-gate.v1"
                    or gate_payload.get("status") != "ready_for_official_adapter_model"
                    or gate_payload.get("next_action") != "invoke_configured_model_for_official_adapter"
                    or any(gate_payload.get(name) is not False for name in (
                        "model_invocation_performed", "network_access_performed",
                        "system_write_performed", "asnumpy_import_performed",
                        "build_performed", "execution_performed", "activation_performed",
                    ))
                ):
                    raise DraftGenerationError("official context gate is not inert and model-ready")
                ref_by_prefix = {}
                for prefix in ("official_handoff_", "official_preflight_", "official_resolution_"):
                    matches = tuple(ref for ref in gap_matches[0].refs if ref.name.startswith(prefix))
                    if len(matches) != 1:
                        raise DraftGenerationError("official recovery state lacks %s binding" % prefix)
                    ref_by_prefix[prefix] = matches[0]
                handoff_path = Path(ref_by_prefix["official_handoff_"].locator).absolute()
                preflight_path = Path(ref_by_prefix["official_preflight_"].locator).absolute()
                official_path = Path(ref_by_prefix["official_resolution_"].locator).absolute()
                for ref, path, label, limit in (
                    (ref_by_prefix["official_handoff_"], handoff_path, "official hand-off", 2 * 1024 * 1024),
                    (ref_by_prefix["official_preflight_"], preflight_path, "official preflight", 8 * 1024 * 1024),
                    (ref_by_prefix["official_resolution_"], official_path, "official resolution", 8 * 1024 * 1024),
                ):
                    payload = read_bounded(path, label, limit).encode("utf-8")
                    if hashlib.sha256(payload).hexdigest() != ref.sha256:
                        raise DraftGenerationError("%s content drift" % label)
                handoff = load_official_candidate_handoff(
                    str(handoff_path), recovery.state.source.path, str(official_path)
                )
                official_report = load_official_resolution_report(str(official_path))
                preparation = prepare_gap_completion(
                    recovery.state.source.path,
                    target_soc=recovery.state.target.target_soc,
                    cann_version=recovery.state.target.cann_version,
                )
                official_report.validate_preparation(preparation)
                specs = tuple(item for item in preparation.specs if item.gap_id == candidate.gap_id)
                decisions = tuple(item for item in official_report.decisions if item.gap_id == candidate.gap_id)
                if len(specs) != 1 or len(decisions) != 1:
                    raise DraftGenerationError("official recovery does not bind one candidate GapSpec and decision")
                spec = specs[0]
                prepared = prepare_official_candidate_adapter_input(candidate, spec, decisions[0])
                if gate_payload.get("candidate_id") != candidate.candidate_id or gate_payload.get("qualified_api") != spec.qualified_api:
                    raise DraftGenerationError("official context gate candidate/API mismatch")
                api_name = spec.api_name
                selected_library = prepared.selected_library
                context_plan = plan_official_aclnn_adapter_context(
                    args.asnumpy_reference, api_name,
                    selected_library_relative_path=selected_library["relative_path"],
                    selected_library_sha256=selected_library["sha256"],
                )
                prepared = bind_official_adapter_context(prepared, context_plan, api_name=api_name)
                prepared.require_model_ready()
                if (
                    gate_payload.get("context_plan_sha256") != context_plan.plan_sha256
                    or gate_payload.get("adapter_contract_sha256") != prepared.adapter_contract_sha256
                ):
                    raise DraftGenerationError("official context gate plan/contract hash mismatch")
                artifact_manifest = prepared.operator_manifest
                adapter_contract = prepared.adapter_contract
                context = dict(context_plan.context)
                header_refs = tuple(ref for ref in candidate.candidate_refs if ref.name == "official_selected_header")
                if len(header_refs) != 1:
                    raise DraftGenerationError("official candidate lacks one selected header")
                selected_header = decisions[0].installed_cann.get("selected_header")
                if not isinstance(selected_header, dict) or not isinstance(selected_header.get("relative_path"), str):
                    raise DraftGenerationError("official resolution lacks selected header path")
                header_path = Path(official_report.cann_root) / selected_header["relative_path"]
                header_content = read_bounded(header_path, "official candidate header", 96 * 1024)
                if hashlib.sha256(header_content.encode("utf-8")).hexdigest() != header_refs[0].sha256:
                    raise DraftGenerationError("official candidate header hash mismatch")
                context["SIGNED_OFFICIAL_HEADER/%s" % header_path.name] = header_content
                adapter_skill_files = ASNUMPY_OFFICIAL_ADAPTER_SKILL_FILES
            else:
                gap_request_refs = tuple(
                    ref for ref in recovery.state.refs if ref.name == "gap_request"
                )
                official_refs = tuple(
                    ref for ref in recovery.state.refs
                    if ref.name == "official_resolution"
                )
                if len(gap_request_refs) != 1 or len(official_refs) != 1:
                    raise DraftGenerationError(
                        "recovery state lacks bound gap/official receipts"
                    )
                gap_request_path = Path(gap_request_refs[0].locator)
                official_path = Path(official_refs[0].locator)
                gap_request_payload = read_bounded(
                    gap_request_path, "gap request", 2 * 1024 * 1024
                ).encode("utf-8")
                official_payload = read_bounded(
                    official_path, "official resolution", 8 * 1024 * 1024
                ).encode("utf-8")
                if (
                    hashlib.sha256(gap_request_payload).hexdigest()
                    != gap_request_refs[0].sha256
                    or hashlib.sha256(official_payload).hexdigest()
                    != official_refs[0].sha256
                ):
                    raise DraftGenerationError(
                        "recovery gap/official receipt content drift"
                    )
                selection = load_script_gap_request(
                    str(gap_request_path),
                    recovery.state.source.path,
                    target_soc=recovery.state.target.target_soc,
                    cann_version=recovery.state.target.cann_version,
                )
                specs = tuple(
                    item for item in selection.specs
                    if item.gap_id == candidate.gap_id
                )
                if len(specs) != 1:
                    raise DraftGenerationError(
                        "gap request does not contain one candidate GapSpec"
                    )
                spec = specs[0]
                official_report = load_official_resolution_report(
                    str(official_path)
                )
                decisions = tuple(
                    item for item in official_report.decisions
                    if item.gap_id == candidate.gap_id
                )
                if len(decisions) != 1:
                    raise DraftGenerationError(
                        "official report does not contain one candidate decision"
                    )
                prepared = prepare_official_candidate_adapter_input(
                    candidate, spec, decisions[0]
                )
                api_name = spec.api_name
                selected_library = prepared.selected_library
                context_plan = plan_official_aclnn_adapter_context(
                    args.asnumpy_reference,
                    api_name,
                    selected_library_relative_path=selected_library[
                        "relative_path"
                    ],
                    selected_library_sha256=selected_library["sha256"],
                )
                prepared = bind_official_adapter_context(
                    prepared, context_plan, api_name=api_name
                )
                prepared.require_model_ready()
                artifact_manifest = prepared.operator_manifest
                adapter_contract = prepared.adapter_contract
                context = dict(context_plan.context)
                header_refs = tuple(
                    ref for ref in candidate.candidate_refs
                    if ref.name == "official_selected_header"
                )
                if len(header_refs) != 1:
                    raise DraftGenerationError(
                        "official candidate lacks one selected header"
                    )
                header_content = read_bounded(
                    header_refs[0].locator,
                    "official candidate header",
                    96 * 1024,
                )
                if hashlib.sha256(header_content.encode("utf-8")).hexdigest() != (
                    header_refs[0].sha256
                ):
                    raise DraftGenerationError(
                        "official candidate header hash mismatch"
                    )
                context[
                    "SIGNED_OFFICIAL_HEADER/%s"
                    % Path(header_refs[0].locator).name
                ] = header_content
                adapter_skill_files = ASNUMPY_OFFICIAL_ADAPTER_SKILL_FILES
        else:
            if not args.spec or not args.artifact or args.recovery_state_dir:
                raise DraftGenerationError(
                    "custom operator mode requires --spec and --artifact"
                )
            spec_data = json.loads(read_bounded(
                args.spec, "gap spec", 2 * 1024 * 1024
            ))
            if not isinstance(spec_data, dict):
                raise DraftGenerationError("gap spec must be a JSON object")
            spec = GapSpec.from_dict(spec_data)
            signed = _load_existing_draft(Path(args.artifact), spec)
            artifact_manifest = json.loads(read_bounded(
                Path(args.artifact) / "operator-artifact.json",
                "operator artifact manifest",
                2 * 1024 * 1024,
            ))
            adapter_paths = sorted(
                path for path in signed.files
                if path.startswith("asnumpy_adapter/") and path.endswith(".json")
            )
            if len(adapter_paths) != 1:
                raise DraftGenerationError(
                    "signed artifact must contain exactly one AsNumpy adapter JSON"
                )
            adapter_path = Path(args.artifact) / adapter_paths[0]
            adapter_contract = read_bounded(
                adapter_path, "AsNumpy adapter contract", 256 * 1024
            )
            if hashlib.sha256(adapter_contract.encode("utf-8")).hexdigest() != (
                signed.files[adapter_paths[0]]
            ):
                raise DraftGenerationError("AsNumpy adapter contract hash mismatch")
            adapter_data = json.loads(adapter_contract)
            api_name = _resolve_asnumpy_adapter_api_name(adapter_data, spec)
            if api_name is None:
                raise DraftGenerationError("AsNumpy adapter API identity mismatch")
            context_plan = plan_asnumpy_adapter_context(
                args.asnumpy_reference, api_name
            )
            context = dict(context_plan.context)
            operator_reference_paths = (
                "CMakeLists.txt",
                "op_api/aclnn_%s.h" % api_name,
                "op_api/%s.cpp" % api_name,
            )
            for relative in operator_reference_paths:
                if relative not in signed.files:
                    continue
                content = read_bounded(
                    Path(args.artifact) / relative,
                    "signed operator reference %s" % relative,
                    96 * 1024,
                )
                if hashlib.sha256(content.encode("utf-8")).hexdigest() != (
                    signed.files[relative]
                ):
                    raise DraftGenerationError(
                        "signed operator reference hash mismatch: %s" % relative
                    )
                context["SIGNED_OPERATOR/%s" % relative] = content
            adapter_skill_files = ASNUMPY_INTEGRATION_SKILL_FILES
        # Validate the provider-facing manifest before the readiness probe so
        # an internal schema mismatch cannot cause a remote call.
        validated_context_manifest = _validate_adapter_context_plan(
            context_plan.manifest()
        )
        if getattr(args, "handoff_only", False):
            bundle = load_asnumpy_integration_skill(
                Path(args.asnumpy_skill_root) if args.asnumpy_skill_root else None
            )
            _run_asnumpy_skill_gate(bundle, required_files=adapter_skill_files)
            metadata = write_adapter_handoff(
                out_root,
                operator_artifact=artifact_manifest,
                adapter_contract=adapter_contract,
                asnumpy_context=context,
                adapter_context_plan=validated_context_manifest,
                skill_files=tuple(adapter_skill_files),
                skill_bundle_sha256=bundle.bundle_sha256,
            )
            print(
                "[Codex交接] PREPARE → WAITING_FOR_AGENT · handoff=%s · "
                "paths=%d · plan_sha256=%s"
                % (out_root, len(metadata["expected_overlay_paths"]),
                   validated_context_manifest["plan_sha256"])
            )
            print(
                "[Codex交接] 不读取 llm.json、不联网、不修改 AsNumpy；"
                "请让外部代理写入 responses/<path>.response"
            )
            return 0
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError,
            DraftGenerationError, GapCompletionError, OfficialAdapterInputError,
            OfficialResolutionError, OfficialCandidateHandoffError, RecoveryError) as exc:
        print("[AsNumpy 接入] 本地输入校验失败: %s" % exc, file=sys.stderr)
        print(
            "[LLM调用] actual_remote_call=no · 未读取 Skill，未生成 overlay",
            file=sys.stderr,
        )
        return 2

    raw_timeout = (
        str(args.llm_timeout)
        if args.llm_timeout is not None
        else os.getenv("NUMPILOT_LLM_TIMEOUT", "120")
    )
    try:
        timeout_seconds = validate_llm_timeout(raw_timeout)
    except LlmConfigError as exc:
        print("[接入门禁 1/2] %s" % exc, file=sys.stderr)
        return 2
    os.environ["NUMPILOT_LLM_TIMEOUT"] = str(timeout_seconds)
    configuration_status = _load_llm_or_report()
    if configuration_status is None:
        print("[接入门禁 1/2] fail · LLM 本地配置不可用", file=sys.stderr)
        return 4
    if _llm_proxy_for_remote_call("接入门禁 1/2") is None:
        return 4
    try:
        llm_gate = _run_llm_readiness_gate(
            configuration_status, require_marker=True
        )
    except (LlmHttpError, LlmTransportError, LlmResponseError) as exc:
        print(
            "[接入门禁 1/2] fail · 简单问答没有取得可验证响应: %s" % exc,
            file=sys.stderr,
        )
        print(
            "[AsNumpy 接入生成] actual_remote_call=no · 未读取 Skill，未生成 overlay",
            file=sys.stderr,
        )
        return 4
    probe_receipt = llm_gate["receipt"]
    print(
        "[接入门禁 1/2] pass · LLM 简单问答 HTTP %s · model=%s · timeout=%ss"
        % (
            probe_receipt.get("http_status"),
            probe_receipt.get("response_model") or configuration_status.model,
            timeout_seconds,
        )
    )

    bundle = load_asnumpy_integration_skill(
        Path(args.asnumpy_skill_root) if args.asnumpy_skill_root else None
    )
    skill_gate = _run_asnumpy_skill_gate(
        bundle, required_files=adapter_skill_files
    )
    print(
        "[接入门禁 2/2] pass · 已读取 NumPilot AsNumpy 案例 Skill · "
        "selected_files=%s · verified_bundle_files=%s · bundle_sha256=%s"
        % (
            len(adapter_skill_files),
            len(bundle.files),
            bundle.bundle_sha256,
        )
    )
    staging_root = None
    adapter_recovery_result = None
    try:
        staging_root = _create_output_stage(out_root, "asnumpy-generate")
        _write_generation_preflight(
            staging_root,
            llm_gate,
            skill_gate,
            timeout_seconds,
            adapter_context_plan=validated_context_manifest,
        )
        print(
            "[AsNumpy 架构] architecture=%s mode=%s context_files=%d "
            "overlay_files=%d plan_sha256=%s"
            % (
                context_plan.architecture,
                context_plan.mode,
                len(context_plan.context),
                len(context_plan.expected_overlay_paths),
                context_plan.plan_sha256,
            )
        )
        provider = LlmDraftProvider(
            max_attempts=args.max_attempts,
            configuration_status=configuration_status,
            skill_bundle=bundle,
            staged_direct=True,
        )
        report = provider.generate_asnumpy_adapter_candidate(
            staging_root,
            operator_artifact=artifact_manifest,
            adapter_contract=adapter_contract,
            asnumpy_context=context,
            expected_paths=context_plan.expected_overlay_paths,
            adapter_context_plan=validated_context_manifest,
            skill_files=adapter_skill_files,
        )
        _publish_output_stage(staging_root, out_root, "AsNumpy adapter")
        staging_root = None
        if args.candidate:
            try:
                adapter_recovery_result = ingest_official_adapter_candidate(
                    args.recovery_state_dir,
                    candidate_path,
                    out_root,
                )
            except AdapterRecoveryError as exc:
                print(
                    "[AsNumpy 接入回执] overlay 已安全发布，但恢复状态未推进: "
                    "%s: %s" % (exc.code, exc),
                    file=sys.stderr,
                )
                print(
                    "[下一步] 使用 recover-gaps ingest-adapter 对已发布产物重放回执",
                    file=sys.stderr,
                )
                return 2
    except DraftRepairError as exc:
        print("[AsNumpy 接入生成] 失败: %s" % exc, file=sys.stderr)
        print(
            "[LLM调用] actual_remote_call=%s verified_new_adapter=no receipts=%d"
            % ("yes" if exc.actual_remote_call else "no", exc.receipt_count),
            file=sys.stderr,
        )
        if exc.actual_remote_call and exc.receipts:
            failure_payload = {
                "schema_version": 1,
                "status": "generation_failed",
                "source_artifact_id": artifact_manifest.get("artifact_id"),
                "model": exc.generation.get("model"),
                "provider": exc.generation.get("provider"),
                "protocol": exc.generation.get("protocol"),
                "configuration_source": exc.generation.get("configuration_source"),
                "attempt_count": exc.request_count,
                "receipt_count": exc.receipt_count,
                "response_sha256": list(exc.response_hashes),
                "remote_receipts": list(exc.receipts),
                "failures": list(exc.failures),
                "failure_message": str(exc),
                "actual_remote_call": True,
                "verified_new_adapter": False,
                "adapter_context_plan": validated_context_manifest,
                "gates": {
                    "generated": False,
                    "built": False,
                    "imported": False,
                    "executed": False,
                    "qualified": False,
                    "activated": False,
                },
            }
            try:
                out_root.mkdir(parents=True, mode=0o700, exist_ok=False)
                failure_path = out_root / "asnumpy-adapter-generation-failure.json"
                encoded = (json.dumps(failure_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
                fd = os.open(str(failure_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                try:
                    with os.fdopen(fd, "wb") as handle:
                        fd = -1
                        handle.write(encoded)
                        handle.flush()
                        os.fsync(handle.fileno())
                finally:
                    if fd >= 0:
                        os.close(fd)
                print("[失败回执] 已封存=%s" % failure_path, file=sys.stderr)
            except (OSError, TypeError, ValueError) as receipt_error:
                print("[失败回执] 写入失败: %s" % receipt_error, file=sys.stderr)
        return 2
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError,
            DraftGenerationError) as exc:
        print("[AsNumpy 接入生成] 失败: %s" % exc, file=sys.stderr)
        return 2
    finally:
        if staging_root is not None:
            shutil.rmtree(staging_root, ignore_errors=True)

    print(
        "[AsNumpy 接入生成] status=%s adapter_artifact_id=%s out=%s"
        % (report["status"], report["adapter_artifact_id"], out_root / "candidate")
    )
    print(
        "[LLM调用] actual_remote_call=yes verified_new_adapter=yes "
        "attempts=%d receipts=%d content_hash_bound=yes"
        % (report["attempt_count"], len(report["remote_receipts"]))
    )
    if adapter_recovery_result is not None:
        print(
            "[恢复状态] phase=%s appended=%d · integrated/qualified/activated=no"
            % (
                adapter_recovery_result.recovery.state.phase.value,
                adapter_recovery_result.appended_revisions,
            )
        )
    return 0


def _cmd_repair_asnumpy_adapter(args):
    """Return a raw AsNumpy diagnostic to the model for one-file repair."""

    from numpilot.asnumpy_adapter_context import (
        OFFICIAL_ACLNN_ARCHITECTURE,
        infer_asnumpy_adapter_api,
        infer_official_adapter_api,
        plan_asnumpy_adapter_context,
        plan_official_aclnn_adapter_context,
    )
    from numpilot.gaps.provider import (
        DraftGenerationError,
        DraftRepairError,
        LlmDraftProvider,
        _load_asnumpy_adapter_report,
    )

    def read_bounded(path, label, limit):
        target = Path(path)
        if target.is_symlink() or not target.is_file():
            raise DraftGenerationError("%s must be a regular file" % label)
        payload = target.read_bytes()
        if len(payload) > limit:
            raise DraftGenerationError("%s exceeds the size limit" % label)
        return payload.decode("utf-8")

    out_root = Path(args.out).absolute()
    if os.path.lexists(out_root):
        print("[AsNumpy 接入修复] 输出目录必须不存在", file=sys.stderr)
        return 2
    try:
        diagnostic = read_bounded(
            args.diagnostic_file, "AsNumpy repair diagnostic", 64 * 1024
        )
        _base_report, _base_payload, base_hashes = (
            _load_asnumpy_adapter_report(Path(args.base_adapter).absolute())
        )
        if args.path not in base_hashes:
            raise DraftGenerationError(
                "repair path is not part of the signed AsNumpy adapter"
            )
        if _base_report.get("schema_version") != 4:
            raise DraftGenerationError(
                "base adapter predates AsNumpy context-plan binding; regenerate it"
            )
        base_plan_manifest = _base_report.get("asnumpy_adapter_context_plan")
        if not isinstance(base_plan_manifest, dict):
            raise DraftGenerationError(
                "base adapter does not bind an AsNumpy context plan"
            )
        if base_plan_manifest.get("architecture") == OFFICIAL_ACLNN_ARCHITECTURE:
            api_name = infer_official_adapter_api(tuple(base_hashes))
            binding = base_plan_manifest.get("official_library_binding")
            if not isinstance(binding, dict):
                raise DraftGenerationError(
                    "official adapter does not bind an installed ACLNN library"
                )
            selected_library_relative_path = binding.get(
                "selected_library_relative_path"
            )
            selected_library_sha256 = binding.get("selected_library_sha256")
            if not isinstance(
                selected_library_relative_path, str
            ) or not isinstance(selected_library_sha256, str):
                raise DraftGenerationError(
                    "official adapter library binding is invalid"
                )
            context_plan = plan_official_aclnn_adapter_context(
                args.asnumpy_reference,
                api_name,
                selected_library_relative_path=selected_library_relative_path,
                selected_library_sha256=selected_library_sha256,
            )
        else:
            api_name = infer_asnumpy_adapter_api(tuple(base_hashes))
            context_plan = plan_asnumpy_adapter_context(
                args.asnumpy_reference, api_name
            )
        context = dict(context_plan.context)
        context_manifest = context_plan.manifest()
        if base_plan_manifest != context_manifest:
            raise DraftGenerationError(
                "AsNumpy reference layout or content changed; regenerate the adapter"
            )
        if set(base_hashes) != set(context_plan.expected_overlay_paths):
            raise DraftGenerationError(
                "base adapter overlay paths differ from the current AsNumpy layout"
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError,
            DraftGenerationError) as exc:
        print("[AsNumpy 接入修复] 本地输入校验失败: %s" % exc, file=sys.stderr)
        print(
            "[LLM调用] actual_remote_call=no · 未读取 Skill，未开始修复",
            file=sys.stderr,
        )
        return 2
    raw_timeout = (
        str(args.llm_timeout)
        if args.llm_timeout is not None
        else os.getenv("NUMPILOT_LLM_TIMEOUT", "120")
    )
    try:
        timeout_seconds = validate_llm_timeout(raw_timeout)
    except LlmConfigError as exc:
        print("[接入修复门禁 1/2] %s" % exc, file=sys.stderr)
        return 2
    os.environ["NUMPILOT_LLM_TIMEOUT"] = str(timeout_seconds)
    configuration_status = _load_llm_or_report()
    if configuration_status is None:
        print("[接入修复门禁 1/2] fail · LLM 本地配置不可用", file=sys.stderr)
        return 4
    if _llm_proxy_for_remote_call("接入修复门禁 1/2") is None:
        return 4
    try:
        llm_gate = _run_llm_readiness_gate(
            configuration_status, require_marker=True
        )
    except (LlmHttpError, LlmTransportError, LlmResponseError) as exc:
        print(
            "[接入修复门禁 1/2] fail · 简单问答没有取得可验证响应: %s" % exc,
            file=sys.stderr,
        )
        print(
            "[AsNumpy 接入修复] actual_remote_call=no · 未读取 Skill，未开始修复",
            file=sys.stderr,
        )
        return 4
    receipt = llm_gate["receipt"]
    print(
        "[接入修复门禁 1/2] pass · LLM 简单问答 HTTP %s · model=%s · "
        "timeout=%ss"
        % (
            receipt.get("http_status"),
            receipt.get("response_model") or configuration_status.model,
            timeout_seconds,
        )
    )

    profile_files = ASNUMPY_INTEGRATION_SKILL_FILES
    bundle = load_asnumpy_integration_skill(
        Path(args.asnumpy_skill_root) if args.asnumpy_skill_root else None
    )
    skill_gate = _run_asnumpy_skill_gate(
        bundle, required_files=profile_files
    )
    print(
        "[接入修复门禁 2/2] pass · 已读取 NumPilot AsNumpy 案例 Skill · "
        "profile=%s · files=%s · bundle_sha256=%s"
        % (args.repair_profile, len(bundle.files), bundle.bundle_sha256)
    )
    staging_root = None
    try:
        staging_root = _create_output_stage(out_root, "asnumpy-repair")
        _write_generation_preflight(
            staging_root,
            llm_gate,
            skill_gate,
            timeout_seconds,
            adapter_context_plan=context_plan.manifest(),
        )
        print(
            "[AsNumpy 架构] architecture=%s mode=%s context_files=%d "
            "plan_sha256=%s"
            % (
                context_plan.architecture,
                context_plan.mode,
                len(context_plan.context),
                context_plan.plan_sha256,
            )
        )
        provider = LlmDraftProvider(
            max_attempts=args.max_attempts,
            configuration_status=configuration_status,
            skill_bundle=bundle,
            staged_direct=True,
        )
        report = provider.repair_asnumpy_adapter_candidate(
            staging_root,
            base_adapter_root=Path(args.base_adapter),
            diagnostic=diagnostic,
            path=args.path,
            asnumpy_context=context,
            adapter_context_plan=context_manifest,
            skill_files=profile_files,
            repair_profile=args.repair_profile,
        )
        _publish_output_stage(staging_root, out_root, "AsNumpy adapter repair")
        staging_root = None
    except DraftRepairError as exc:
        print("[AsNumpy 接入修复] 失败: %s" % exc, file=sys.stderr)
        print(
            "[LLM调用] actual_remote_call=%s verified_new_adapter_repair=no "
            "receipts=%d"
            % ("yes" if exc.actual_remote_call else "no", exc.receipt_count),
            file=sys.stderr,
        )
        return 2
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError,
            DraftGenerationError) as exc:
        print("[AsNumpy 接入修复] 失败: %s" % exc, file=sys.stderr)
        return 2
    finally:
        if staging_root is not None:
            shutil.rmtree(staging_root, ignore_errors=True)
    print(
        "[AsNumpy 接入修复] status=%s adapter_artifact_id=%s "
        "parent_adapter_artifact_id=%s path=%s out=%s"
        % (
            report["status"],
            report["adapter_artifact_id"],
            report["parent_adapter_artifact_id"],
            args.path,
            out_root / "candidate",
        )
    )
    print(
        "[LLM调用] actual_remote_call=yes verified_new_adapter_repair=yes "
        "attempts=%d receipts=%d content_hash_bound=yes"
        % (report["attempt_count"], len(report["remote_receipts"]))
    )
    return 0


def _preflight(backend_name, skip):
    """migrate 前置自检。npu 后端缺硬依赖 → 拦截退出;shim 仅软告警。"""
    if skip:
        print("[env-check] 已通过 --skip-env-check 跳过环境自检")
        return
    with terminal_status(
        "执行迁移前环境自检 · backend=%s" % backend_name,
    ) as progress:
        rep = Env.check(backend=backend_name)
        if rep.errors:
            progress.error()
        elif rep.warnings:
            progress.warning()
    if rep.errors:
        print(rep.render(), file=sys.stderr)
        print(
            "\n[env-check] 环境不满足 backend=%s 的硬依赖,已拦截。\n"
            "  修复以上 ✗ 项后重试,或加 --skip-env-check 强制继续(后果自负)。"
            % backend_name,
            file=sys.stderr,
        )
        sys.exit(3)
    if rep.warnings:
        print("[env-check] %d 个警告(不阻断):%s"
              % (len(rep.warnings), "; ".join(w.name for w in rep.warnings)))


def _cmd_migrate(args):
    from numpilot.script_migration import (
        ScriptMigrationError,
        build_gap_request,
        classify_script,
        detect_source_mode,
        generate_script_with_configured_model,
        generate_qualified_script_with_configured_model,
        read_script_source,
        scan_numpy_script,
        write_script_artifacts,
    )

    source = read_script_source(args.file)
    # Demo v1 has one unsurprising product entry: an ordinary NumPy file.  The
    # legacy build_inputs/forward contract remains available only when the
    # caller explicitly asks for ``--mode contract``.  ``auto`` is retained as
    # a compatibility switch for older automation, but it is no longer the
    # default because innocent function names must not silently change the
    # migration product.
    mode = detect_source_mode(source) if args.mode == "auto" else args.mode
    if mode == "script" and args.backend is None and not os.getenv(
        "NUMPILOT_BACKEND"
    ):
        # The ordinary user lane must inspect the actually installed AsNumpy,
        # not silently classify everything through the offline NumPy shim.
        backend_name = "npu"
    else:
        backend_name = _resolve_backend_name(args)
    backend, analysis_profiles = _qualified_runtime_inputs(args, backend_name)
    _preflight(backend_name, args.skip_env_check)

    if mode == "script":
        if analysis_profiles is not None:
            print(
                "[普通脚本] --analysis-profiles 只用于 build_inputs/forward 严格模式",
                file=sys.stderr,
            )
            return 2
        if args.engine == "ast":
            print(
                "[普通脚本] 迁移代码必须由已配置模型生成，不能使用 --engine ast",
                file=sys.stderr,
            )
            return 2
        if args.out:
            out_dir = args.out
        else:
            default_root = default_demo_dir()
            default_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            out_dir = tempfile.mkdtemp(
                prefix=_net_name(args.file) + "-",
                dir=str(default_root),
            )
        qualified_analysis = None
        overlay_admission = None
        overlay_admission_provider = None
        runtime_identity_provider = getattr(backend, "runtime_identity", None)
        if callable(runtime_identity_provider):
            from numpilot.qualified_analysis import (
                QualifiedAnalysisError,
                analyze_qualified_script,
            )

            try:
                initial_runtime_identity = runtime_identity_provider()
                overlay_admission_provider = getattr(
                    backend, "qualified_overlay_admission", None
                )
                if (
                    callable(overlay_admission_provider)
                    and getattr(backend, "_overlay_config", None) is not None
                ):
                    overlay_admission = overlay_admission_provider(
                        initial_runtime_identity
                    )
                qualified_analysis = analyze_qualified_script(
                    source,
                    str(Path(args.file).absolute()),
                    runtime_identity=initial_runtime_identity,
                    overlay_admission=overlay_admission,
                )
            except QualifiedAnalysisError as exc:
                raise ScriptMigrationError(
                    "qualified source analysis failed: %s" % exc
                ) from exc
            except Exception as exc:
                raise ScriptMigrationError(
                    "runtime identity collection failed: %s"
                    % type(exc).__name__
                ) from exc
        scan = scan_numpy_script(
            source,
            str(Path(args.file).absolute()),
            qualified_analysis=qualified_analysis,
        )
        analysis = classify_script(
            scan,
            backend,
            qualified_analysis=qualified_analysis,
        )

        def write_current_script_result(*, status, fallback_accepted, **values):
            return write_script_artifacts(
                out_dir,
                analysis,
                status=status,
                fallback_accepted=fallback_accepted,
                qualified_analysis=qualified_analysis,
                **values,
            )

        print(
            "── NumPilot 普通脚本迁移 · %s ──" % _net_name(args.file)
        )
        print(
            "[扫描] mode=script · NumPy 使用=%d · strict=%d · scoped=%d · "
            "fallback=%d · audit_required=%d · unsupported=%d"
            % (
                len(scan.uses),
                len(analysis.strict),
                len(analysis.scoped),
                len(analysis.fallbacks),
                len(analysis.audit_required),
                len(analysis.unsupported),
            )
        )
        print(
            "[能力] backend=%s version=%s audit=%s"
            % (
                analysis.backend,
                analysis.backend_version or "unknown",
                analysis.audit_id,
            )
        )
        if qualified_analysis is not None:
            print(
                "[资格] decision=%s runtime=%s identity=%s"
                % (
                    qualified_analysis.decision.value,
                    qualified_analysis.runtime_identity_status,
                    qualified_analysis.runtime_identity_sha256 or "unavailable",
                )
            )
            overlay_qualification = qualified_analysis.overlay_qualification
            if overlay_qualification is not None:
                print(
                    "[资格] overlay=%s admitted_audit=%s entries=%s"
                    % (
                        overlay_qualification["qualification_id"],
                        overlay_qualification["admitted_audit_id"],
                        ",".join(overlay_qualification["entry_keys"]),
                    )
                )
        for decision in analysis.decisions:
            print(
                "[%s] %s · provenance=%s"
                % (decision.status, decision.qualified_api, decision.provenance)
            )

        if not scan.ok:
            for issue in scan.issues:
                print("[普通脚本阻断] " + issue, file=sys.stderr)
            paths = write_current_script_result(
                status="blocked",
                fallback_accepted=False,
            )
            print("[报告] %s" % paths["report"])
            return 2

        if qualified_analysis is not None and qualified_analysis.decision.value not in {
            "DIRECT",
            "FALLBACK_CONSENT",
        }:
            qualified_decision = qualified_analysis.decision.value
            status = {
                "BLOCKED": "blocked",
                "CAPABILITY_AUDIT_REQUIRED": "capability_audit_required",
                "GAP_REQUIRED": "gap_generation_required",
            }[qualified_decision]
            for issue in qualified_analysis.issues:
                print("[资格阻断] " + issue, file=sys.stderr)
            for call in qualified_analysis.calls:
                if call.decision.value != "DIRECT":
                    print(
                        "[资格阻断] %s · %s · %s"
                        % (
                            call.source_api,
                            call.decision.value,
                            call.reason_code,
                        ),
                        file=sys.stderr,
                    )
            if qualified_decision == "GAP_REQUIRED":
                gap_request = build_gap_request(
                    analysis,
                    analysis.unsupported,
                    target_soc=args.target_soc,
                    cann_version=args.cann_version,
                )
                paths = write_current_script_result(
                    status="gap_generation_required",
                    fallback_accepted=False,
                    gap_request=gap_request,
                )
                print(
                    "[缺口] 当前不会调用迁移模型；先处理 unsupported，再重新扫描。"
                )
                print("[缺口] request=%s" % paths["gap_request"])
                if gap_request["gap_specs"]:
                    print("[下一步] " + shlex.join([
                        sys.executable,
                        "-m",
                        "numpilot",
                        "complete-gaps",
                        str(Path(args.file).absolute()),
                        "--gap-request",
                        paths["gap_request"],
                        "--out",
                        str(Path(out_dir).absolute() / "operator-gaps"),
                        "--target-soc",
                        args.target_soc,
                        "--cann-version",
                        args.cann_version,
                        "--provider",
                        "prepare",
                    ]))
                print("[报告] %s" % paths["report"])
                return 1
            paths = write_current_script_result(
                status=status,
                fallback_accepted=False,
            )
            print(
                "[资格] 当前判定禁止调用迁移模型；先完成所列改写或审计。",
                file=sys.stderr,
            )
            print("[报告] %s" % paths["report"])
            return 2 if status == "blocked" else 1

        if analysis.audit_required:
            for decision in analysis.audit_required:
                print(
                    "[能力审计待办] %s · %s"
                    % (decision.qualified_api, decision.evidence),
                    file=sys.stderr,
                )
            paths = write_current_script_result(
                status="capability_audit_required",
                fallback_accepted=False,
            )
            print(
                "[能力审计] API 已存在但来源/适用范围尚未完成版本绑定审计；"
                "不会把它误当成缺失算子，也不会调用迁移模型。",
                file=sys.stderr,
            )
            print("[报告] %s" % paths["report"])
            return 1

        if analysis.unsupported:
            gap_request = build_gap_request(
                analysis,
                analysis.unsupported,
                target_soc=args.target_soc,
                cann_version=args.cann_version,
            )
            status = gap_request["status"]
            paths = write_current_script_result(
                status=status,
                fallback_accepted=False,
                gap_request=gap_request,
            )
            print(
                "[缺口] 当前不会调用迁移模型；先处理 unsupported，再重新扫描。"
            )
            print("[缺口] request=%s" % paths["gap_request"])
            if gap_request["gap_specs"]:
                print("[下一步] " + shlex.join([
                    sys.executable,
                    "-m",
                    "numpilot",
                    "complete-gaps",
                    str(Path(args.file).absolute()),
                    "--gap-request",
                    paths["gap_request"],
                    "--out",
                    str(Path(out_dir).absolute() / "operator-gaps"),
                    "--target-soc",
                    args.target_soc,
                    "--cann-version",
                    args.cann_version,
                    "--provider",
                    "prepare",
                ]))
            print("[报告] %s" % paths["report"])
            return 1

        fallback_accepted = False
        if analysis.fallbacks:
            policy = args.fallback_policy
            if policy == "ask":
                print("[fallback] 发现以下非 strict-NPU 路径:")
                for decision in analysis.fallbacks:
                    print(
                        "  - %s: %s · %s"
                        % (
                            decision.qualified_api,
                            decision.provenance,
                            decision.evidence,
                        )
                    )
                if not sys.stdin.isatty():
                    paths = write_current_script_result(
                        status="confirmation_required",
                        fallback_accepted=False,
                    )
                    print(
                        "[fallback] 非交互环境不会默认同意；请重新运行并显式传入 "
                        "--fallback-policy accept 或 reject。",
                        file=sys.stderr,
                    )
                    print("[报告] %s" % paths["report"])
                    return 1
                answer = input(
                    "接受上述 fallback 并继续迁移？"
                    "[a=接受 / g=准备缺口生成 / q=终止]: "
                ).strip().lower()
                if answer in {"a", "accept", "y", "yes"}:
                    policy = "accept"
                elif answer in {"g", "gap", "reject", "n", "no"}:
                    policy = "reject"
                else:
                    print("[fallback] 用户终止，未调用迁移模型，未生成迁移文件。")
                    return 130
            if policy == "reject":
                gap_request = build_gap_request(
                    analysis,
                    analysis.fallbacks,
                    target_soc=args.target_soc,
                    cann_version=args.cann_version,
                )
                paths = write_current_script_result(
                    status=(
                        "gap_generation_required"
                        if gap_request["gap_specs"]
                        else "gap_generation_unavailable"
                    ),
                    fallback_accepted=False,
                    gap_request=gap_request,
                )
                print(
                    "[fallback] 用户拒绝；未调用迁移模型，%s。"
                    % (
                        "已准备 replacement gap hand-off"
                        if gap_request["gap_specs"]
                        else "当前缺口不可由现有顶层算子生成通道处理"
                    )
                )
                print("[缺口] request=%s" % paths["gap_request"])
                if gap_request["gap_specs"]:
                    print("[下一步] " + shlex.join([
                        sys.executable,
                        "-m",
                        "numpilot",
                        "complete-gaps",
                        str(Path(args.file).absolute()),
                        "--gap-request",
                        paths["gap_request"],
                        "--out",
                        str(Path(out_dir).absolute() / "operator-gaps"),
                        "--target-soc",
                        args.target_soc,
                        "--cann-version",
                        args.cann_version,
                        "--provider",
                        "prepare",
                    ]))
                print("[报告] %s" % paths["report"])
                return 1
            fallback_accepted = True

        configuration_status = _load_llm_or_report()
        if configuration_status is None:
            paths = write_current_script_result(
                status="llm_configuration_required",
                fallback_accepted=fallback_accepted,
            )
            print("[报告] %s" % paths["report"])
            return 4
        if _llm_proxy_for_remote_call("迁移模型门禁") is None:
            paths = write_current_script_result(
                status="llm_configuration_required",
                fallback_accepted=fallback_accepted,
            )
            print("[报告] %s" % paths["report"])
            return 4
        try:
            generation_values = {
                "expected_model": configuration_status.model,
                "expected_provider": configuration_status.provider,
                "expected_protocol": getattr(
                    configuration_status,
                    "protocol",
                    "openai-chat-completions",
                ),
                "response_model_aliases": getattr(
                    configuration_status, "response_model_aliases", ()
                ),
            }
            with terminal_status(
                "调用已配置模型 %s 生成迁移代码"
                % (configuration_status.model or "unknown"),
            ):
                if qualified_analysis is None:
                    artifact = generate_script_with_configured_model(
                        analysis,
                        **generation_values,
                    )
                else:
                    artifact = generate_qualified_script_with_configured_model(
                        analysis,
                        qualified_analysis=qualified_analysis,
                        runtime_identity_provider=runtime_identity_provider,
                        overlay_admission_provider=(
                            overlay_admission_provider
                            if overlay_admission is not None
                            else None
                        ),
                        fallback_accepted=fallback_accepted,
                        **generation_values,
                    )
        except ScriptMigrationError:
            raise
        if fallback_accepted:
            status = "migrated_with_fallback"
        elif analysis.scoped:
            status = "migrated_scoped_npu"
        else:
            status = "migrated_strict_npu"
        paths = write_current_script_result(
            status=status,
            fallback_accepted=fallback_accepted,
            artifact=artifact,
        )
        print(
            "[迁移] status=%s model=%s generated_by=configured_model "
            "ast_audit=pass"
            % (status, artifact.model or configuration_status.model)
        )
        if fallback_accepted:
            print(
                "[真实性] strict_npu=false performance_valid=false；"
                "该产物是用户明确接受的兼容性迁移。"
            )
        elif analysis.scoped:
            print(
                "[真实性] strict_npu=false performance_valid=false；调用签名在资格范围内，"
                "本次未执行 profile guard；需进入 guarded contract/benchmark 验证。"
            )
        else:
            print("[真实性] 所有已扫描 API 均通过当前 strict-NPU capability。")
        print("[产物] %s" % paths["migrated"])
        print("[报告] %s" % paths["report"])
        return 0

    if mode != "contract":
        raise ScriptMigrationError("unknown migration mode: %s" % mode)
    contract_engine = args.engine or "llm"
    if contract_engine == "llm":
        if _load_llm_or_report() is None:
            return 4
        if _llm_proxy_for_remote_call("合同迁移门禁") is None:
            return 4

    orch = Orchestrator(
        backend=backend,
        engine=_make_engine(contract_engine),
        threshold=args.threshold,
        analysis_profiles=analysis_profiles,
    )
    res = orch.run(args.file, out_dir=args.out)

    print(_render_terminal(res))
    try:
        rp = save_migration_report(
            _migrate_report_info(res), reports_dir=args.reports_dir
        )
        print(f"[报告] 迁移报告已落盘:{rp}")
    except Exception as e:
        print(f"[报告] 落盘失败(不影响迁移):{e}")
    if res["status"] == "unsupported":
        return 1
    return 0 if res["status"] == "pass" else 2


def _migrate_report_info(res):
    """把 migrate 结果规范化为 report.build_report_md 的 info dict(无性能字段)。"""
    from numpilot.report import agent_roster
    net = _net_name(res.get("source_path", "task"))
    info = {"net": net, "source_path": res.get("source_path"),
            "engine": res.get("engine"), "status": res["status"],
            "agents": agent_roster(res.get("engine"))}
    a = res.get("analysis")
    if res["status"] == "unsupported":
        info["unsupported"] = res.get("unsupported", [])
        info["unsupported_details"] = [
            {"name": op.name, "lineno": op.lineno, "reason": op.reason}
            for op in (a.op_uses if a else []) if not op.supported
        ]
        return info
    rep = res["report"]
    plan = res.get("plan")
    rewrites = res.get("rewrites") or {}
    info["backend"] = rep.backend
    info["n_calls"] = len(a.op_uses) if a else None
    info["n_numpy_calls"] = (
        sum(op.kind == "numpy_api" for op in a.op_uses) if a else None
    )
    info["n_operator_calls"] = (
        sum(op.kind == "domain_operator" for op in a.op_uses) if a else None
    )
    info["n_ops"] = len({o.name for o in a.op_uses}) if a else None
    info["rewrites"] = rewrites
    info["total_rewrites"] = sum(rewrites.values())
    info["extension_apis"] = {op: n for op, n in rewrites.items() if is_custom(op)}
    if a:
        info["operator_specs"] = sorted({
            op.name for op in a.op_uses if op.kind == "domain_operator"
        })
        info["extension_apis"].update({
            op: rewrites.get(op, 0) for op in info["operator_specs"]
        })
    if plan:
        info["strategy"] = plan.strategy
        info["npu_ops"] = sorted(o for o, p in plan.placement.items() if p == "NPU")
        info["cpu_ops"] = sorted(o for o, p in plan.placement.items() if p == "CPU")
    info["backend_version"] = rep.backend_version
    info["backend_module"] = rep.backend_module
    info["numerically_consistent"] = rep.numerically_consistent
    info["max_rel_diff"] = rep.max_rel_diff
    info["max_abs_diff"] = rep.max_abs_diff
    info["tol"] = rep.tol
    info["atol"] = rep.atol
    info["rtol"] = rep.rtol
    info["failure_path"] = rep.failure_path
    info["failure_index"] = rep.failure_index
    info["failure_reason"] = rep.failure_reason
    info["exception_phase"] = rep.exception_phase
    info["exception_type"] = rep.exception_type
    info["exception_message"] = rep.exception_message
    info["artifact_inputs_executed"] = rep.artifact_inputs_executed
    info["input_consistent"] = rep.input_consistent
    info["kernel_consistent"] = rep.kernel_consistent
    info["artifact_consistent"] = rep.artifact_consistent
    info["comparisons"] = rep.comparisons
    info["backend_provenance"] = rep.backend_provenance
    info["performance_valid"] = False
    info["migrated_path"] = res.get("migrated_path")
    return info


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else list(argv)
    if len(arguments) >= 2 and [item.lower() for item in arguments[:2]] == [
        "check",
        "llm",
    ]:
        # Keep the exact phrase shown by the persistent terminal useful for a
        # one-shot shell invocation too: ``numpilot check llm``.
        arguments = ["check-llm", *arguments[2:]]
    if not arguments:
        if rich_available():
            from numpilot.repl import can_start_repl, run_repl

            if can_start_repl(sys.stdin, sys.stdout):
                return run_repl(__version__, main)
        if render_home(__version__):
            return 0

    parser = argparse.ArgumentParser(
        prog="numpilot",
        description=(
            "将普通 NumPy 程序迁移为可验证、可审计的 Ascend NPU 产物。"
        ),
        epilog=(
            "常用流程:\n"
            "  numpilot doctor --backend npu\n"
            "  numpilot analyze script.py\n"
            "  numpilot migrate script.py --out RUN_DIR\n"
            "  numpilot verify script.py --migration-report RUN_DIR/migration/script-migration-report.json --output result --out RUN_DIR/verification.json\n"
            "  numpilot replay RUN_DIR\n"
            "使用 `numpilot <command> --help` 查看命令参数。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--llm-config",
        type=_llm_config_arg,
        default=None,
        metavar="PATH",
        help=(
            "本次命令使用指定的私有 llm.json；可放在命令前或子命令后，"
            "不会打印 Token"
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    analyze = sub.add_parser(
        "analyze",
        help="只读分析普通 NumPy 文件的调用画像与迁移资格；不执行源码或模型",
    )
    analyze.add_argument("file", help="要静态分析的普通 Python/NumPy 文件")
    analyze.add_argument(
        "--json",
        action="store_true",
        help="向标准输出写完整、机器可读的资格分析 JSON",
    )
    analyze.add_argument(
        "--out",
        metavar="PATH",
        help="write-once JSON 回执文件；文件 0600、直接父目录 0700",
    )
    analyze.add_argument(
        "--require-direct",
        action="store_true",
        help="CI 门禁：分析仍会完成，但最终判定不是 DIRECT 时退出 1",
    )

    bootstrap = sub.add_parser(
        "bootstrap",
        help="只读检查依赖，或显式准备固定 AsNumpy/CANNBot 版本",
    )
    bootstrap_mode = bootstrap.add_mutually_exclusive_group()
    bootstrap_mode.add_argument(
        "--plan",
        action="store_true",
        help="只读展示计划（默认）；不联网、不修改环境",
    )
    bootstrap_mode.add_argument(
        "--apply",
        action="store_true",
        help="执行显式计划并原子写入 ready/failed receipt",
    )
    bootstrap_mode.add_argument(
        "--verify",
        action="store_true",
        help="复验当前 AsNumpy；可同时本地校验 CANNBot Skill 清单（不联网）",
    )
    bootstrap.add_argument("--json", action="store_true", help="输出完整 JSON")
    asnumpy_mode = bootstrap.add_mutually_exclusive_group()
    asnumpy_mode.add_argument(
        "--asnumpy-component",
        metavar="PATH",
        help=(
            "安装发行附件中的固定 AsNumpy 内部组件目录（默认只读本地 "
            "manifest.json + 精确 SHA-256 wheel，不联网）"
        ),
    )
    asnumpy_mode.add_argument(
        "--asnumpy-wheel",
        metavar="PATH",
        help="可信发布物提供的本地 AsNumpy wheel；不会自动猜测下载地址",
    )
    asnumpy_mode.add_argument(
        "--build-asnumpy",
        action="store_true",
        help=(
            "维护者构建下一版 AsNumpy（显式维护者开关，可能联网获取固定源码；"
            "用户安装请使用不联网的 --asnumpy-component）"
        ),
    )
    bootstrap.add_argument(
        "--asnumpy-wheel-sha256",
        metavar="HEX",
        help="本地 wheel 的可信 SHA-256；与 --asnumpy-wheel 成对使用",
    )
    bootstrap.add_argument(
        "--with-cannbot",
        action="store_true",
        help="同时准备固定 CANNBot Skills（缺口算子流程才需要）",
    )
    bootstrap.add_argument(
        "--accept-cannbot-license",
        action="store_true",
        help="确认已审阅并接受 CANN Open Software License Agreement v2.0",
    )
    bootstrap.add_argument(
        "--asnumpy-cache",
        metavar="PATH",
        help="固定 AsNumpy 源码缓存路径；默认使用版本化用户缓存",
    )
    bootstrap.add_argument(
        "--cannbot-cache",
        metavar="PATH",
        help="固定 CANNBot 缓存路径；默认使用版本化用户缓存",
    )
    bootstrap.add_argument(
        "--receipt",
        default=str(default_bootstrap_receipt()),
        help="apply 结果路径（默认 %(default)s）",
    )

    mig = sub.add_parser("migrate", help="迁移一个 numpy 程序到 AsNumpy(NPU)")
    mig.add_argument("file")
    mig.add_argument("--backend", choices=["shim", "npu"], default=None)
    mig.add_argument(
        "--engine",
        choices=["ast", "llm"],
        default=None,
        help=(
            "普通脚本始终要求已配置模型；contract 默认 llm，ast 只保留给框架离线回归"
        ),
    )
    mig.add_argument(
        "--mode",
        choices=["auto", "script", "contract"],
        default="script",
        help=(
            "默认 script：整个普通纯 NumPy 文件走扫描/询问/已配置模型迁移；"
            "旧 build_inputs+forward 严格通道必须显式选择 contract；"
            "auto 仅为旧自动化兼容"
        ),
    )
    mig.add_argument(
        "--fallback-policy",
        choices=["ask", "accept", "reject"],
        default="ask",
        help=(
            "普通脚本发现已审计 CPU/host fallback 时的策略；默认交互询问，"
            "非交互环境必须显式选择"
        ),
    )
    mig.add_argument(
        "--target-soc",
        default=os.getenv("ASCEND_SOC_VERSION", "ascend910b"),
        help="拒绝 fallback 后 replacement gap hand-off 的目标 SoC",
    )
    mig.add_argument(
        "--cann-version",
        default=os.getenv("CANN_VERSION", "unknown"),
        help="拒绝 fallback 后 replacement gap hand-off 的 CANN 版本",
    )
    mig.add_argument("--out", default=None, help="结构化报告与迁移代码输出目录")
    mig.add_argument(
        "--reports-dir",
        default=None,
        help="时间戳报告目录；默认使用仓库外用户状态目录",
    )
    mig.add_argument("--threshold", type=int, default=1_000_000)
    mig.add_argument("--skip-env-check", action="store_true",
                     help="跳过 migrate 前置环境自检(缺依赖也强制继续)")
    mig.add_argument(
        "--capability-overlay",
        default=None,
        help="显式启用的 qualification overlay JSON（仅 NPU；不自动发现）",
    )
    mig.add_argument(
        "--qualification-evidence-root",
        default=None,
        help="overlay 证据根目录；必须与 --capability-overlay 成对提供",
    )
    mig.add_argument(
        "--analysis-profiles",
        default=None,
        help="与源 SHA/callsite 精确绑定的 runtime profile JSON",
    )

    run_parser = sub.add_parser(
        "run",
        aliases=["pipeline"],
        help=(
            "一键执行 ordinary 主链：静态分析→configured-model 迁移→"
            "数值验证（可选 strict-NPU 性能门）"
        ),
    )
    run_parser.add_argument("file", help="普通 NumPy 源文件")
    run_parser.add_argument(
        "--backend",
        choices=["npu", "shim"],
        default="npu",
        help="目标后端；严格验证使用 npu，shim 仅用于数值诊断（不产生 NPU 结论）",
    )
    run_parser.add_argument(
        "--fallback-policy",
        choices=["ask", "accept", "reject"],
        default="reject",
        help="遇到已审计 fallback 时的策略（默认 reject，避免非交互误放行）",
    )
    run_parser.add_argument(
        "--target-soc",
        default=os.getenv("ASCEND_SOC_VERSION", "unknown"),
        help="缺口恢复绑定的目标 SoC",
    )
    run_parser.add_argument(
        "--cann-version",
        default=os.getenv("CANN_VERSION", "unknown"),
        help="缺口恢复绑定的 CANN 版本",
    )
    run_parser.add_argument(
        "--output",
        dest="outputs",
        action="append",
        default=None,
        help="显式命名的数值/布尔全局输出（可重复，默认 result）",
    )
    run_parser.add_argument(
        "--out",
        default=None,
        help="全新私有运行目录；省略时创建在 NumPilot runs/demo 下",
    )
    run_parser.add_argument(
        "--benchmark",
        action="store_true",
        help="在数值验证通过后追加 strict-NPU resident 性能门（仅 backend=npu，可能耗时）",
    )
    run_parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="性能门预热次数（默认 3）",
    )
    run_parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="性能门原始样本数（默认 20；启用 --benchmark 时至少 7）",
    )
    run_parser.add_argument(
        "--core",
        type=int,
        default=None,
        help="CPU/NPU worker 共用的逻辑核；默认当前 affinity 中第一个核",
    )
    run_parser.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="每个阶段子进程的超时秒数（默认 3600）",
    )
    run_parser.add_argument(
        "--python",
        default=None,
        help="阶段子进程使用的 Python；默认当前解释器",
    )
    run_parser.add_argument(
        "--skip-env-check",
        action="store_true",
        help="传递给迁移阶段；仅在已自行完成环境审计时使用",
    )
    run_parser.add_argument(
        "--json",
        action="store_true",
        help="只输出 workflow-report JSON，不混入阶段日志",
    )

    interactive = sub.add_parser(
        "interactive-migrate",
        aliases=["interactive-run"],
        help=(
            "交互式普通脚本迁移：读取任意绝对路径→静态分析/模型解读→"
            "y 确认→迁移、验证与可选性能门"
        ),
    )
    interactive.add_argument(
        "file",
        nargs="?",
        help="待迁移 Python 源文件绝对路径；省略时仅在 TTY 中交互询问",
    )
    interactive.add_argument(
        "--source",
        dest="source_option",
        help="与位置参数等价的源文件绝对路径，便于脚本/自动化调用",
    )
    interactive.add_argument("--backend", choices=["npu", "shim"], default="npu")
    interactive.add_argument(
        "--fallback-policy",
        choices=["ask", "accept", "reject"],
        default="reject",
        help="发现已审计 fallback 时的策略（默认 reject）",
    )
    interactive.add_argument(
        "--target-soc",
        default=os.getenv("ASCEND_SOC_VERSION", "unknown"),
    )
    interactive.add_argument(
        "--cann-version",
        default=os.getenv("CANN_VERSION", "unknown"),
    )
    interactive.add_argument(
        "--output",
        dest="outputs",
        action="append",
        default=None,
        help="脚本全局输出变量，可重复；默认 result",
    )
    interactive.add_argument(
        "--out",
        default=None,
        help="本次交互会话的全新绝对目录（含算子、迁移、验证、性能与总报告）",
    )
    benchmark_mode = interactive.add_mutually_exclusive_group()
    benchmark_mode.add_argument(
        "--benchmark",
        dest="benchmark",
        action="store_true",
        help="验证通过后追加 strict-NPU resident benchmark（默认开启）",
    )
    benchmark_mode.add_argument(
        "--no-benchmark",
        dest="benchmark",
        action="store_false",
        help="只做迁移与数值验证",
    )
    interactive.set_defaults(benchmark=True)
    interactive.add_argument("--warmup", type=int, default=3)
    interactive.add_argument("--samples", type=int, default=20)
    interactive.add_argument("--core", type=int, default=None)
    interactive.add_argument("--timeout", type=float, default=3600.0)
    interactive.add_argument("--python", default=None)
    interactive.add_argument("--skip-env-check", action="store_true")
    interactive.add_argument(
        "--interpret-max-tokens",
        type=int,
        default=1800,
        help="迁移前源码解读的单次模型预算（默认 1800；空响应最多一次加大重试）",
    )
    interactive.add_argument(
        "--yes",
        action="store_true",
        help="自动确认迁移；仅用于明确授权的自动化/录制",
    )
    interactive.add_argument(
        "--execute-source",
        action="store_true",
        help="独立授权执行已展示且绑定哈希的用户源码；通用 --yes 不包含此授权",
    )
    interactive.add_argument(
        "--non-interactive",
        action="store_true",
        help="与 --yes 一起用于无 TTY 自动化；没有 --yes 时仍 fail-closed",
    )
    interactive.add_argument(
        "--cannbot-gap-draft",
        action="store_true",
        help="确认后对发现的 GapSpec 走官方解析→CANNBot Skill→真实模型 draft-only 路由",
    )
    interactive.add_argument(
        "--accept-cannbot-license",
        action="store_true",
        help="明确接受本次 CANNBot/CANN OSL（CANNBot 路由必需）",
    )
    interactive.add_argument(
        "--show-cannbot-skill",
        choices=["none", "summary", "full"],
        default="full",
        help="CANNBot 生成阶段的公开 Skill 展示级别（默认 full）",
    )
    interactive.add_argument("--cannbot-skill-root", default=None)
    interactive.add_argument("--cann-root", default=os.getenv("ASCEND_HOME_PATH", ""))
    interactive.add_argument("--cannbot-max-tokens", type=int, default=4096)
    interactive.add_argument("--cannbot-timeout", type=float, default=300.0)
    interactive.add_argument(
        "--skill-path",
        default=None,
        help="显式提供的已校验 NumPilot Skill 路径",
    )

    verify = sub.add_parser(
        "verify",
        aliases=["verify-script"],
        help="canonical：执行 ordinary 原脚本与已发布迁移产物并做哈希绑定数值对拍",
    )
    verify.add_argument("file", help="migrate 使用的原始 NumPy 脚本")
    verify.add_argument(
        "--migration-report",
        required=True,
        help="ordinary migrate 发布的 script-migration-report.json",
    )
    verify.add_argument(
        "--migrated",
        default=None,
        help="迁移脚本；默认只接受 migration report 同目录的 migrated.py",
    )
    verify.add_argument(
        "--output",
        dest="outputs",
        action="append",
        required=True,
        help="两侧脚本都必须赋值的输出变量；可重复提供",
    )
    verify.add_argument(
        "--out",
        required=True,
        help="write-once script verification JSON 回执路径",
    )
    verify.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="每个隔离 worker 的超时秒数（默认 %(default)s）",
    )
    verify.add_argument(
        "--max-output-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="每个 worker 的输出总字节上限（默认 64 MiB）",
    )
    verify.add_argument(
        "--source-python",
        default=None,
        help="原始 NumPy 脚本 worker 的 Python；默认当前解释器",
    )
    verify.add_argument(
        "--migrated-python",
        default=None,
        help="迁移脚本 worker 的 Python；默认当前解释器",
    )
    verify.add_argument(
        "--json",
        action="store_true",
        help="将完整 verification receipt 输出到 stdout",
    )

    replay = sub.add_parser(
        "replay",
        help="canonical：只读校验 workflow report 与阶段回执，不执行模型、源码或 NPU",
    )
    replay.add_argument("run_dir", help="已完成或已停止的 workflow 运行目录")
    replay.add_argument(
        "--json",
        action="store_true",
        help="将完整 replay receipt 输出到 stdout",
    )

    benchmark_script_parser = sub.add_parser(
        "benchmark-script",
        aliases=["bench-script"],
        help=(
            "对 hash-bound ordinary GLM 产物做单核 CPU/strict-NPU "
            "设备驻留性能验证"
        ),
    )
    benchmark_script_parser.add_argument(
        "file", help="migrate 使用的原始 NumPy 脚本"
    )
    benchmark_script_parser.add_argument(
        "--migration-report",
        required=True,
        help="ordinary migrate 发布的 script-migration-report.json",
    )
    benchmark_script_parser.add_argument(
        "--migrated",
        default=None,
        help="迁移脚本；默认且只接受 migration report 同目录的 migrated.py",
    )
    benchmark_script_parser.add_argument(
        "--output",
        dest="outputs",
        action="append",
        required=True,
        help="两侧脚本都必须赋值的最终输出变量；可重复提供",
    )
    benchmark_script_parser.add_argument(
        "--out",
        required=True,
        help="write-once ordinary benchmark JSON 回执路径",
    )
    benchmark_script_parser.add_argument(
        "--marker",
        default="# NUMPILOT_DEVICE_INPUTS_END",
        help="输入准备与设备驻留计时区之间唯一出现的注释 marker",
    )
    benchmark_script_parser.add_argument(
        "--core",
        type=int,
        default=None,
        help="CPU/NPU worker 共用的逻辑核；默认当前 affinity 中第一个核",
    )
    benchmark_script_parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="每侧预热次数（默认 %(default)s）",
    )
    benchmark_script_parser.add_argument(
        "--samples",
        "--repeats",
        dest="samples",
        type=int,
        default=20,
        help="每侧有序样本数，至少 7（默认 %(default)s）",
    )
    benchmark_script_parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="每个隔离 worker 的超时秒数（默认 %(default)s）",
    )
    benchmark_script_parser.add_argument(
        "--max-output-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="每个 worker 最终输出总字节上限（默认 64 MiB）",
    )
    benchmark_script_parser.add_argument(
        "--source-python",
        default=None,
        help="NumPy CPU worker 的 Python；默认当前解释器",
    )
    benchmark_script_parser.add_argument(
        "--migrated-python",
        default=None,
        help="AsNumpy NPU worker 的 Python；默认当前解释器",
    )
    benchmark_script_parser.add_argument(
        "--json",
        action="store_true",
        help="将完整 benchmark receipt 输出到 stdout",
    )

    bc = sub.add_parser("bench-cpu", help="① 纯 numpy 基线, 进程内钉 1 核, 存基线供对比")
    bc.add_argument("file")
    bc.add_argument(
        "--out",
        default=str(default_benchmark_dir()),
        help="基线文件输出目录（默认 %(default)s）",
    )
    bc.add_argument("--cpu-threads", default="1",
                    help="请求的 NumPy BLAS 线程数(默认 1；0/all 表示不主动限制)")

    bn = sub.add_parser(
        "bench-npu", help="② 审计并评测 asnumpy；仅严格 NPU 路径发布加速比")
    bn.add_argument("file")
    bn.add_argument(
        "--out",
        default=str(default_benchmark_dir()),
        help="读取 ① 基线 / 写迁移代码的目录（默认 %(default)s）",
    )
    bn.add_argument("--engine", choices=["ast", "llm"], default=None)
    bn.add_argument("--threshold", type=int, default=1_000_000)
    bn.add_argument("--skip-env-check", action="store_true")
    bn.add_argument(
        "--allow-cpu-fallback", action="store_true",
        help="仅供诊断：允许已知 CPU fallback/来源未知路径继续运行，但不发布 NPU 加速比")
    bn.add_argument(
        "--capability-overlay",
        default=None,
        help="显式启用的 qualification overlay JSON（不自动发现）",
    )
    bn.add_argument(
        "--qualification-evidence-root",
        default=None,
        help="overlay 证据根目录；必须与 --capability-overlay 成对提供",
    )
    bn.add_argument(
        "--analysis-profiles",
        default=None,
        help="与源 SHA/callsite 精确绑定的 runtime profile JSON",
    )

    wiz = sub.add_parser(
        "wizard",
        help="legacy contract 开发向导(build_inputs/forward；ordinary 请用 migrate)",
    )
    wiz.add_argument("file", nargs="?", default=None,
                     help="待迁移程序路径(省略则进入交互提示)")
    wiz.add_argument("--engine", choices=["ast", "llm"], default=None,
                     help="迁移引擎(省略则交互选择)")
    wiz.add_argument("--backend", choices=["shim", "npu"], default="npu")
    wiz.add_argument(
        "--out",
        default=None,
        help="迁移代码目录；默认使用仓库外用户状态目录",
    )
    wiz.add_argument(
        "--reports-dir",
        default=None,
        help="时间戳报告目录；默认使用仓库外用户状态目录",
    )

    doc = sub.add_parser("doctor", help="环境自检:检查 Python/numpy/CANN/NPU/asnumpy")
    doc.add_argument("--backend", choices=["shim", "npu"], default=None,
                     help="以指定后端语义评估硬依赖(npu 更严格)")
    doc.add_argument(
        "--json",
        action="store_true",
        help="只向标准输出写机器可读 JSON；不混入 Rich/进度文本",
    )

    paths = sub.add_parser(
        "paths", help="显示配置、缓存与运行产物的仓库外路径"
    )
    paths.add_argument("--json", action="store_true", help="输出 JSON")
    paths.add_argument(
        "--get",
        choices=("persistent_home", "state", "config", "cache", "runs",
                 "reports", "benchmark", "wizard", "gaps", "demo",
                 "bootstrap_receipt", "llm_config", "asnumpy_cache_base",
                 "asnumpy_component", "asnumpy_cache", "cannbot_cache_base", "cannbot_cache"),
        help="只输出一个路径值",
    )

    suite = sub.add_parser(
        "check-suite",
        help="不执行源码，静态检查一组案例是否符合直接迁移/单缺口测试合同",
    )
    suite.add_argument(
        "paths",
        nargs="+",
        help="Python 案例文件或包含案例的目录（目录只扫描当前层 *.py）",
    )
    suite.add_argument(
        "--target-soc",
        default=os.getenv("ASCEND_SOC_VERSION", "ascend910b"),
        help="单缺口 GapSpec 的目标 SoC（默认 %(default)s）",
    )
    suite.add_argument(
        "--cann-version",
        default=os.getenv("CANN_VERSION", "9.2.0-beta.2"),
        help="单缺口 GapSpec 的目标 CANN 版本（默认 %(default)s）",
    )
    suite.add_argument(
        "--expect-direct",
        type=int,
        default=None,
        help="期望的无缺口直接迁移候选数，例如本测试集填 5",
    )
    suite.add_argument(
        "--expect-single-gap",
        type=int,
        default=None,
        help="期望的恰好一个简单缺口候选数，例如本测试集填 5",
    )
    suite.add_argument(
        "--out",
        default=None,
        help="可选：原子写入 JSON 准入报告",
    )
    suite.add_argument(
        "--json",
        action="store_true",
        help="只向标准输出写机器可读 JSON；不混入人类摘要",
    )

    check_llm_parser = sub.add_parser(
        "check-llm",
        help="向 llm.json 中的模型发送最小消息并验证真实应答",
    )

    cfg = sub.add_parser(
        "configure-llm",
        help="在 NumPilot 内填写并安全保存 LLM Base URL、Token 与 Model",
    )
    cfg.add_argument("--base-url", default=None, help="OpenAI 兼容 API 根地址")
    cfg.add_argument("--model", default=None, help="模型名")
    cfg.add_argument(
        "--provider", default=None,
        help="仅用于审计的 provider 标签，不参与厂商授权",
    )
    cfg.add_argument(
        "--protocol",
        choices=(OPENAI_CHAT_COMPLETIONS_PROTOCOL,),
        default=None,
        help="协议合同（当前仅支持 %(choices)s）",
    )
    cfg.add_argument(
        "--response-model-alias",
        action="append",
        default=None,
        help="显式允许的响应 model 别名；可重复，默认不允许别名",
    )
    cfg.add_argument(
        "--token-stdin",
        action="store_true",
        help="从标准输入读取 Token；避免把 Token 放入命令行历史/进程参数",
    )
    cfg_mode = cfg.add_mutually_exclusive_group()
    cfg_mode.add_argument(
        "--init",
        action="store_true",
        help="生成私有 JSON 配置模板供手动填写；绝不覆盖已有文件",
    )
    cfg_mode.add_argument(
        "--check",
        action="store_true",
        help="只检查配置是否完整，不联网、不显示 Token",
    )
    cfg_mode.add_argument(
        "--probe",
        action="store_true",
        help="发起一次真实轻量请求；失败返回非零退出码且不显示 Token/响应正文",
    )

    skills = sub.add_parser(
        "setup-skills",
        help="获取并校验固定版本的华为官方 CANNBot 算子开发 Skill",
    )
    skills.add_argument(
        "--root",
        default=None,
        help="校验已有 cannbot-skills checkout；省略时使用版本化用户缓存",
    )
    skills.add_argument(
        "--check",
        action="store_true",
        help="只检查，不在缓存缺失时联网拉取",
    )
    skills.add_argument(
        "--accept-cannbot-license",
        action="store_true",
        help=(
            "获取默认缓存前确认已审阅并接受 CANN Open Software License "
            "Agreement v2.0；--check 或显式 --root 不触发下载"
        ),
    )

    resolve_gaps = sub.add_parser(
        "resolve-gaps",
        help=(
            "只读解析缺口：优先已安装 CANN ACLNN，其次固定版本 ops-nn，"
            "两者均明确无匹配后才允许新 kernel"
        ),
    )
    resolve_gaps.add_argument("file", help="待静态分析的 NumPy Python 源文件")
    resolve_gaps.add_argument(
        "--gap-request",
        help="可选：migrate 生成且会被重新绑定校验的 gap-request.json",
    )
    resolve_gaps.add_argument(
        "--out",
        required=True,
        help="write-once、0600 的 official-resolution.json 收据路径",
    )
    resolve_gaps.add_argument(
        "--target-soc",
        default=os.getenv("ASCEND_SOC_VERSION", "unknown"),
        help=(
            "精确目标 SoC（例如 Ascend910B4）；family 名称 Ascend910B "
            "不会被推断为某个具体芯片"
        ),
    )
    resolve_gaps.add_argument(
        "--cann-version",
        default=os.getenv("CANN_VERSION", "unknown"),
        help="明确的目标 CANN 版本；必须与本地工具链和源码 pin 一致",
    )
    resolve_gaps.add_argument(
        "--cann-root",
        default=os.getenv("ASCEND_HOME_PATH", ""),
        help="明确、规范且只读盘点的 CANN toolkit 根目录",
    )
    resolve_gaps.add_argument(
        "--ops-nn-root",
        default=os.getenv("NUMPILOT_OPS_NN_ROOT", ""),
        help=(
            "本地 Huawei ops-nn Git checkout；只读固定 tag 的 Git object，"
            "不 fetch/checkout/build"
        ),
    )
    resolve_gaps.add_argument(
        "--accept-cann-osl",
        action="store_true",
        help=(
            "确认已审阅并接受 CANN Open Software License Agreement v2.0；"
            "仅 installed no_match 后读取 ops-nn 时需要"
        ),
    )
    resolve_gaps.add_argument(
        "--allow-custom-operator-generation",
        action="store_true",
        help=(
            "显式允许 provenance=custom_generation_candidate 且 route="
            "ascendc_registry_candidate 的 OperatorSpec 进入 draft-only "
            "路由；不把 catalog_unreviewed_api 伪装成 clean no_match，"
            "官方候选仍优先"
        ),
    )

    official_candidate = sub.add_parser(
        "prepare-official-candidate",
        help=(
            "验证已发现的 installed ACLNN candidate，并发布独立的"
            " official_candidate_gap hand-off（不调用模型、不构建）"
        ),
    )
    official_candidate.add_argument(
        "file", help="原始 NumPy 源文件"
    )
    official_candidate.add_argument(
        "--official-resolution-report", required=True,
        help="resolve-gaps 生成的官方解析回执",
    )
    official_candidate.add_argument(
        "--out", required=True,
        help="新的 write-once official_candidate_gap JSON 回执",
    )
    official_candidate.add_argument(
        "--json", action="store_true",
        help="只输出机器可读回执",
    )

    official_preflight = sub.add_parser(
        "preflight-official-candidate",
        help="重新盘点官方候选并发布只读 adapter materialization preflight",
    )
    official_preflight.add_argument("file", help="原始 NumPy 源文件")
    official_preflight.add_argument(
        "--handoff", required=True, help="prepare-official-candidate 生成的 hand-off"
    )
    official_preflight.add_argument(
        "--official-resolution-report", required=True,
        help="resolve-gaps 生成的官方解析回执",
    )
    official_preflight.add_argument("--out", required=True, help="新的 preflight 回执")
    official_preflight.add_argument("--json", action="store_true", help="只输出机器可读回执")

    adapter_input = sub.add_parser(
        "prepare-official-adapter-input",
        help="从 official candidate preflight 生成 inert AsNumpy adapter input",
    )
    adapter_input.add_argument("file", help="原始 NumPy 源文件")
    adapter_input.add_argument("--handoff", required=True, help="official candidate hand-off 回执")
    adapter_input.add_argument("--preflight", required=True, help="official candidate preflight 回执")
    adapter_input.add_argument("--official-resolution-report", required=True, help="官方解析回执")
    adapter_input.add_argument("--out", required=True, help="新的 write-once adapter input")
    adapter_input.add_argument("--json", action="store_true", help="只输出机器可读回执")

    adapter_gate = sub.add_parser(
        "validate-official-adapter-context",
        help="校验 context-bound official adapter model gate，不调用模型",
    )
    adapter_gate.add_argument("bound", help="context-bound adapter receipt")
    adapter_gate.add_argument("--handoff", required=True, help="official candidate hand-off")
    adapter_gate.add_argument("--preflight", required=True, help="official candidate preflight")
    adapter_gate.add_argument("--source", required=True, help="原始 NumPy 源文件")
    adapter_gate.add_argument("--official-resolution-report", required=True, help="官方解析回执")
    adapter_gate.add_argument("--out", required=True, help="新的 write-once gate receipt")
    adapter_gate.add_argument("--json", action="store_true", help="只输出机器可读回执")

    official_adapter_generate = sub.add_parser(
        "official-adapter-generate",
        help=(
            "官方 adapter 生成前的只读 contract preflight；仅校验绑定，"
            "不调用模型、不生成 overlay、不写 RecoveryStore"
        ),
        description=(
            "官方 adapter 生成前的只读 contract preflight。仅校验绑定，"
            "不调用模型、不生成 overlay、不写 RecoveryStore。"
        ),
    )
    official_adapter_generate.add_argument("bound", help="context-bound adapter receipt")
    official_adapter_generate.add_argument("--handoff", required=True, help="official candidate hand-off")
    official_adapter_generate.add_argument("--preflight", required=True, help="official candidate preflight")
    official_adapter_generate.add_argument("--source", required=True, help="原始 NumPy 源文件")
    official_adapter_generate.add_argument("--official-resolution-report", required=True, help="官方解析回执")
    official_adapter_generate.add_argument("--out", required=True, help="新的 write-once gate receipt")
    official_adapter_generate.add_argument("--json", action="store_true", help="只输出机器可读回执")

    official_recovery = sub.add_parser(
        "materialize-official-candidate-recovery",
        help="消费 official_candidate_gap 并安全推进到 CANDIDATE_READY，不调用模型",
        description="消费 official_candidate_gap，推进到 CANDIDATE_READY；本命令不调用模型、不联网、不生成或激活 adapter。",
    )
    official_recovery.add_argument("--handoff", required=True, help="official candidate hand-off")
    official_recovery.add_argument("--preflight", required=True, help="official candidate preflight")
    official_recovery.add_argument("--source", required=True, help="原始 NumPy 源文件")
    official_recovery.add_argument("--official-resolution-report", required=True, help="官方解析回执")
    official_recovery.add_argument("--context-gate", required=True, help="context-bound official adapter gate receipt")
    official_recovery.add_argument("--state-dir", required=True, help="私有 RecoveryStore 目录")
    official_recovery.add_argument("--output-dir", help="私有官方候选 artifact 目录")
    official_recovery.add_argument("--runtime-backend", required=True, help="显式 runtime backend")
    official_recovery.add_argument("--runtime-audit-id", required=True, help="显式 runtime audit id")
    official_recovery.add_argument("--runtime-identity-sha256", required=True, help="显式 runtime identity SHA-256")
    official_recovery.add_argument("--run-id", help="可选 recovery run id")
    official_recovery.add_argument("--profile", default="receipt-only", choices=["receipt-only", "candidate-generation"], help="receipt-only 仅建候选；candidate-generation 为后续模型请求冻结权限")
    official_recovery.add_argument("--json", action="store_true", help="只输出机器可读回执")

    recover_gaps = sub.add_parser(
        "recover-gaps",
        help="导入可信官方回执并查询可续接的缺口恢复状态",
    )
    recovery_sub = recover_gaps.add_subparsers(
        dest="recovery_command", required=True
    )
    recovery_ingest = recovery_sub.add_parser(
        "ingest-official",
        help="校验并导入迁移、缺口和官方解析回执",
    )
    recovery_ingest.add_argument(
        "source", metavar="SOURCE", help="回执绑定的原始 NumPy 源文件"
    )
    recovery_ingest.add_argument(
        "--migration-report", required=True, help="脚本迁移报告 JSON"
    )
    recovery_ingest.add_argument(
        "--gap-request", required=True, help="缺口请求 JSON"
    )
    recovery_ingest.add_argument(
        "--official-resolution-report",
        required=True,
        help="官方能力解析报告 JSON",
    )
    recovery_ingest.add_argument(
        "--state-dir", required=True, help="write-once 恢复状态目录"
    )
    recovery_ingest.add_argument(
        "--run-id", default=None, help="可选的稳定恢复运行标识"
    )
    recovery_ingest.add_argument(
        "--authorization-profile",
        choices=(
            "receipt-only",
            "candidate-generation",
            "full-local-recovery",
        ),
        default="receipt-only",
        help=(
            "在 run 创建时冻结后续授权；默认 receipt-only。"
            "旧状态不能升级或改写该选择"
        ),
    )
    recovery_ingest.add_argument(
        "--json", action="store_true", help="仅向 stdout 输出机器可读 JSON"
    )
    recovery_status_parser = recovery_sub.add_parser(
        "status", help="校验完整状态链并显示当前恢复状态"
    )
    recovery_status_parser.add_argument(
        "--state-dir", required=True, help="恢复状态目录"
    )
    recovery_status_parser.add_argument(
        "--json", action="store_true", help="仅向 stdout 输出机器可读 JSON"
    )
    recovery_materialize = recovery_sub.add_parser(
        "materialize-candidates",
        help=(
            "重新盘点当前环境并把 RESOLVED 的已安装 CANN 官方候选"
            "物化为 CANDIDATE_READY 收据"
        ),
    )
    recovery_materialize.add_argument(
        "--state-dir", required=True, help="已通过完整 hash-chain 校验的恢复状态目录"
    )
    recovery_materialize.add_argument(
        "--out", required=True, help="必须为绝对路径的私有、write-once 候选目录"
    )
    recovery_materialize.add_argument(
        "--json", action="store_true", help="仅向 stdout 输出机器可读 JSON"
    )
    recovery_adapter = recovery_sub.add_parser(
        "ingest-adapter",
        help=(
            "复验已发布的 official ACLNN AsNumpy overlay，并把 "
            "CANDIDATE_READY 推进到 ADAPTER_CANDIDATE"
        ),
    )
    recovery_adapter.add_argument(
        "--state-dir", required=True, help="已通过完整 hash-chain 校验的恢复状态目录"
    )
    recovery_adapter.add_argument(
        "--candidate", required=True, help="恢复状态精确绑定的 operator-candidate.json"
    )
    recovery_adapter.add_argument(
        "--adapter-output",
        required=True,
        help="包含 asnumpy-adapter-report.json 与 candidate/ 的已发布目录",
    )
    recovery_adapter.add_argument(
        "--json", action="store_true", help="仅向 stdout 输出机器可读 JSON"
    )

    recovery_codex_adapter = recovery_sub.add_parser(
        "ingest-codex-adapter",
        help=(
            "复验外部 Codex CLI 生成的 schema-5 本地 overlay，并把 "
            "CANDIDATE_READY 推进到 ADAPTER_CANDIDATE；不声明远程模型调用"
        ),
    )
    recovery_codex_adapter.add_argument(
        "--state-dir", required=True, help="已通过完整 hash-chain 校验的恢复状态目录"
    )
    recovery_codex_adapter.add_argument(
        "--candidate", required=True, help="恢复状态精确绑定的 operator-candidate.json"
    )
    recovery_codex_adapter.add_argument(
        "--adapter-output",
        required=True,
        help="包含 schema-5 asnumpy-adapter-report.json 与 candidate/ 的已发布目录",
    )
    recovery_codex_adapter.add_argument(
        "--json", action="store_true", help="仅向 stdout 输出机器可读 JSON"
    )

    recovery_adapter_repair = recovery_sub.add_parser(
        "ingest-adapter-repair",
        help=(
            "复验修复后的 official ACLNN AsNumpy overlay，并在记录过集成"
            "失败后于 ADAPTER_CANDIDATE 原地恢复；最多三轮"
        ),
    )
    recovery_adapter_repair.add_argument(
        "--state-dir", required=True, help="已通过完整 hash-chain 校验的恢复状态目录"
    )
    recovery_adapter_repair.add_argument(
        "--candidate", required=True, help="恢复状态精确绑定的 operator-candidate.json"
    )
    recovery_adapter_repair.add_argument(
        "--adapter-output",
        required=True,
        help="包含 repaired_unverified 的 asnumpy-adapter-report.json 与 candidate/",
    )
    recovery_adapter_repair.add_argument(
        "--json", action="store_true", help="仅向 stdout 输出机器可读 JSON"
    )
    recovery_preflight = recovery_sub.add_parser(
        "preflight-integrate-candidate",
        help=(
            "只读校验 adapter、候选、AsNumpy reference 和 full-local-recovery 授权；"
            "不构建、不导入、不执行、不写 RecoveryStore"
        ),
    )
    recovery_preflight.add_argument("--state-dir", required=True, help="已通过完整 hash-chain 校验的恢复状态目录")
    recovery_preflight.add_argument("--candidate", required=True, help="恢复状态精确绑定的 operator-candidate.json")
    recovery_preflight.add_argument("--adapter-output", required=True, help="包含 asnumpy-adapter-report.json 与 candidate/ 的已发布目录")
    recovery_preflight.add_argument("--asnumpy-reference", required=True, help="adapter context plan 绑定的干净固定 AsNumpy Git 工作树")
    recovery_preflight.add_argument("--python", default=None, help="仅计算解释器文件 hash，不会执行解释器")
    recovery_preflight.add_argument("--json", action="store_true", help="仅向 stdout 输出机器可读 JSON")

    recovery_integrate = recovery_sub.add_parser(
        "integrate-candidate",
        help=(
            "把当前 adapter 叠加到固定 AsNumpy 的隔离副本，构建 wheel、在私有"
            " runtime 中加载，并执行确定性 NumPy oracle API gate"
        ),
    )
    recovery_integrate.add_argument(
        "--state-dir", required=True, help="已通过完整 hash-chain 校验的恢复状态目录"
    )
    recovery_integrate.add_argument(
        "--candidate", required=True, help="恢复状态精确绑定的 operator-candidate.json"
    )
    recovery_integrate.add_argument(
        "--adapter-output",
        required=True,
        help="包含 asnumpy-adapter-report.json 与 candidate/ 的已发布目录",
    )
    recovery_integrate.add_argument(
        "--asnumpy-reference",
        required=True,
        help="adapter context plan 绑定的干净固定 AsNumpy Git 工作树",
    )
    recovery_integrate.add_argument(
        "--out", required=True, help="必须为绝对路径的私有 write-once 事务目录"
    )
    recovery_integrate.add_argument(
        "--python", default=None, help="用于构建和隔离 runtime 的 Python 解释器"
    )
    recovery_integrate.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="每一步构建或 gate 的超时秒数（默认 1800）",
    )
    recovery_integrate.add_argument(
        "--json", action="store_true", help="仅向 stdout 输出机器可读 JSON"
    )

    gaps = sub.add_parser(
        "complete-gaps",
        help="检测真正缺失的 NumPy API，并准备或生成未验证的 Ascend C 算子草稿",
    )
    gaps.add_argument("file", help="待分析的 NumPy Python 源文件")
    gaps.add_argument(
        "--gap-request",
        help=(
            "读取 migrate 在缺口/拒绝 fallback 时生成的 gap-request.json；"
            "会在联网或读取远程 Skill 前重新校验源码、调用点、签名和目标绑定"
        ),
    )
    gaps.add_argument(
        "--official-resolution-report",
        help=(
            "由 resolve-gaps 生成的 hash-bound 收据；只有所有 Gap 都记录"
            "installed CANN 与固定 ops-nn 的 clean no_match 时才允许 llm 写新 kernel"
        ),
    )
    gaps.add_argument(
        "--allow-custom-operator-generation",
        action="store_true",
        help=(
            "显式接受 custom OperatorSpec 的未验证 LLM 草稿路由；仅当 "
            "official-resolution.json 明确记录 custom route 时生效，"
            "仍需 LLM/Skill/代理门禁且不可激活"
        ),
    )
    gaps.add_argument(
        "--out",
        default=str(default_gap_dir()),
        help="规格、草稿和审计报告目录（默认 %(default)s）",
    )
    gaps.add_argument(
        "--target-soc",
        default=os.getenv("ASCEND_SOC_VERSION", "unknown"),
        help=(
            "精确目标 SoC（例如 Ascend910B4）；会进入规格、官方解析收据和产物身份"
        ),
    )
    gaps.add_argument(
        "--cann-version",
        default=os.getenv("CANN_VERSION", "unknown"),
        help="目标 CANN 版本；未知时只建议 prepare，不可视为可复用能力",
    )
    gaps.add_argument(
        "--provider",
        choices=["prepare", "llm", "injected"],
        default="prepare",
        help=(
            "prepare 只写规格；llm 通过 llm.json 生成远程草稿；"
            "injected 读取 --response-file 的本地模型候选，不联网且不读取 llm.json"
        ),
    )
    gaps.add_argument(
        "--response-file",
        help=(
            "provider=injected 时的 UTF-8 本地模型响应（JSON 或 NUMPILOT_BUNDLE_V1）；"
            "只读普通文件，最大 3 MiB，候选仍标记为 generated_unverified"
        ),
    )
    gaps.add_argument(
        "--operator-goal",
        choices=["accuracy", "performance"],
        default=None,
        help=(
            "算子补全目标；accuracy 是当前已实现流程；performance 为保留选项，"
            "当前会在联网、读取 Skill 和写输出前明确拒绝。交互终端会询问，"
            "非交互默认 accuracy"
        ),
    )
    gaps.add_argument(
        "--max-attempts",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help="模型输出未通过本地结构校验时的最大尝试次数(最多 3)",
    )
    gaps.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "多缺口 llm 生成时，单个缺口失败后继续处理其余缺口；"
            "最终仍以非零退出，并在 gap-report.json 中记录已脱敏失败状态"
        ),
    )
    gaps.add_argument(
        "--llm-timeout",
        type=int,
        help=(
            "单次模型请求超时秒数（10～3600）；省略时读取 "
            "NUMPILOT_LLM_TIMEOUT，默认 120"
        ),
    )
    gaps.add_argument(
        "--skill-root",
        default=os.getenv("NUMPILOT_CANNBOT_SKILLS", "auto"),
        help=(
            "llm 模式使用的固定 cannbot-skills 根目录；auto 只读显式 "
            "bootstrap/setup 已准备的版本化缓存，绝不在本命令中下载"
        ),
    )
    gaps.add_argument(
        "--offline-skills",
        action="store_true",
        help="兼容旧命令；complete-gaps 现在始终不联网获取 Skill",
    )
    gaps.add_argument(
        "--show-cannbot-skill",
        choices=["none", "summary", "full"],
        default="none",
        help=(
            "展示本次请求实际嵌入的官方 CANNBot 生成上下文；"
            "full 仅用于本地录制/审阅"
        ),
    )
    gaps.add_argument(
        "--asnumpy-reference",
        default=os.getenv("NUMPILOT_ASNUMPY_REFERENCE", ""),
        help="AsNumpy 接入模板目录，仅作为生成契约引用",
    )

    repair = sub.add_parser(
        "repair-draft",
        help="将 CANN 编译/测试诊断回传 LLM，生成新的未验证修复副本",
    )
    repair.add_argument("--spec", required=True, help="签名 draft 对应的 gap-spec.json")
    repair.add_argument("--draft", required=True, help="不可变的原始签名 draft 目录")
    repair.add_argument(
        "--base-candidate",
        help="可选：经 repair report 校验的上一轮候选；只对新增失败文件接力修复",
    )
    repair.add_argument("--out", required=True, help="新修复副本目录，必须不存在")
    repair.add_argument(
        "--generation-request",
        required=True,
        help="原 draft 对应的 GENERATION_REQUEST.md",
    )
    repair.add_argument(
        "--diagnostic-file",
        required=True,
        help="脱敏且有界的 CANN/测试诊断文本",
    )
    repair.add_argument(
        "--path",
        action="append",
        required=True,
        help="要修复的签名项目相对路径；可重复指定",
    )
    repair.add_argument(
        "--max-attempts",
        type=int,
        choices=[1, 2, 3],
        default=2,
    )
    repair.add_argument(
        "--llm-timeout",
        type=int,
        help=(
            "单次模型问答/修复超时秒数（10～3600）；省略时读取 "
            "NUMPILOT_LLM_TIMEOUT，默认 120"
        ),
    )
    repair.add_argument(
        "--repair-profile",
        choices=["compile", "precision", "runtime"],
        default="compile",
        help=(
            "按失败阶段读取对应的外部 CANNBot Skill；默认 compile，"
            "精度/测试失败使用 precision，真机运行失败使用 runtime"
        ),
    )
    repair.add_argument(
        "--skill-root",
        default=os.getenv("NUMPILOT_CANNBOT_SKILLS", "auto"),
    )
    repair.add_argument("--offline-skills", action="store_true")

    seal = sub.add_parser(
        "seal-repair",
        help="复验完整 repair 链并发布新的不可变未验证 Artifact（不调用 LLM）",
    )
    seal.add_argument("--spec", required=True, help="原始 gap-spec.json")
    seal.add_argument("--draft", required=True, help="不可变的原始签名 draft")
    seal.add_argument("--candidate", required=True, help="最终 repair candidate")
    seal.add_argument("--out", required=True, help="新 Artifact 目录，必须不存在")
    seal.add_argument(
        "--skill-root",
        default=os.getenv("NUMPILOT_CANNBOT_SKILLS", "auto"),
        help="repair 时使用的固定 CANNBot checkout；省略时仅查找本地固定缓存",
    )
    seal.add_argument(
        "--offline-skills",
        action="store_true",
        help="兼容 repair-draft；seal-repair 始终离线且绝不下载 Skill",
    )

    adapter = sub.add_parser(
        "generate-asnumpy-adapter",
        help="由已配置模型按候选路由生成隔离的 AsNumpy 接入 overlay",
    )
    adapter.add_argument(
        "--spec", help="custom-OPP 模式：签名 Artifact 对应的 gap-spec.json"
    )
    adapter_source = adapter.add_mutually_exclusive_group(required=True)
    adapter_source.add_argument(
        "--artifact", help="已封存的 remote_llm custom-operator Artifact"
    )
    adapter_source.add_argument(
        "--candidate", help="已物化且绑定 recovery 的 official_binary candidate JSON"
    )
    adapter.add_argument(
        "--recovery-state-dir",
        help="official candidate 模式：处于 CANDIDATE_READY 的恢复状态目录",
    )
    adapter.add_argument(
        "--official-context-gate",
        help=(
            "official_candidate_gap 模式的 context gate 回执；与 --candidate 和 "
            "--recovery-state-dir 一起使用，重新校验 hand-off/GapSpec/官方解析/"
            "AsNumpy context plan 后才允许模型请求"
        ),
    )
    adapter.add_argument(
        "--asnumpy-reference",
        required=True,
        help=(
            "精确目标 AsNumpy 源码树；仅作为只读、hash-bound 的工程接入契约"
        ),
    )
    adapter.add_argument("--out", required=True, help="隔离 overlay 与审计报告目录")
    adapter.add_argument(
        "--max-attempts", type=int, choices=[1, 2, 3], default=3
    )
    adapter.add_argument(
        "--llm-timeout",
        type=int,
        help=(
            "单次模型问答/接入文件生成超时秒数（10～3600）；省略时读取 "
            "NUMPILOT_LLM_TIMEOUT，默认 120"
        ),
    )
    adapter.add_argument(
        "--asnumpy-skill-root",
        "--skill-root",
        dest="asnumpy_skill_root",
        default=None,
        help="NumPilot 自带 AsNumpy 接入 Skill；默认读取仓库内固定目录",
    )
    adapter.add_argument(
        "--offline-skills",
        action="store_true",
        help="兼容旧命令；内置 AsNumpy Skill 从不联网",
    )


    handoff = sub.add_parser(
        "prepare-asnumpy-adapter-handoff",
        help="只读准备 Codex/人工代理的 AsNumpy adapter 交接包，不读取 llm.json",
    )
    handoff.set_defaults(handoff_only=True)
    handoff.add_argument(
        "--spec", help="保留参数；official candidate 模式不使用"
    )
    handoff_source = handoff.add_mutually_exclusive_group(required=True)
    handoff_source.add_argument("--artifact", help="custom-OPP Artifact")
    handoff_source.add_argument("--candidate", help="绑定 recovery 的 official candidate JSON")
    handoff.add_argument("--recovery-state-dir", required=True, help="CANDIDATE_READY recovery 状态目录")
    handoff.add_argument("--official-context-gate", required=True, help="ready_for_official_adapter_model gate 回执")
    handoff.add_argument("--asnumpy-reference", required=True, help="只读 AsNumpy 源码树")
    handoff.add_argument("--out", required=True, help="新建 handoff 目录")
    handoff.add_argument("--asnumpy-skill-root", "--skill-root", dest="asnumpy_skill_root", default=None, help="NumPilot 自带 AsNumpy Skill 根目录")
    handoff.add_argument("--offline-skills", action="store_true", help="兼容参数；handoff 始终不联网")

    validate_handoff = sub.add_parser(
        "validate-asnumpy-adapter-handoff",
        help="只读回收 Codex/人工代理 responses，生成 injected AsNumpy candidate",
    )
    validate_handoff.add_argument("--handoff-dir", required=True, help="prepare 命令生成的 handoff 目录")
    validate_handoff.add_argument("--response-dir", required=True, help="Codex 写入的 responses 根目录")
    validate_handoff.add_argument("--out", required=True, help="新建 injected adapter 输出目录")
    validate_handoff.add_argument("--max-attempts", type=int, choices=[1, 2, 3], default=1)
    validate_handoff.add_argument("--asnumpy-skill-root", "--skill-root", dest="asnumpy_skill_root", default=None, help="NumPilot 自带 AsNumpy Skill 根目录")
    validate_handoff.add_argument("--offline-skills", action="store_true", help="兼容参数；validate 始终不联网")

    adapter_repair = sub.add_parser(
        "repair-asnumpy-adapter",
        help="把 AsNumpy 构建/测试原始诊断交给已配置模型，修复一个 overlay 文件",
    )
    adapter_repair.add_argument(
        "--base-adapter",
        required=True,
        help="上一轮带 asnumpy-adapter-report.json 的模型 overlay 根目录",
    )
    adapter_repair.add_argument(
        "--asnumpy-reference",
        required=True,
        help="当前 AsNumpy 隔离工作树，只读提供构建与生成器契约",
    )
    adapter_repair.add_argument(
        "--diagnostic-file",
        required=True,
        help="原始且有界的 AsNumpy 构建/测试诊断",
    )
    adapter_repair.add_argument(
        "--path",
        required=True,
        help="只能修复上一轮模型 overlay 中的一个相对路径",
    )
    adapter_repair.add_argument("--out", required=True, help="新的隔离修复 overlay")
    adapter_repair.add_argument(
        "--repair-profile",
        choices=["compile", "precision", "runtime"],
        default="compile",
        help=(
            "把失败阶段绑定进提示词和审计报告；三个阶段共用同一内置 "
            "AsNumpy 接入 Skill"
        ),
    )
    adapter_repair.add_argument(
        "--max-attempts", type=int, choices=[1, 2, 3], default=3
    )
    adapter_repair.add_argument(
        "--llm-timeout",
        type=int,
        help="单次模型问答/修复超时秒数（10～3600，默认 120）",
    )
    adapter_repair.add_argument(
        "--asnumpy-skill-root",
        "--skill-root",
        dest="asnumpy_skill_root",
        default=None,
        help="NumPilot 自带 AsNumpy 接入 Skill；默认读取仓库内固定目录",
    )
    adapter_repair.add_argument(
        "--offline-skills",
        action="store_true",
        help="兼容旧命令；内置 AsNumpy Skill 从不联网",
    )

    # Repeat the safe path option on subcommands so both of these ergonomic
    # forms work without changing the long-standing command contracts:
    # ``numpilot --llm-config FILE migrate ...`` and
    # ``numpilot migrate ... --llm-config FILE``.  ``SUPPRESS`` is important:
    # an omitted child option must not overwrite a value parsed by the root.
    _llm_config_parsers = [
        analyze,
        bootstrap,
        mig,
        run_parser,
        interactive,
        verify,
        benchmark_script_parser,
        bc,
        bn,
        wiz,
        doc,
        paths,
        suite,
        check_llm_parser,
        cfg,
        skills,
        resolve_gaps,
        recover_gaps,
        gaps,
        repair,
        seal,
        adapter,
        adapter_repair,
        recovery_ingest,
        recovery_status_parser,
        recovery_materialize,
        recovery_adapter,
    ]
    for _command_parser in _llm_config_parsers:
        _command_parser.add_argument(
            "--llm-config",
            dest="llm_config",
            type=_llm_config_arg,
            default=argparse.SUPPRESS,
            metavar="PATH",
            help=argparse.SUPPRESS,
        )

    args = parser.parse_args(arguments)
    _llm_config_override = getattr(args, "llm_config", None)
    _llm_config_had_previous = "NUMPILOT_LLM_CONFIG_FILE" in os.environ
    _llm_config_previous = os.environ.get("NUMPILOT_LLM_CONFIG_FILE")
    if _llm_config_override:
        # Keep the existing field-by-field environment precedence inside
        # ``load_llm_config`` while making the one-shot CLI choice explicit.
        # The override is scoped to this invocation; ``main`` is also used as
        # an in-process API by notebooks/tests and must not leak a temporary
        # credential path into the next command.
        os.environ["NUMPILOT_LLM_CONFIG_FILE"] = _llm_config_override
    try:
        if args.cmd == "analyze":
            return _cmd_analyze(args)
        if args.cmd in {"verify", "verify-script"}:
            return _cmd_verify_script(args)
        if args.cmd == "replay":
            return _cmd_replay(args)
        if args.cmd in {"benchmark-script", "bench-script"}:
            return _cmd_benchmark_script(args)
        if args.cmd == "bootstrap":
            return _cmd_bootstrap(args)
        if args.cmd in {"run", "pipeline"}:
            return _cmd_run(args)
        if args.cmd in {"interactive-migrate", "interactive-run"}:
            return _cmd_interactive_migrate(args)
        if args.cmd == "check-llm":
            return _cmd_check_llm(args)
        if args.cmd == "configure-llm":
            return _cmd_configure_llm(args)
        if args.cmd == "setup-skills":
            return _cmd_setup_skills(args)
        if args.cmd == "resolve-gaps":
            return _cmd_resolve_gaps(args)
        if args.cmd == "prepare-official-candidate":
            return _cmd_prepare_official_candidate(args)
        if args.cmd == "preflight-official-candidate":
            return _cmd_preflight_official_candidate(args)
        if args.cmd == "prepare-official-adapter-input":
            return _cmd_prepare_official_adapter_input(args)
        if args.cmd in {"validate-official-adapter-context", "official-adapter-generate"}:
            return _cmd_validate_official_adapter_context(args)
        if args.cmd == "materialize-official-candidate-recovery":
            return _cmd_materialize_official_candidate_recovery(args)
        if args.cmd == "recover-gaps":
            return _cmd_recover_gaps(args)
        if args.cmd == "paths":
            return _cmd_paths(args)
        if args.cmd == "check-suite":
            return _cmd_check_suite(args)
        if args.cmd == "complete-gaps":
            return _cmd_complete_gaps(args)
        if args.cmd == "repair-draft":
            return _cmd_repair_draft(args)
        if args.cmd == "seal-repair":
            return _cmd_seal_repair(args)
        if args.cmd == "prepare-asnumpy-adapter-handoff":
            return _cmd_generate_asnumpy_adapter(args)
        if args.cmd == "validate-asnumpy-adapter-handoff":
            return _cmd_validate_asnumpy_adapter_handoff(args)
        if args.cmd == "generate-asnumpy-adapter":
            return _cmd_generate_asnumpy_adapter(args)
        if args.cmd == "repair-asnumpy-adapter":
            return _cmd_repair_asnumpy_adapter(args)
        if args.cmd == "wizard":
            try:
                from numpilot.wizard import run_wizard
            except ModuleNotFoundError as exc:
                if exc.name != "rich" and not (exc.name or "").startswith("rich."):
                    raise
                print(
                    "numpilot wizard 需要可选依赖 rich："
                    "pip install 'numpilot[ui]'",
                    file=sys.stderr,
                )
                return 2
            return run_wizard(
                file=args.file,
                engine=args.engine,
                backend=args.backend,
                out_dir=args.out,
                reports_dir=args.reports_dir,
            )
        if args.cmd == "doctor":
            return _cmd_doctor(args)
        if args.cmd == "bench-cpu":
            return run_cpu_baseline(args.file, args.out, parse_cpu_threads(args.cpu_threads))
        if args.cmd == "bench-npu":
            if args.engine == "llm":
                if _load_llm_or_report() is None:
                    return 4
                if _llm_proxy_for_remote_call("NPU 评测迁移门禁") is None:
                    return 4
            backend, analysis_profiles = _qualified_runtime_inputs(args, "npu")
            _preflight("npu", args.skip_env_check)
            return run_npu_compare(args.file, args.out, engine=_make_engine(args.engine),
                                   threshold=args.threshold,
                                   allow_cpu_fallback=args.allow_cpu_fallback,
                                   backend=backend,
                                   analysis_profiles=analysis_profiles)
        return _cmd_migrate(args)
    except LlmError as e:
        print(f"\n[LLM] 迁移终止:{e}", file=sys.stderr)
        print("[LLM] 已按 strict 语义直接终止(未回退 AST)。", file=sys.stderr)
        return 4
    except LlmConfigError as e:
        print(f"[LLM] 配置或代理门禁失败:{e}", file=sys.stderr)
        print("[LLM] 未发起或已终止远端模型调用。", file=sys.stderr)
        return 4
    except RuntimeActivationConfigError as e:
        print(f"[qualification runtime] {e}", file=sys.stderr)
        return 2
    except ScriptMigrationError as e:
        print(f"[普通脚本迁移] {e}", file=sys.stderr)
        return 2
    except CannbotSkillError as e:
        print(f"[CANNBot skill] {e}", file=sys.stderr)
        if (
            getattr(args, "cmd", None) == "complete-gaps"
            and getattr(args, "provider", None) == "llm"
        ):
            print(
                "[生成门禁 2/2] fail · CANNBot Skill 准备或读取失败",
                file=sys.stderr,
            )
            print(
                "[算子生成] actual_remote_generation_call=no；"
                "可能已执行 LLM readiness 问答，但没有开始算子生成。",
                file=sys.stderr,
            )
        elif getattr(args, "cmd", None) == "repair-draft":
            print(
                "[修复门禁 2/2] fail · CANNBot Skill 准备或读取失败",
                file=sys.stderr,
            )
            print(
                "[算子修复] actual_remote_repair_call=no；"
                "可能已执行 LLM readiness 问答，但没有开始模型修复。",
                file=sys.stderr,
            )
        elif getattr(args, "cmd", None) == "generate-asnumpy-adapter":
            print(
                "[接入门禁 2/2] fail · CANNBot Skill 准备或读取失败",
                file=sys.stderr,
            )
            print(
                "[AsNumpy 接入生成] actual_remote_call=no；"
                "可能已执行 LLM readiness 问答，但没有开始 overlay 生成。",
                file=sys.stderr,
            )
        elif getattr(args, "cmd", None) == "repair-asnumpy-adapter":
            print(
                "[接入修复门禁 2/2] fail · CANNBot Skill 准备或读取失败",
                file=sys.stderr,
            )
            print(
                "[AsNumpy 接入修复] actual_remote_call=no；"
                "可能已执行 LLM readiness 问答，但没有开始修复。",
                file=sys.stderr,
            )
        return 2
    except AsnumpyIntegrationSkillError as e:
        print(f"[AsNumpy integration skill] {e}", file=sys.stderr)
        command = getattr(args, "cmd", None)
        if command == "generate-asnumpy-adapter":
            print(
                "[接入门禁 2/2] fail · NumPilot AsNumpy 案例 Skill 读取失败",
                file=sys.stderr,
            )
            print(
                "[AsNumpy 接入生成] actual_remote_call=no；"
                "可能已执行 LLM readiness 问答，但没有开始 overlay 生成。",
                file=sys.stderr,
            )
        elif command == "repair-asnumpy-adapter":
            print(
                "[接入修复门禁 2/2] fail · NumPilot AsNumpy 案例 Skill 读取失败",
                file=sys.stderr,
            )
            print(
                "[AsNumpy 接入修复] actual_remote_call=no；"
                "可能已执行 LLM readiness 问答，但没有开始修复。",
                file=sys.stderr,
            )
        return 2
    finally:
        if _llm_config_override:
            if _llm_config_had_previous:
                os.environ["NUMPILOT_LLM_CONFIG_FILE"] = (
                    _llm_config_previous or ""
                )
            else:
                os.environ.pop("NUMPILOT_LLM_CONFIG_FILE", None)


if __name__ == "__main__":
    sys.exit(main())
