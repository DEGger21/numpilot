"""环境自检:用框架前检查本机是否具备迁移/执行条件。

参考 cannbot ascendc-env-check 的"分项检查 + 汇总 error/warn"风格,但聚焦
NumPilot 真正需要的东西:Python 版本、numpy、CANN Toolkit(set_env)、NPU 设备、
以及最关键的 **asnumpy 可导入性**。

用法:
- `Env.check(backend="npu")` 返回 `EnvReport`,含逐项 CheckItem 与 error/warn 计数。
- CLI `python -m numpilot doctor` 打印全部分项 + 汇总。
- `migrate` 前置:npu 后端缺硬依赖(CANN/asnumpy)→ 判 error 拦截退出;
  shim 后端同样缺失只软告警(warn),不阻断。
"""
import os
import importlib.util
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

OK = "ok"
WARN = "warn"
ERROR = "error"

_MARK = {OK: "✓", WARN: "⚠", ERROR: "✗"}


@dataclass
class CheckItem:
    name: str
    status: str          # ok | warn | error
    detail: str = ""


@dataclass
class EnvReport:
    backend: str
    items: list = field(default_factory=list)

    def add(self, name, status, detail=""):
        self.items.append(CheckItem(name, status, detail))

    @property
    def errors(self):
        return [i for i in self.items if i.status == ERROR]

    @property
    def warnings(self):
        return [i for i in self.items if i.status == WARN]

    @property
    def ok(self):
        return not self.errors

    @property
    def status(self):
        """Return the aggregate status used by machine-readable consumers."""
        if self.errors:
            return ERROR
        if self.warnings:
            return WARN
        return OK

    def to_dict(self):
        """Return a detached, JSON-safe report without terminal decorations.

        ``doctor --json`` and CodeArts/CI adapters consume this shape.  Keep
        the human ``render`` method separate so ANSI/progress presentation can
        never contaminate a machine-readable stdout contract.
        """
        return {
            "schema_version": 1,
            "record_type": "numpilot_environment_report",
            "backend": self.backend,
            "status": self.status,
            "ok": bool(self.ok),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "items": [
                {
                    "name": item.name,
                    "status": item.status,
                    "detail": item.detail,
                }
                for item in self.items
            ],
        }

    def render(self) -> str:
        lines = ["NumPilot 环境自检(backend=%s)" % self.backend, "=" * 40]
        for it in self.items:
            lines.append("%s %s%s" % (
                _MARK.get(it.status, "?"), it.name,
                (" — " + it.detail) if it.detail else "",
            ))
        lines.append("=" * 40)
        if self.errors:
            lines.append("✗ 发现 %d 个错误、%d 个警告" % (len(self.errors), len(self.warnings)))
        elif self.warnings:
            lines.append("⚠ 发现 %d 个警告(可继续,建议修复)" % len(self.warnings))
        else:
            lines.append("✓ 环境检查通过")
        return "\n".join(lines)


def _run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except (OSError, subprocess.SubprocessError):
        return -1, ""


_NPU_SMI_NO_DEVICE = re.compile(
    r"\bno\s+"
    r"(?:(?:ascend\s*(?:(?:310|610|910)"
    r"(?:\s*[A-Za-z][A-Za-z0-9_-]*)?)?|npu)\s+)?"
    r"devices?\s+(?:(?:is|are|was|were)\s+)?"
    r"(?:found|detected|available|present)\b",
    re.IGNORECASE,
)
_NPU_SMI_HEALTH = re.compile(r"\b(?:OK|Healthy|Normal)\b", re.IGNORECASE)
_NPU_SMI_UNHEALTHY = re.compile(
    r"\b(?:Not\s+OK|Unhealthy|Abnormal|Warning|Warn|Error|Fault|Critical|Offline)\b",
    re.IGNORECASE,
)
_NPU_SMI_MODEL = (
    r"(?:Ascend\s*)?"
    r"(?P<family>310|610|910)"
    r"(?P<attached>[A-Za-z][A-Za-z0-9_-]*)?"
    r"(?:\s+(?P<spaced>B[0-9]+|P[0-9]+|Pro|Premium|Lite))?"
)
_NPU_SMI_INDEXED_MODEL = re.compile(
    r"^\s*(?:"
    r"(?:\|\s*)?\d+\s+(?:\|\s*)?" + _NPU_SMI_MODEL
    + r"|NPU\s*(?:ID\s*)?[:#]?\s*\d+\s*[:|,-]?\s*"
    + _NPU_SMI_MODEL.replace("?P<family>", "?P<npu_family>")
    .replace("?P<attached>", "?P<npu_attached>")
    .replace("?P<spaced>", "?P<npu_spaced>")
    + r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def _canonical_soc_model(family: str, suffix: str) -> str:
    normalized = suffix.strip().replace("_", "").replace("-", "")
    known_words = {
        "premium": "Premium",
        "pro": "Pro",
        "lite": "Lite",
    }
    normalized = known_words.get(normalized.lower(), normalized.upper())
    return "Ascend%s%s" % (family, normalized)


def _npu_smi_healthy_models(output: str):
    """Return canonical model names from healthy ``npu-smi info`` rows."""

    if (
        not isinstance(output, str)
        or not output.strip()
        or len(output) > 1024 * 1024
        or _NPU_SMI_NO_DEVICE.search(output)
    ):
        return ()

    models = set()
    for line in output.splitlines()[:4096]:
        if (
            len(line) > 8192
            or not _NPU_SMI_HEALTH.search(line)
            or _NPU_SMI_UNHEALTHY.search(line)
        ):
            continue
        match = _NPU_SMI_INDEXED_MODEL.search(line)
        if match is None:
            continue
        family = match.group("family") or match.group("npu_family")
        suffix = (
            match.group("attached")
            or match.group("spaced")
            or match.group("npu_attached")
            or match.group("npu_spaced")
            or ""
        )
        models.add(_canonical_soc_model(family, suffix))
    return tuple(sorted(models))


def _parse_npu_smi_soc_model(output: str):
    """Return one healthy canonical SoC model, failing closed on ambiguity."""

    models = _npu_smi_healthy_models(output)
    return models[0] if len(models) == 1 else None


def _probe_npu_smi_soc_model():
    """Read the healthy SoC model through ``npu-smi info`` only."""

    executable = shutil.which("npu-smi")
    if not executable:
        return None
    code, output = _run([executable, "info"])
    if code != 0:
        return None
    return _parse_npu_smi_soc_model(output)


def _npu_smi_has_device(output: str) -> bool:
    """Recognise both legacy and current ``npu-smi info`` device tables.

    Older releases printed ``Ascend`` in the model name.  Recent 910B4
    releases use a compact two-line device row whose model cell is only
    ``910B4``.  Checking a product word therefore reports a healthy device as
    missing.  Require a successful command separately and accept only a
    recognisable table/model row here; headings and ``No running processes``
    alone are not device evidence.
    """

    return bool(_npu_smi_healthy_models(output))


_CANN_PATH_VARIABLES = (
    "PATH",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
    "CMAKE_PREFIX_PATH",
)


def _cann_toolkit_root(entry: str):
    """Extract only toolkit-owned roots, excluding driver/NNAL/custom paths."""

    parts = Path(entry).expanduser().parts
    for index, part in enumerate(parts):
        lowered = part.lower()
        has_ascend_parent = any(
            parent.lower() == "ascend" for parent in parts[:index]
        )
        if has_ascend_parent and (
            lowered == "cann" or lowered.startswith(("cann-", "cann_"))
        ):
            return Path(*parts[:index + 1])
        if (
            has_ascend_parent
            and lowered == "ascend-toolkit"
            and index + 1 < len(parts)
        ):
            return Path(*parts[:index + 2])
    return None


def _audit_cann_paths(toolkit: str):
    """Return severity/detail for stale, duplicate, alias and devlib entries."""

    active_text = os.path.abspath(os.path.expanduser(toolkit)) if toolkit else ""
    active_real = os.path.realpath(active_text) if active_text else ""
    devlibs = []
    conflicts = []
    aliases = []
    duplicates = []
    for variable in _CANN_PATH_VARIABLES:
        entries = [entry for entry in os.getenv(variable, "").split(":") if entry]
        seen = set()
        for index, entry in enumerate(entries):
            location = "%s[%d]=%s" % (variable, index, entry)
            normalized_entry = os.path.normpath(
                os.path.abspath(os.path.expanduser(entry))
            )
            if normalized_entry in seen:
                duplicates.append(location)
            seen.add(normalized_entry)
            root = _cann_toolkit_root(entry)
            if root is None or not active_real:
                continue
            if any(part.lower() == "devlib" for part in Path(entry).parts):
                devlibs.append(location)
            root_text = os.path.abspath(str(root))
            root_real = os.path.realpath(root_text)
            if root_real != active_real:
                conflicts.append("%s -> %s" % (location, root_text))
            elif root_text != active_text:
                aliases.append("%s -> %s" % (location, root_text))

    hard = []
    soft = []
    if devlibs:
        hard.append("devlib=%s" % ",".join(dict.fromkeys(devlibs)))
    if conflicts:
        hard.append("冲突 toolkit=%s" % ",".join(dict.fromkeys(conflicts)))
    if duplicates:
        soft.append("重复=%s" % ",".join(dict.fromkeys(duplicates)))
    if aliases:
        soft.append("同根别名=%s" % ",".join(dict.fromkeys(aliases)))
    return hard, soft


class Env:
    @staticmethod
    def check(backend: str = "shim") -> EnvReport:
        """按后端语义做环境自检。

        npu 后端:CANN Toolkit / asnumpy 属硬依赖,缺失判 error(拦截)。
        shim 后端:同类缺失仅 warn(离线也能跑 numpy 仿真)。
        """
        hard = (backend == "npu")     # 硬依赖是否阻断
        miss = ERROR if hard else WARN
        rep = EnvReport(backend=backend)
        toolkit = os.getenv("ASCEND_TOOLKIT_HOME") or os.getenv("ASCEND_HOME_PATH")

        hard_path_issues, soft_path_issues = _audit_cann_paths(toolkit or "")
        if hard_path_issues:
            path_status = ERROR if hard else WARN
            path_detail = "; ".join((*hard_path_issues, *soft_path_issues))
        elif soft_path_issues:
            path_status = WARN
            path_detail = "; ".join(soft_path_issues)
        else:
            path_status = OK
            path_detail = "未发现冲突、重复或 devlib"
        rep.add("CANN 动态库路径", path_status, path_detail[:4096])

        # 1) Python 版本(asnumpy 用 PEP 604 联合类型,需 3.10+)
        v = sys.version_info
        if v >= (3, 10):
            rep.add("Python", OK, "%d.%d.%d" % (v.major, v.minor, v.micro))
        else:
            # 仅 npu 路径强依赖 asnumpy→3.10;shim 路径 3.9 也能跑
            rep.add("Python", miss if hard else WARN,
                    "当前 %d.%d.%d;asnumpy 需 ≥3.10(PEP 604 类型注解)"
                    % (v.major, v.minor, v.micro))

        # 2) numpy(迁移与验证两侧都需要)
        try:
            import numpy as _np
            rep.add("numpy", OK, _np.__version__)
        except ImportError:
            rep.add("numpy", ERROR, "未安装 numpy(迁移/验证均依赖)")

        # shim 不触碰 CANN/NPU/asnumpy 运行时，避免离线测试仅因探测而初始化 ACL。
        if not hard:
            rep.add("CANN Toolkit", OK, "shim 后端不需要")
            rep.add("NPU 设备", OK, "shim 后端不访问设备")
            try:
                installed = importlib.util.find_spec("asnumpy") is not None
            except (ImportError, ValueError):
                installed = False
            rep.add(
                "asnumpy",
                OK if installed else WARN,
                "已安装但 shim 不加载" if installed else "未安装；shim 后端不需要",
            )
            return rep

        # 3) CANN Toolkit(npu 后端硬依赖)
        if toolkit and os.path.isdir(toolkit):
            rep.add("CANN Toolkit", OK, toolkit)
            opp = os.getenv("ASCEND_OPP_PATH")
            if opp and os.path.isdir(opp):
                rep.add("ASCEND_OPP_PATH", OK, opp)
            else:
                rep.add("ASCEND_OPP_PATH", miss,
                        "未设置/无效;需 source set_env.sh(运行态算子依赖)")
        else:
            rep.add("CANN Toolkit", miss,
                    "未探测到 ASCEND_TOOLKIT_HOME;请先 "
                    "`source /usr/local/Ascend/ascend-toolkit/set_env.sh`")

        # 4) NPU 设备(npu-smi)
        smi = shutil.which("npu-smi")
        if smi:
            code, out = _run([smi, "info"])
            if code == 0 and _npu_smi_has_device(out):
                rep.add("NPU 设备", OK, "npu-smi 可用,检测到 Ascend 设备")
            else:
                rep.add("NPU 设备", miss, "npu-smi 存在但未返回设备信息")
        else:
            rep.add("NPU 设备", miss if hard else WARN, "未找到 npu-smi 命令")

        # 5) asnumpy 可导入性(npu 后端的关键硬依赖)
        try:
            import asnumpy as _ap
            ver = getattr(_ap, "__version__", "unknown")
            # 真正触发一次 ap.ones,确认运行期(而非仅 import)可用
            try:
                probe = _ap.ones((2, 2), dtype=_ap.float32)
                probe.to_numpy()
                del probe
                rep.add("asnumpy", OK, "可导入且可运行(version=%s)" % ver)
            except Exception as e:  # noqa: BLE001
                rep.add("asnumpy", miss,
                        "可 import 但运行期报错:%s" % (str(e).splitlines()[0][:120]))
        except Exception as e:  # import 本身也可能因 Python/ACL/二进制不兼容失败
            rep.add(
                "asnumpy",
                miss,
                "导入失败:%s: %s"
                % (type(e).__name__, str(e).splitlines()[0][:120]),
            )

        return rep
