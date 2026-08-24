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


def _npu_smi_has_device(output: str) -> bool:
    """Recognise both legacy and current ``npu-smi info`` device tables.

    Older releases printed ``Ascend`` in the model name.  Recent 910B4
    releases use a compact two-line device row whose model cell is only
    ``910B4``.  Checking a product word therefore reports a healthy device as
    missing.  Require a successful command separately and accept only a
    recognisable table/model row here; headings and ``No running processes``
    alone are not device evidence.
    """

    if not isinstance(output, str) or not output.strip():
        return False

    # npu-smi may echo the requested product in a negative result, for example
    # ``No Ascend910 devices found``.  Treat explicit no-device statements as
    # authoritative before looking for a model token anywhere else.
    no_device = re.compile(
        r"\bno\s+"
        r"(?:(?:ascend\s*(?:(?:310|610|910)"
        r"(?:\s*[A-Za-z][A-Za-z0-9_-]*)?)?|npu)\s+)?"
        r"devices?\s+(?:(?:is|are|was|were)\s+)?"
        r"(?:found|detected|available|present)\b",
        re.IGNORECASE,
    )
    if no_device.search(output):
        return False

    # Require a device-index prefix, an explicit supported model token and a
    # health-state token on the same line.  This accepts both compact table
    # rows (``| 2 910B4 | OK``) and legacy/current named rows such as
    # ``0 Ascend910 Premium OK`` or ``NPU 0: Ascend 910B4 Health: OK`` while
    # rejecting headings, documentation and model names in error prose.
    model_token = (
        r"(?:Ascend\s*)?(?:310|610|910)"
        r"(?:\s*[A-Za-z][A-Za-z0-9_-]*)?"
    )
    indexed_model = re.compile(
        r"^\s*(?:"
        r"(?:\|\s*)?\d+\s+(?:\|\s*)?" + model_token
        + r"|NPU\s*(?:ID\s*)?[:#]?\s*\d+\s*[:|,-]?\s*"
        + model_token
        + r")(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    health = re.compile(
        r"\b(?:OK|Healthy|Normal)\b",
        re.IGNORECASE,
    )
    return any(
        indexed_model.search(line) and health.search(line)
        for line in output.splitlines()
    )


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
        toolkit = os.getenv("ASCEND_TOOLKIT_HOME") or os.getenv("ASCEND_HOME_PATH")
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
