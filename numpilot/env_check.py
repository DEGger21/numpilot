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

        # 3) CANN Toolkit(仅 npu 后端为硬依赖)
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
            if code == 0 and "Ascend" in out:
                n = out.count("Ascend910") or out.count("Ascend")
                rep.add("NPU 设备", OK, "npu-smi 可用,检测到 Ascend 设备")
            else:
                rep.add("NPU 设备", miss, "npu-smi 存在但未返回设备信息")
        else:
            rep.add("NPU 设备", miss if hard else WARN, "未找到 npu-smi 命令")

        # 5) asnumpy 可导入性(npu 后端的关键硬依赖)
        try:
            import asnumpy as _ap  # noqa: F401
            ver = getattr(_ap, "__version__", "unknown")
            # 真正触发一次 ap.ones,确认运行期(而非仅 import)可用
            try:
                _ap.ones((2, 2), dtype=_ap.float32)
                rep.add("asnumpy", OK, "可导入且可运行(version=%s)" % ver)
            except Exception as e:  # noqa: BLE001
                rep.add("asnumpy", miss,
                        "可 import 但运行期报错:%s" % (str(e).splitlines()[0][:120]))
        except ImportError:
            rep.add("asnumpy", miss,
                    "未安装 asnumpy;npu 后端需先 build/install(见 asnumpy README)")

        return rep
