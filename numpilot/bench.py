"""两个独立的性能对比程序(分进程,互不干扰线程设置):

- run_cpu_baseline: 纯 numpy 基线。进程内已在 __main__ 早期把 BLAS 钉到 1 核
  (本机 scipy-openblas 默认吃满 640 核,不限核则 numpy 虚快、对 NPU 单卡不公平)。
  跑原始程序 forward,计时,把输出+耗时存到基线文件供 bench-npu 读取。

- run_npu_compare: 放开所有线程限制,把程序迁移到 asnumpy 在 NPU 上跑,计时;
  就地用 numpy 参考做数值校验;若存在 CPU 基线则算 加速比 = t_cpu(1核) / t_npu(满线程)。

两者分属不同进程,故 CPU 侧的钉核不会波及 NPU 侧的 CANN/OpenMP。
"""
import json
import os
import sys
import time

import numpy as np

from numpilot.agents.analysis import AnalysisAgent
from numpilot.agents.decision import DecisionAgent
from numpilot.agents.migration import MigrationAgent
from numpilot.backends.base import select_backend

_TOL = {"float32": 1e-4, "float16": 1e-2}
_BASELINE_JSON = "cpu_baseline.json"
_BASELINE_NPY = "cpu_baseline.npy"


def _tol_for(dtype):
    return _TOL.get(np.dtype(dtype).name, 0.0)


def _net_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def _load_namespace(source):
    ns = {}
    exec(compile(source, "<bench>", "exec"), ns)
    return ns


def _to_numpy(x):
    return np.asarray(x.to_numpy() if hasattr(x, "to_numpy") else x)


def _median_ms(fn, args, sync=None, warmup=3, iters=10):
    """fn(*args) 的中位耗时(ms)。sync(result) 强制异步后端(NPU)完成后再停表。"""
    for _ in range(warmup):
        r = fn(*args)
        if sync is not None:
            sync(r)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        r = fn(*args)
        if sync is not None:
            sync(r)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1e3


def _blas_threads_now():
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        v = os.environ.get(k)
        if v and v.isdigit() and int(v) > 0:
            return int(v)
    return None


# ───────────────────────── ① CPU 基线 ─────────────────────────
def run_cpu_baseline(file, out_dir, cpu_threads):
    with open(file, encoding="utf-8") as f:
        source = f.read()
    ns = _load_namespace(source)
    inputs = ns["build_inputs"]()
    out = _to_numpy(ns["forward"](*inputs))
    t_cpu = _median_ms(ns["forward"], inputs)     # numpy 自身同步

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, _BASELINE_NPY), out)
    threads = _blas_threads_now()                  # 实际生效的 BLAS 线程数
    meta = {
        "file": file, "net": _net_name(file),
        "t_cpu_ms": t_cpu, "n_elements": int(out.size),
        "cpu_threads": threads, "dtype": str(out.dtype),
    }
    with open(os.path.join(out_dir, _BASELINE_JSON), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    net = _net_name(file)
    core = f"{threads} 核" if threads else f"全核({os.cpu_count()} 核)"
    print(f"── NumPilot CPU 基线 · {net} " + "─" * max(4, 30 - len(net)))
    print(f"[CPU] numpy@CPU · 限 {core}(本机共 {os.cpu_count()} 核)· BLAS=scipy-openblas")
    print(f"[性能] {out.size:,} 元素   numpy forward = {t_cpu:.3f} ms")
    print(f"[基线] 已存 {os.path.join(out_dir, _BASELINE_NPY)}(供 ② bench-npu 对比)")
    return 0


# ───────────────────────── ② NPU 对比 ─────────────────────────
def run_npu_compare(file, out_dir, engine=None, threshold=1_000_000):
    with open(file, encoding="utf-8") as f:
        source = f.read()

    analysis = AnalysisAgent().analyze(source, file)
    net = _net_name(file)
    print(f"── NumPilot NPU 对比 · {net} " + "─" * max(4, 30 - len(net)))
    if not analysis.migratable:
        print(f"[分析] 命中 {len(analysis.unsupported)} 个覆盖集缺口 ✗")
        print("[缺口] " + ", ".join(analysis.unsupported) + "  (无法迁移)")
        return 1

    plan = DecisionAgent(threshold).plan(analysis)
    migration = MigrationAgent(engine).migrate(source, plan)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "migrated.py"), "w", encoding="utf-8") as f:
        f.write(migration.migrated_source)
    total = migration.total_rewrites
    items = "  ".join(f"{op}×{n}" for op, n in
                      sorted(migration.rewrites.items(), key=lambda kv: (-kv[1], kv[0])))
    print(f"[迁移] 引擎={migration.engine} · 替换 {total} 处 np.* → ap.*   {items}")

    # numpy 参考(仅为正确性,不计时;此进程满线程)
    orig_ns = _load_namespace(source)
    inputs = orig_ns["build_inputs"]()
    ref = _to_numpy(orig_ns["forward"](*inputs))

    backend = select_backend("npu")
    ap_module = backend.get_module()
    saved = sys.modules.get("asnumpy")
    sys.modules["asnumpy"] = ap_module
    try:
        mig_ns = _load_namespace(migration.migrated_source)
        forward = mig_ns["forward"]
        dev_inputs = tuple(backend.to_device(_to_numpy(x)) for x in inputs)
        out_npu = _to_numpy(backend.to_host(forward(*dev_inputs)))
        t_npu = _median_ms(forward, dev_inputs, sync=backend.to_host)
    finally:
        if saved is not None:
            sys.modules["asnumpy"] = saved
        else:
            sys.modules.pop("asnumpy", None)

    print(f"[执行] 后端=npu · asnumpy {getattr(ap_module, '__version__', '?')} ✓ 真机")
    print(f"       {getattr(ap_module, '__file__', '?')}")

    tol = _tol_for(ref.dtype)
    denom = np.max(np.abs(ref))
    max_rel = float(np.max(np.abs(ref - out_npu)) / denom) if denom > 0 else 0.0
    ok = max_rel <= tol
    print(f"[校验] 数值一致 {'✓' if ok else '✗'}  max_rel_diff={max_rel:.2e}  (tol={tol:.0e}, "
          "对照:本进程内 numpy 满线程参考)")
    print(f"[性能] {ref.size:,} 元素   asnumpy@npu = {t_npu:.3f} ms")

    # 读 ① 的 CPU 基线算加速比
    base_path = os.path.join(out_dir, _BASELINE_JSON)
    if os.path.exists(base_path):
        with open(base_path, encoding="utf-8") as f:
            base = json.load(f)
        t_cpu = base["t_cpu_ms"]
        threads = base.get("cpu_threads")
        core = f"{threads} 核" if threads else "全核"
        speedup = t_cpu / t_npu if t_npu > 0 else float("nan")
        faster = "asnumpy 更快" if speedup > 1 else "numpy 更快"
        print(f"[对比] numpy@CPU({core}) {t_cpu:.3f} ms  vs  asnumpy@npu {t_npu:.3f} ms")
        print(f"       → 加速比 {speedup:.2f}x ({faster})   [基线来自 ① bench-cpu]")
        if base.get("n_elements") != int(ref.size):
            print("       ⚠️ 基线规模与当前不一致,请对同一 file 先跑 bench-cpu")
    else:
        print(f"[对比] 未找到 CPU 基线 {base_path}")
        print(f"       请先跑:python -m numpilot bench-cpu {file} --out {out_dir}")
    return 0 if ok else 2
