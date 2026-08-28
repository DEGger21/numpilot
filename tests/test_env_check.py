import sys

import pytest

import numpilot.env_check as env_check
from numpilot.env_check import (
    Env,
    _npu_smi_has_device,
    _parse_npu_smi_soc_model,
    _probe_npu_smi_soc_model,
)


def test_shim_environment_check_does_not_import_asnumpy(monkeypatch):
    saved = sys.modules.pop("asnumpy", None)
    try:
        report = Env.check("shim")
        assert report.ok
        assert "asnumpy" not in sys.modules
        assert any(item.name == "NPU 设备" and "不访问" in item.detail for item in report.items)
    finally:
        if saved is not None:
            sys.modules["asnumpy"] = saved


def test_npu_smi_device_detection_accepts_current_910b4_table():
    output = """\
| NPU   Name                | Health        | Power(W) |
| 2     910B4               | OK            | 95.3     |
| 0                         | 0000:C2:00.0  | 0        |
"""
    assert _npu_smi_has_device(output)


def test_npu_smi_device_detection_accepts_legacy_ascend_name():
    assert _npu_smi_has_device("0 Ascend910 Premium OK")


@pytest.mark.parametrize(
    "output",
    (
        "| 0     Ascend 910B        | OK      | 86.0 |",
        "| 0     Ascend 910B4       | Healthy | 86.0 |",
        "NPU 0: Ascend 910B4 Health: OK",
    ),
)
def test_npu_smi_device_detection_accepts_spaced_910b_models(output):
    assert _npu_smi_has_device(output)


@pytest.mark.parametrize(
    "output",
    (
        "No Ascend910 devices found",
        "No Ascend 910B devices were found",
        "No NPU devices detected",
        "No devices available",
        "Supported product: Ascend 910B; parser status OK",
        "0 Ascend 910B device initialization failed",
        "| 0 Ascend 910B | Warning |",
        "| 0 Ascend 910B | Alarm |",
        "| 0 Ascend 910B | Unknown |",
    ),
)
def test_npu_smi_device_detection_rejects_negative_or_non_device_prose(output):
    assert not _npu_smi_has_device(output)


def test_npu_smi_device_detection_rejects_empty_device_table():
    output = """\
| NPU   Name                | Health        | Power(W) |
| No running processes found in NPU 2                    |
"""
    assert not _npu_smi_has_device(output)


@pytest.mark.parametrize(
    ("output", "expected"),
    (
        ("| 2     910B4 | OK | 95.3 |", "Ascend910B4"),
        ("| 0     Ascend 910B | Healthy | 86.0 |", "Ascend910B"),
        ("NPU 0: Ascend 910B4 Health: Normal", "Ascend910B4"),
        ("0 Ascend910 Premium OK", "Ascend910Premium"),
    ),
)
def test_npu_smi_soc_parser_returns_canonical_healthy_model(output, expected):
    assert _parse_npu_smi_soc_model(output) == expected


@pytest.mark.parametrize(
    "output",
    (
        "No Ascend 910B4 devices found",
        "| 0 910B4 | Warning |",
        "| 0 910B4 | Not OK |",
        "NPU 0: Ascend 910B4 Health: Abnormal",
        "| NPU Name | Health |",
        "| 0 910B4 | OK |\n| 1 310P | OK |",
        "",
    ),
)
def test_npu_smi_soc_parser_rejects_unhealthy_empty_or_ambiguous_output(output):
    assert _parse_npu_smi_soc_model(output) is None


def test_npu_smi_soc_probe_uses_read_only_info_command(monkeypatch):
    calls = []
    monkeypatch.setattr(env_check.shutil, "which", lambda name: "/sbin/" + name)

    def fake_run(command):
        calls.append(command)
        return 0, "| 0 910B4 | OK |"

    monkeypatch.setattr(env_check, "_run", fake_run)

    assert _probe_npu_smi_soc_model() == "Ascend910B4"
    assert calls == [["/sbin/npu-smi", "info"]]


def test_npu_smi_soc_probe_fails_closed_on_command_error(monkeypatch):
    monkeypatch.setattr(env_check.shutil, "which", lambda _name: "/sbin/npu-smi")
    monkeypatch.setattr(env_check, "_run", lambda _command: (1, "910B4 OK"))

    assert _probe_npu_smi_soc_model() is None


def test_npu_environment_check_uses_current_npu_smi_format(monkeypatch):
    class FakeArray:
        def to_numpy(self):
            return None

    class FakeAsNumpy:
        __version__ = "test"
        float32 = object()

        @staticmethod
        def ones(_shape, dtype=None):
            return FakeArray()

    monkeypatch.setenv("ASCEND_HOME_PATH", "/tmp")
    monkeypatch.setenv("ASCEND_OPP_PATH", "/tmp")
    for name in ("PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "CMAKE_PREFIX_PATH"):
        monkeypatch.setenv(name, "/usr/bin:/bin" if name == "PATH" else "")
    monkeypatch.setattr(env_check.shutil, "which", lambda _name: "/usr/bin/npu-smi")
    monkeypatch.setattr(
        env_check,
        "_run",
        lambda _cmd: (
            0,
            "| 2     910B4               | OK            | 95.3 |\n",
        ),
    )
    monkeypatch.setitem(sys.modules, "asnumpy", FakeAsNumpy())

    report = Env.check("npu")

    assert report.ok
    assert any(
        item.name == "NPU 设备" and item.status == env_check.OK
        for item in report.items
    )


def _stub_healthy_npu(monkeypatch, tmp_path):
    class FakeArray:
        def to_numpy(self):
            return None

    class FakeAsNumpy:
        __version__ = "test"
        float32 = object()

        @staticmethod
        def ones(_shape, dtype=None):
            return FakeArray()

    root = tmp_path / "Ascend" / "cann-9.0.0"
    opp = root / "opp"
    opp.mkdir(parents=True)
    monkeypatch.setenv("ASCEND_TOOLKIT_HOME", str(root))
    monkeypatch.setenv("ASCEND_HOME_PATH", str(root))
    monkeypatch.setenv("ASCEND_OPP_PATH", str(opp))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("PYTHONPATH", "")
    monkeypatch.setenv("CMAKE_PREFIX_PATH", "")
    monkeypatch.setattr(env_check.shutil, "which", lambda _name: "/usr/bin/npu-smi")
    monkeypatch.setattr(
        env_check,
        "_run",
        lambda _cmd: (0, "| 2     910B4 | OK | 95.3 |\n"),
    )
    monkeypatch.setitem(sys.modules, "asnumpy", FakeAsNumpy())
    return root


def test_npu_environment_rejects_devlib_and_a_different_toolkit_root(
    monkeypatch, tmp_path
):
    root = _stub_healthy_npu(monkeypatch, tmp_path)
    stale = root.parent / "cann-8.5.0"
    monkeypatch.setenv(
        "LD_LIBRARY_PATH",
        ":".join((str(root / "lib64"), str(root / "devlib"), str(stale / "lib64"))),
    )

    report = Env.check("npu")

    item = next(item for item in report.items if item.name == "CANN 动态库路径")
    assert item.status == env_check.ERROR
    assert "devlib" in item.detail
    assert "冲突" in item.detail
    assert "LD_LIBRARY_PATH[1]" in item.detail
    assert "LD_LIBRARY_PATH[2]" in item.detail


def test_npu_environment_rejects_stale_cann_from_another_install_prefix(
    monkeypatch, tmp_path
):
    root = _stub_healthy_npu(monkeypatch, tmp_path)
    stale = tmp_path / "other-prefix" / "Ascend" / "cann-8.0.0" / "lib64"
    monkeypatch.setenv(
        "LD_LIBRARY_PATH", ":".join((str(root / "lib64"), str(stale)))
    )

    report = Env.check("npu")

    item = next(item for item in report.items if item.name == "CANN 动态库路径")
    assert item.status == env_check.ERROR
    assert "LD_LIBRARY_PATH[1]" in item.detail
    assert str(stale.parent) in item.detail


def test_npu_environment_warns_for_duplicate_or_same_root_alias(
    monkeypatch, tmp_path
):
    root = _stub_healthy_npu(monkeypatch, tmp_path)
    latest = root.parent / "ascend-toolkit" / "latest"
    latest.parent.mkdir()
    latest.symlink_to(root, target_is_directory=True)
    monkeypatch.setenv(
        "LD_LIBRARY_PATH",
        ":".join((str(root / "lib64"), str(root / "lib64"), str(latest / "lib64"))),
    )

    report = Env.check("npu")

    item = next(item for item in report.items if item.name == "CANN 动态库路径")
    assert item.status == env_check.WARN
    assert "重复" in item.detail
    assert "别名" in item.detail


def test_cann_path_audit_preserves_driver_nnal_and_unrelated_paths(
    monkeypatch, tmp_path
):
    root = _stub_healthy_npu(monkeypatch, tmp_path)
    entries = (
        root / "lib64",
        root.parent / "driver" / "lib64",
        root.parent / "nnal" / "atb" / "latest" / "lib",
        tmp_path / "custom-opp" / "op_api" / "lib",
        tmp_path / "unrelated" / "lib",
        tmp_path / "unrelated" / "devlib",
    )
    monkeypatch.setenv("LD_LIBRARY_PATH", ":".join(map(str, entries)))

    report = Env.check("npu")

    item = next(item for item in report.items if item.name == "CANN 动态库路径")
    assert item.status == env_check.OK
