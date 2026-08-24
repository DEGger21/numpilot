import sys

import pytest

import numpilot.env_check as env_check
from numpilot.env_check import Env, _npu_smi_has_device


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
