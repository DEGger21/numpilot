import sys

from numpilot.env_check import Env


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
