import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "docs" / "NumPilot-Demo-v1-Quickstart.ipynb"


def test_fresh_notebook_template_is_clean_and_covers_the_release_path():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    assert payload["nbformat"] == 4
    assert payload["cells"]
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell["cell_type"] == "code"
    )
    for cell in payload["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile(
                "".join(cell.get("source", [])),
                str(NOTEBOOK),
                "exec",
            )

    assert "install.sh" in code
    assert "--allow-network" in code
    assert "bootstrap', '--verify'" in code
    assert "--accept-cannbot-license" in code
    assert "doctor', '--backend', 'npu'" in code
    assert "ALLOW_NETWORK = False" in code
    assert "WITH_CANNBOT = False" in code
    assert "ACCEPT_CANNBOT_LICENSE = False" in code
    assert "if WITH_CANNBOT:" in code
    assert "127.0.0.1:7897" not in code
    assert "auth_token" not in code

    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell["cell_type"] == "markdown"
    )
    assert "不使用 wheelhouse" in markdown
    assert "1ef6372a538f39b0b7ed34666a3e1f465fb4936a" in markdown
    assert "protocol=openai-chat-completions" in markdown
    assert "GLM-5.3/provider=glm 只是配置示例" in markdown
    assert "v0.2.0a2` 只是基线且不包含本文合同" in markdown
    assert "strict_npu=false" in markdown
    assert "performance_valid=false" in markdown
    assert "开始迁移" in markdown
