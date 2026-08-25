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

    assert "v0.2.0a1" in code
    assert "bootstrap', '--apply'" in code
    assert "bootstrap', '--verify'" in code
    assert "--accept-cannbot-license" in code
    assert "configure-llm', '--check'" in code
    assert "migrate" in code and "migrated.py" in code
    assert "127.0.0.1:7897" not in code
    assert "auth_token" not in code
