from pathlib import Path
import hashlib


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_asnumpy_gap_skill_mirrors_codearts_discovery_copy():
    root = Path(__file__).parents[1]
    codearts = root / ".codeartsdoer/skills/asnumpy-gap-completion"
    visible = root / "skill/asnumpy-gap-completion"
    for relative in (
        "SKILL.md",
        "references/contract.md",
        "agents/openai.yaml",
    ):
        left = codearts / relative
        right = visible / relative
        assert left.is_file() and right.is_file()
        assert _digest(left) == _digest(right)


def test_asnumpy_gap_skill_mentions_fail_closed_runtime_contract():
    root = Path(__file__).parents[1]
    text = (root / ".codeartsdoer/skills/asnumpy-gap-completion/SKILL.md").read_text()
    for marker in (
        "asnumpy.adapter.manifest.v1",
        "explicit-handle-dlsym",
        "libcust_opapi.so",
        "generated_unverified",
        "150-example",
    ):
        assert marker in text
