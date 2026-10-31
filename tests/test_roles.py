"""六角色主链契约的单元测试。"""
from numpilot.roles import (ROLE_DEFINITIONS, ROLE_ORDER, Role, RoleRecord,
                            RoleStage, records_to_dicts, role_roster)


def test_six_roles_in_pipeline_order():
    assert len(ROLE_DEFINITIONS) == 6
    assert ROLE_ORDER == (
        Role.ANALYZER,
        Role.PLANNER,
        Role.GAP_FILLER,
        Role.CODE_MIGRATOR,
        Role.DIAGNOSTICS,
        Role.REPORTER,
    )
    assert len({definition.display_name for definition in ROLE_DEFINITIONS}) == 6


def test_stages_cover_three_phases():
    assert {definition.stage for definition in ROLE_DEFINITIONS} == {
        RoleStage.UNDERSTAND,
        RoleStage.MIGRATE,
        RoleStage.VALIDATE,
    }


def test_only_code_migrator_claims_a_model(monkeypatch):
    monkeypatch.setenv("NUMPILOT_LLM_MODEL", "model-x")
    ast_rows = role_roster("ast")
    llm_rows = role_roster("llm")
    assert len(ast_rows) == len(llm_rows) == 6
    ast_migrator = [m for n, _, m in ast_rows if "Code Migrator" in n][0]
    llm_migrator = [m for n, _, m in llm_rows if "Code Migrator" in n][0]
    assert "无模型" in ast_migrator and "model-x" in llm_migrator
    assert all("无模型" in m for n, _, m in ast_rows if "Code Migrator" not in n)


def test_records_serialise_in_canonical_order():
    records = [
        RoleRecord(Role.REPORTER, RoleStage.VALIDATE, "ok", "r"),
        RoleRecord(Role.ANALYZER, RoleStage.UNDERSTAND, "ok", "a"),
    ]
    serialised = records_to_dicts(records, "ast")
    assert [row["role"] for row in serialised] == ["analyzer", "reporter"]
    assert serialised[0]["stage"] == "understand"
    assert serialised[0]["model"] == "规则(无模型)"
