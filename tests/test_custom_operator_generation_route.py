"""Offline contracts for the explicit custom OperatorSpec generation lane."""

import json
from pathlib import Path

import pytest

from numpilot.gaps.official_resolution import (
    BLOCKED_ROUTE,
    CUSTOM_OPERATOR_ROUTE,
    OfficialResolutionError,
    OfficialOperatorResolutionReport,
    resolve_official_operators,
    write_official_resolution_report,
)
from numpilot.gaps.pipeline import prepare_gap_completion
from numpilot.gaps.spec import is_custom_generation_operator_gap

from tests.test_official_operator_resolution import _inventory


def _custom_preparation(tmp_path):
    source = Path(__file__).parents[1] / "examples" / (
        "custom_operator_euclidean_distance.py"
    )
    # Rebind the source path to the actual checkout path.  The preparation is
    # still pure local AST work; no model or external checkout is touched.
    return prepare_gap_completion(
        str(source),
        target_soc="Ascend910B4",
        cann_version="9.0.0",
    )


def test_custom_route_requires_explicit_opt_in_and_preserves_ambiguity(tmp_path):
    preparation = _custom_preparation(tmp_path)
    spec = preparation.specs[0]
    assert is_custom_generation_operator_gap(preparation.source, spec)

    default = resolve_official_operators(
        preparation,
        cann_root=str(tmp_path / "cann"),
        inventory_loader=lambda _root: _inventory(tmp_path, arange=False),
    )
    assert default.status == "blocked"
    assert default.decisions[0].route == BLOCKED_ROUTE

    opted = resolve_official_operators(
        preparation,
        cann_root=str(tmp_path / "cann"),
        allow_custom_operator_generation=True,
        inventory_loader=lambda _root: _inventory(tmp_path, arange=False),
    )
    decision = opted.decisions[0]
    assert opted.status == "custom_operator_generation_allowed"
    assert opted.model_kernel_generation_allowed is True
    assert decision.route == CUSTOM_OPERATOR_ROUTE
    assert decision.new_kernel_generation_allowed is True
    assert decision.activation_allowed is False
    # The provider remains explicitly ambiguous; this is not a synthetic
    # clean no_match permission.
    assert decision.installed_cann["status"] == "ambiguous"
    assert decision.installed_cann["reason_code"] == "catalog_unreviewed_api"
    assert decision.installed_cann["custom_generation_proof"] == {
        "operator_spec_provenance": "custom_generation_candidate",
        "operator_spec_route": "ascendc_registry_candidate",
        "official_equivalence": "not_proven",
    }
    assert decision.to_dict()["custom_generation_proof"]["official_equivalence"] == (
        "not_proven"
    )


def test_custom_route_receipt_roundtrip_rechecks_source_binding(tmp_path):
    preparation = _custom_preparation(tmp_path)
    report = resolve_official_operators(
        preparation,
        cann_root=str(tmp_path / "cann"),
        allow_custom_operator_generation=True,
        inventory_loader=lambda _root: _inventory(tmp_path, arange=False),
    )
    restored = OfficialOperatorResolutionReport.from_dict(report.to_dict())
    restored.validate_preparation(preparation)
    assert restored.to_dict() == report.to_dict()

    tampered = json.loads(json.dumps(report.to_dict()))
    tampered["decisions"][0]["custom_generation_proof"][
        "official_equivalence"
    ] = "clean_no_match"
    with pytest.raises(OfficialResolutionError, match="custom generation proof"):
        OfficialOperatorResolutionReport.from_dict(tampered)


def test_custom_opt_in_cannot_relabel_an_ordinary_numpy_gap(tmp_path):
    source = Path(__file__).parents[1] / "examples" / "gap_isnan.py"
    preparation = prepare_gap_completion(
        str(source), target_soc="Ascend910B4", cann_version="9.0.0"
    )
    report = resolve_official_operators(
        preparation,
        cann_root=str(tmp_path / "cann"),
        allow_custom_operator_generation=True,
        inventory_loader=lambda _root: _inventory(tmp_path, arange=False),
    )
    assert report.status == "blocked"
    assert all(item.route == BLOCKED_ROUTE for item in report.decisions)


def test_official_source_candidate_still_wins_over_custom_opt_in(tmp_path):
    preparation = _custom_preparation(tmp_path)

    class CandidateSource:
        def resolve(self, **kwargs):
            return type("Result", (), {"to_dict": lambda self: {
                "schema_version": 1,
                "provider": "ops_nn_source",
                "status": "ambiguous",
                "reason": "reviewed official candidate",
                "candidates": [{"candidate_id": "f" * 64}],
                "requested": dict(kwargs),
                "source_ref": "refs/tags/v9.0.0",
                "source_commit": "1" * 40,
                "source_tree": "2" * 40,
                "selected_candidate_id": "",
                "requires_audit": True,
                "resolution_kind": "ambiguous",
                "search_contract_revision": "",
                "searched_aliases": [kwargs["op_name"]],
                "negative_search_complete": False,
                "activation_allowed": False,
            }})()

    report = resolve_official_operators(
        preparation,
        cann_root=str(tmp_path / "cann"),
        ops_nn_root=str(tmp_path / "ops-nn"),
        accept_cann_osl=True,
        allow_custom_operator_generation=True,
        inventory_loader=lambda _root: _inventory(tmp_path, arange=False),
        source_resolver_factory=lambda _root: CandidateSource(),
    )
    # Installed catalog ambiguity plus a positive official source candidate
    # cannot be hidden behind the custom flag.
    assert report.status == "blocked"
    assert report.decisions[0].route == BLOCKED_ROUTE


def test_complete_cli_requires_custom_flag_before_model_or_fresh_resolver(
    tmp_path, monkeypatch, capsys
):
    preparation = _custom_preparation(tmp_path)
    report = resolve_official_operators(
        preparation,
        cann_root=str(tmp_path / "cann"),
        allow_custom_operator_generation=True,
        inventory_loader=lambda _root: _inventory(tmp_path, arange=False),
    )
    receipt = tmp_path / "custom-resolution.json"
    write_official_resolution_report(report, str(receipt))

    from numpilot import __main__ as cli
    from numpilot.gaps import official_resolution as resolution_module

    monkeypatch.setattr(
        resolution_module,
        "resolve_official_operators",
        lambda *_args, **_kwargs: pytest.fail(
            "custom receipt without opt-in must stop before fresh resolver"
        ),
    )
    monkeypatch.setattr(
        cli,
        "_load_llm_or_report",
        lambda: pytest.fail("custom receipt without opt-in must stop before model"),
    )
    result = cli.main([
        "complete-gaps",
        str(Path(__file__).parents[1] / "examples" /
            "custom_operator_euclidean_distance.py"),
        "--provider", "llm",
        "--target-soc", "Ascend910B4",
        "--cann-version", "9.0.0",
        "--official-resolution-report", str(receipt),
        "--out", str(tmp_path / "no-model"),
    ])
    captured = capsys.readouterr()
    assert result == 3
    assert "allow-custom-operator-generation" in captured.err
    _assert_only_prepared_taskbooks((tmp_path / "no-model"))


def _assert_only_prepared_taskbooks(output):
    from numpilot.gap_requirements import validate_gap_taskbook
    files = sorted(path for path in output.rglob("*") if path.is_file())
    assert len(files) == 2
    assert {path.name for path in files} == {"GAP_TASKBOOK.json", "GAP_TASKBOOK.md"}
    assert all(path.relative_to(output).parts[0] == "gaps" for path in files)
    book = validate_gap_taskbook(json.loads(next(path for path in files if path.suffix == ".json").read_text()))
    assert book["status"] == "requirements_prepared_unverified"
    assert book["generation"]["claim_level"] == "unverified_candidate_only"
