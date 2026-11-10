import hashlib
import json
from types import SimpleNamespace

import pytest

from numpilot.gaps.installed_cann import (
    FileEvidence,
    HeaderEvidence,
    InstalledCannInventory,
    LibraryEvidence,
    SymbolProbeOutput,
    VersionEvidence,
)
from numpilot.gaps.models import GapCallsite, GapSpec
from numpilot.gaps.official_resolution import (
    BLOCKED_ROUTE,
    INSTALLED_ADAPTER_ROUTE,
    NEW_KERNEL_ROUTE,
    OFFICIAL_SOURCE_ROUTE,
    OfficialResolutionError,
    load_official_resolution_report,
    resolve_official_operators,
    write_official_resolution_report,
)
from numpilot.gaps.official_source import SOURCE_SEARCH_CONTRACT_REVISION
from numpilot.gaps.pipeline import GapPreparation
from numpilot.gaps.pipeline import prepare_gap_completion


ZERO = "0" * 64


def _file(path):
    return FileEvidence(relative_path=path, sha256=ZERO, size=1)


def _inventory(tmp_path, *, arange=True):
    versions = (
        VersionEvidence(
            file=_file("aarch64-linux/ascend_toolkit_install.info"),
            version="9.0.0",
            source_field="version",
        ),
    )
    headers = ()
    libraries = ()
    if arange:
        symbols = ("aclnnArange", "aclnnArangeGetWorkspaceSize")
        probe = SymbolProbeOutput(tool="injected-test-probe", symbols=symbols)
        headers = (
            HeaderEvidence(
                file=_file("aarch64-linux/include/aclnnop/aclnn_arange.h"),
                declared_symbols=symbols,
            ),
        )
        libraries = (
            LibraryEvidence(
                file=_file("aarch64-linux/lib64/libopapi_math.so"),
                probe_tool=probe.tool,
                symbols=probe.symbols,
                symbols_sha256=probe.symbols_sha256,
            ),
        )
    return InstalledCannInventory(
        root=str((tmp_path / "cann-9.0.0").absolute()),
        platform="aarch64-linux",
        architecture="aarch64",
        operating_system="linux",
        version="9.0.0",
        version_evidence=versions,
        headers=headers,
        libraries=libraries,
    )


def _preparation(tmp_path, *apis, target_soc="Ascend910B4"):
    source = "\n".join(
        "value_%d = np.%s(value)" % (index, api)
        for index, api in enumerate(apis, 1)
    ) + "\n"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    source_path = str((tmp_path / "input.py").absolute())
    specs = tuple(
        GapSpec(
            gap_id="%s-%016d" % (api.replace("_", "-"), index),
            api_name=api,
            qualified_api="numpy." + api,
            asnumpy_api="asnumpy." + api,
            source_path=source_path,
            source_sha256=digest,
            callsites=(
                GapCallsite(
                    lineno=index,
                    col_offset=10,
                    qualified_api="numpy." + api,
                    call_expr="numpy.%s(...)" % api,
                    reason="test gap",
                ),
            ),
            signature_variants=({"positional_args": 1, "keyword_args": []},),
            target_soc=target_soc,
            cann_version="9.0.0",
        )
        for index, api in enumerate(apis, 1)
    )
    return GapPreparation(
        source_path=source_path,
        source=source,
        unsupported=tuple("numpy." + api for api in apis),
        specs=specs,
        blocked=(),
    )


class _SourceResolver:
    def __init__(self, payload, calls):
        self.payload = payload
        self.calls = calls

    def resolve(self, **kwargs):
        self.calls.append(kwargs)
        payload = dict(self.payload)
        payload.setdefault("provider", "ops_nn_source")
        payload.setdefault("schema_version", 1)
        payload.setdefault("reason", "test source result")
        payload.setdefault("candidates", [])
        payload.setdefault("requested", dict(kwargs))
        payload.setdefault("source_ref", "refs/tags/v9.0.0")
        payload.setdefault("source_commit", "1" * 40)
        payload.setdefault("source_tree", "2" * 40)
        payload.setdefault("selected_candidate_id", "")
        payload.setdefault("requires_audit", payload.get("status") != "no_match")
        payload.setdefault("resolution_kind", "none")
        payload.setdefault(
            "search_contract_revision",
            SOURCE_SEARCH_CONTRACT_REVISION
            if payload.get("status") == "no_match" else "",
        )
        payload.setdefault("searched_aliases", [kwargs["op_name"]])
        payload.setdefault(
            "negative_search_complete", payload.get("status") == "no_match"
        )
        payload.setdefault("activation_allowed", False)
        return SimpleNamespace(to_dict=lambda: payload)


def _source_factory(payload, calls, factory_calls=None):
    def factory(root):
        if factory_calls is not None:
            factory_calls.append(root)
        return _SourceResolver(payload, calls)

    return factory


def test_installed_exact_candidate_precedes_source_and_forbids_new_kernel(tmp_path):
    preparation = _preparation(
        tmp_path,
        "arange",
        target_soc="Ascend910B4",
    )
    source_factories = []

    report = resolve_official_operators(
        preparation,
        cann_root=str(tmp_path / "cann-9.0.0"),
        ops_nn_root=str(tmp_path / "ops-nn"),
        accept_cann_osl=True,
        inventory_loader=lambda root: _inventory(tmp_path),
        source_resolver_factory=lambda root: source_factories.append(root),
    )

    assert report.status == "official_candidate_found"
    assert report.model_kernel_generation_allowed is False
    assert report.decisions[0].route == INSTALLED_ADAPTER_ROUTE
    assert report.decisions[0].adapter_generation_required is True
    assert report.decisions[0].activation_allowed is False
    assert report.decisions[0].ops_nn_source is None
    assert source_factories == []


def test_target_soc_family_blocks_before_inventory_or_source_access(tmp_path):
    preparation = _preparation(
        tmp_path, "cbrt", target_soc="Ascend910B"
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a family SoC must block before provider access")

    report = resolve_official_operators(
        preparation,
        cann_root=str(tmp_path / "missing-cann"),
        ops_nn_root=str(tmp_path / "missing-ops-nn"),
        accept_cann_osl=True,
        inventory_loader=forbidden,
        source_resolver_factory=forbidden,
    )

    decision = report.decisions[0]
    assert report.status == "blocked"
    assert decision.route == BLOCKED_ROUTE
    assert decision.installed_cann["status"] == "ambiguous"
    assert decision.installed_cann["reason_code"] == (
        "target_soc_family_not_exact"
    )


def test_installed_miss_then_source_candidate_requires_audit_and_build(tmp_path):
    preparation = _preparation(tmp_path, "cbrt")
    calls = []
    candidate = {
        "candidate_id": "1" * 64,
        "surfaces": {"integration_surface_complete": True},
    }

    report = resolve_official_operators(
        preparation,
        cann_root=str(tmp_path / "cann-9.0.0"),
        ops_nn_root=str(tmp_path / "ops-nn"),
        accept_cann_osl=True,
        inventory_loader=lambda root: _inventory(tmp_path),
        source_resolver_factory=_source_factory(
            {"status": "ambiguous", "candidates": [candidate]}, calls
        ),
    )

    decision = report.decisions[0]
    assert decision.installed_cann["status"] == "no_match"
    assert decision.route == OFFICIAL_SOURCE_ROUTE
    assert decision.new_kernel_generation_allowed is False
    assert decision.adapter_generation_required is False
    assert decision.semantic_audit_required is True
    assert decision.source_build_required is True
    assert calls == [{
        "cann_version": "9.0.0",
        "target_soc": "Ascend910B4",
        "op_name": "cbrt",
        "qualified_api": "numpy.cbrt",
    }]


def test_new_kernel_requires_two_clean_no_match_results(tmp_path):
    preparation = _preparation(tmp_path, "cbrt")
    calls = []

    report = resolve_official_operators(
        preparation,
        cann_root=str(tmp_path / "cann-9.0.0"),
        ops_nn_root=str(tmp_path / "ops-nn"),
        accept_cann_osl=True,
        inventory_loader=lambda root: _inventory(tmp_path),
        source_resolver_factory=_source_factory(
            {"status": "no_match", "candidates": []}, calls
        ),
    )

    assert report.status == "new_kernel_generation_allowed"
    assert report.model_kernel_generation_allowed is True
    assert report.decisions[0].route == NEW_KERNEL_ROUTE
    assert report.decisions[0].new_kernel_generation_allowed is True


def test_ops_nn_checkout_symlink_loop_blocks_without_escaping_the_resolver(tmp_path):
    preparation = _preparation(tmp_path, "cbrt")
    loop = tmp_path / "ops-nn-loop"
    loop.symlink_to(loop, target_is_directory=True)

    report = resolve_official_operators(
        preparation,
        cann_root=str(tmp_path / "cann-9.0.0"),
        ops_nn_root=str(loop),
        accept_cann_osl=True,
        inventory_loader=lambda _root: _inventory(tmp_path, arange=False),
    )

    decision = report.decisions[0]
    assert report.status == "blocked"
    assert decision.route == BLOCKED_ROUTE
    assert decision.ops_nn_source["status"] == "error"
    assert "ValueError" in decision.ops_nn_source["reason"]


@pytest.mark.parametrize(
    ("accept_license", "ops_root"),
    ((False, "ops-nn"), (True, "")),
)
def test_missing_license_or_source_checkout_blocks_without_source_read(
    tmp_path, accept_license, ops_root
):
    preparation = _preparation(tmp_path, "cbrt")
    factory_calls = []

    report = resolve_official_operators(
        preparation,
        cann_root=str(tmp_path / "cann-9.0.0"),
        ops_nn_root=str(tmp_path / ops_root) if ops_root else "",
        accept_cann_osl=accept_license,
        inventory_loader=lambda root: _inventory(tmp_path),
        source_resolver_factory=lambda root: factory_calls.append(root),
    )

    assert report.status == "blocked"
    assert report.decisions[0].route == BLOCKED_ROUTE
    assert report.decisions[0].new_kernel_generation_allowed is False
    assert factory_calls == []


def test_inventory_failure_blocks_all_gaps_and_never_queries_source(tmp_path):
    preparation = _preparation(tmp_path, "arange", "cbrt")
    source_factories = []

    def fail_inventory(root):
        raise ValueError("do not leak this diagnostic")

    report = resolve_official_operators(
        preparation,
        cann_root=str(tmp_path / "broken-cann"),
        ops_nn_root=str(tmp_path / "ops-nn"),
        accept_cann_osl=True,
        inventory_loader=fail_inventory,
        source_resolver_factory=lambda root: source_factories.append(root),
    )

    assert report.status == "blocked"
    assert {item.route for item in report.decisions} == {BLOCKED_ROUTE}
    assert all(
        item.installed_cann["reason"]
        == "installed CANN inventory failed safely: ValueError"
        for item in report.decisions
    )
    assert source_factories == []


def test_multiple_gaps_share_one_inventory_and_one_source_resolver(tmp_path):
    preparation = _preparation(tmp_path, "cbrt", "arange")
    inventory_calls = []
    source_calls = []
    factory_calls = []

    def inventory_loader(root):
        inventory_calls.append(root)
        return _inventory(tmp_path, arange=False)

    report = resolve_official_operators(
        preparation,
        cann_root=str(tmp_path / "cann-9.0.0"),
        ops_nn_root=str(tmp_path / "ops-nn"),
        accept_cann_osl=True,
        inventory_loader=inventory_loader,
        source_resolver_factory=_source_factory(
            {"status": "no_match"}, source_calls, factory_calls
        ),
    )

    assert report.model_kernel_generation_allowed is True
    assert len(inventory_calls) == 1
    assert len(factory_calls) == 1
    assert len(source_calls) == 2


def test_report_roundtrip_is_private_digest_bound_and_preparation_bound(tmp_path):
    preparation = _preparation(tmp_path, "cbrt")
    report = resolve_official_operators(
        preparation,
        cann_root=str(tmp_path / "cann-9.0.0"),
        ops_nn_root=str(tmp_path / "ops-nn"),
        accept_cann_osl=True,
        inventory_loader=lambda root: _inventory(tmp_path),
        source_resolver_factory=_source_factory(
            {"status": "no_match"}, []
        ),
    )
    target = tmp_path / "official-resolution.json"

    write_official_resolution_report(report, str(target))
    loaded = load_official_resolution_report(str(target))

    assert target.stat().st_mode & 0o777 == 0o600
    assert loaded.to_dict() == report.to_dict()
    loaded.validate_preparation(preparation)

    changed = _preparation(tmp_path, "sinc")
    with pytest.raises(OfficialResolutionError, match="bound"):
        loaded.validate_preparation(changed)


def test_receipt_read_and_write_reject_an_ancestor_symlink(tmp_path):
    preparation = _preparation(tmp_path, "cbrt")
    report = resolve_official_operators(
        preparation,
        cann_root=str(tmp_path / "cann-9.0.0"),
        ops_nn_root=str(tmp_path / "ops-nn"),
        accept_cann_osl=True,
        inventory_loader=lambda root: _inventory(tmp_path),
        source_resolver_factory=_source_factory({"status": "no_match"}, []),
    )
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    alias = tmp_path / "linked-parent"
    alias.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(OfficialResolutionError, match="unsafe ancestor"):
        write_official_resolution_report(report, str(alias / "write.json"))
    assert not (real_parent / "write.json").exists()

    real_target = real_parent / "read.json"
    write_official_resolution_report(report, str(real_target))
    with pytest.raises(OfficialResolutionError, match="unsafe ancestor"):
        load_official_resolution_report(str(alias / "read.json"))


def test_identical_existing_receipt_with_0644_mode_fails_without_repair(tmp_path):
    preparation = _preparation(tmp_path, "cbrt")
    report = resolve_official_operators(
        preparation,
        cann_root=str(tmp_path / "cann-9.0.0"),
        ops_nn_root=str(tmp_path / "ops-nn"),
        accept_cann_osl=True,
        inventory_loader=lambda root: _inventory(tmp_path),
        source_resolver_factory=_source_factory({"status": "no_match"}, []),
    )
    target = tmp_path / "official-resolution.json"
    write_official_resolution_report(report, str(target))
    original = target.read_bytes()
    target.chmod(0o644)

    with pytest.raises(OfficialResolutionError, match="exactly 0600"):
        write_official_resolution_report(report, str(target))
    assert target.read_bytes() == original
    assert target.stat().st_mode & 0o777 == 0o644

    with pytest.raises(OfficialResolutionError, match="exactly 0600"):
        load_official_resolution_report(str(target))
    assert target.stat().st_mode & 0o777 == 0o644


def test_receipt_publication_does_not_use_path_based_chmod(
    tmp_path, monkeypatch
):
    from numpilot.gaps import official_resolution as resolution_module

    preparation = _preparation(tmp_path, "cbrt")
    report = resolve_official_operators(
        preparation,
        cann_root=str(tmp_path / "cann-9.0.0"),
        ops_nn_root=str(tmp_path / "ops-nn"),
        accept_cann_osl=True,
        inventory_loader=lambda root: _inventory(tmp_path),
        source_resolver_factory=_source_factory({"status": "no_match"}, []),
    )
    monkeypatch.setattr(
        resolution_module.os,
        "chmod",
        lambda *_args, **_kwargs: pytest.fail("path-based chmod is forbidden"),
    )
    target = tmp_path / "official-resolution.json"

    write_official_resolution_report(report, str(target))

    assert target.stat().st_mode & 0o777 == 0o600
    assert load_official_resolution_report(str(target)).to_dict() == report.to_dict()


@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_existing_non_regular_receipt_is_rejected(tmp_path, kind):
    preparation = _preparation(tmp_path, "cbrt")
    report = resolve_official_operators(
        preparation,
        cann_root=str(tmp_path / "cann-9.0.0"),
        ops_nn_root=str(tmp_path / "ops-nn"),
        accept_cann_osl=True,
        inventory_loader=lambda root: _inventory(tmp_path),
        source_resolver_factory=_source_factory({"status": "no_match"}, []),
    )
    target = tmp_path / "official-resolution.json"
    if kind == "symlink":
        backing = tmp_path / "backing.json"
        backing.write_text("{}\n", encoding="utf-8")
        target.symlink_to(backing)
    else:
        target.mkdir()

    with pytest.raises(OfficialResolutionError, match="regular"):
        write_official_resolution_report(report, str(target))
    with pytest.raises(OfficialResolutionError, match="regular"):
        load_official_resolution_report(str(target))


def test_tampered_report_is_rejected_even_when_json_remains_valid(tmp_path):
    preparation = _preparation(tmp_path, "cbrt")
    report = resolve_official_operators(
        preparation,
        cann_root=str(tmp_path / "cann-9.0.0"),
        ops_nn_root=str(tmp_path / "ops-nn"),
        accept_cann_osl=True,
        inventory_loader=lambda root: _inventory(tmp_path),
        source_resolver_factory=_source_factory(
            {"status": "no_match"}, []
        ),
    )
    target = tmp_path / "official-resolution.json"
    payload = report.to_dict()
    payload["decisions"][0]["reason"] = "tampered but internally plausible"
    target.write_text(json.dumps(payload), encoding="utf-8")
    target.chmod(0o600)

    with pytest.raises(OfficialResolutionError, match="digest"):
        load_official_resolution_report(str(target))


def test_existing_different_report_is_never_overwritten_by_default(tmp_path):
    preparation = _preparation(tmp_path, "cbrt")
    report = resolve_official_operators(
        preparation,
        cann_root=str(tmp_path / "cann-9.0.0"),
        ops_nn_root=str(tmp_path / "ops-nn"),
        accept_cann_osl=True,
        inventory_loader=lambda root: _inventory(tmp_path),
        source_resolver_factory=_source_factory(
            {"status": "no_match"}, []
        ),
    )
    target = tmp_path / "official-resolution.json"
    target.write_text("{}\n", encoding="utf-8")
    target.chmod(0o600)

    with pytest.raises(OfficialResolutionError, match="overwrite"):
        write_official_resolution_report(report, str(target))

    assert target.read_text(encoding="utf-8") == "{}\n"


def _write_private_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)


@pytest.mark.parametrize("schema_version", ["1", True, 1.0, None])
def test_receipt_rejects_non_integer_schema_version(tmp_path, schema_version):
    preparation = _preparation(tmp_path, "cbrt")
    report = resolve_official_operators(
        preparation,
        cann_root=str(tmp_path / "cann-9.0.0"),
        ops_nn_root=str(tmp_path / "ops-nn"),
        accept_cann_osl=True,
        inventory_loader=lambda root: _inventory(tmp_path),
        source_resolver_factory=_source_factory({"status": "no_match"}, []),
    )
    payload = report.to_dict()
    payload["schema_version"] = schema_version
    target = tmp_path / "bad-schema.json"
    _write_private_json(target, payload)

    with pytest.raises(OfficialResolutionError, match="schema"):
        load_official_resolution_report(str(target))


def test_receipt_path_apis_reject_nul_before_filesystem_access(tmp_path):
    preparation = _preparation(tmp_path, "cbrt")
    report = resolve_official_operators(
        preparation,
        cann_root=str(tmp_path / "cann-9.0.0"),
        ops_nn_root=str(tmp_path / "ops-nn"),
        accept_cann_osl=True,
        inventory_loader=lambda root: _inventory(tmp_path),
        source_resolver_factory=_source_factory({"status": "no_match"}, []),
    )
    invalid = str(tmp_path / "nul") + "\x00/receipt.json"

    with pytest.raises(OfficialResolutionError, match="path is invalid"):
        write_official_resolution_report(report, invalid)
    with pytest.raises(OfficialResolutionError, match="path is invalid"):
        load_official_resolution_report(invalid)


def test_receipt_rejects_nul_in_bound_source_path(tmp_path):
    preparation = _preparation(tmp_path, "cbrt")
    report = resolve_official_operators(
        preparation,
        cann_root=str(tmp_path / "cann-9.0.0"),
        ops_nn_root=str(tmp_path / "ops-nn"),
        accept_cann_osl=True,
        inventory_loader=lambda root: _inventory(tmp_path),
        source_resolver_factory=_source_factory({"status": "no_match"}, []),
    )
    payload = report.to_dict()
    payload["source_path"] = "/tmp/nul\x00source.py"
    target = tmp_path / "nul-source-path.json"
    _write_private_json(target, payload)

    with pytest.raises(OfficialResolutionError, match="source path"):
        load_official_resolution_report(str(target))


def test_receipt_rejects_integer_token_beyond_parser_bound(tmp_path):
    target = tmp_path / "huge-integer.json"
    target.write_bytes(
        b'{"schema_version":' + b"9" * 10_000 + b"}"
    )
    target.chmod(0o600)

    with pytest.raises(OfficialResolutionError, match="bounded strict"):
        load_official_resolution_report(str(target))


def test_receipt_rejects_extreme_json_nesting_without_leaking_recursion(tmp_path):
    target = tmp_path / "deep.json"
    depth = 2_000
    target.write_bytes(b"[" * depth + b"0" + b"]" * depth)
    target.chmod(0o600)

    with pytest.raises(OfficialResolutionError, match="bounded strict"):
        load_official_resolution_report(str(target))


def _static_cbrt_preparation(tmp_path):
    source = tmp_path / "cbrt_case.py"
    source.write_text(
        "import numpy as np\n\n"
        "def forward(x):\n"
        "    return np.cbrt(x)\n",
        encoding="utf-8",
    )
    preparation = prepare_gap_completion(
        str(source),
        target_soc="Ascend910B4",
        cann_version="9.0.0",
    )
    assert [spec.qualified_api for spec in preparation.specs] == ["numpy.cbrt"]
    return source, preparation


def test_resolve_gaps_cli_is_local_only_and_writes_bound_receipt(
    tmp_path, monkeypatch, capsys
):
    from numpilot import __main__ as cli
    from numpilot.gaps import official_resolution as resolution_module

    source, _preparation = _static_cbrt_preparation(tmp_path)
    output = tmp_path / "resolution.json"
    original = resolution_module.resolve_official_operators

    def hermetic_resolve(preparation, **kwargs):
        return original(
            preparation,
            cann_root=kwargs["cann_root"],
            ops_nn_root=kwargs["ops_nn_root"],
            accept_cann_osl=kwargs["accept_cann_osl"],
            inventory_loader=lambda root: _inventory(tmp_path),
            source_resolver_factory=_source_factory(
                {"status": "no_match"}, []
            ),
        )

    monkeypatch.setattr(
        resolution_module, "resolve_official_operators", hermetic_resolve
    )
    monkeypatch.setattr(
        cli,
        "_load_llm_or_report",
        lambda: pytest.fail("resolve-gaps must never inspect model configuration"),
    )

    result = cli.main([
        "resolve-gaps",
        str(source),
        "--out", str(output),
        "--target-soc", "Ascend910B4",
        "--cann-version", "9.0.0",
        "--cann-root", str(tmp_path / "cann-9.0.0"),
        "--ops-nn-root", str(tmp_path / "ops-nn"),
        "--accept-cann-osl",
    ])

    captured = capsys.readouterr()
    assert result == 0
    assert "model_kernel_allowed=yes" in captured.out
    assert "model_call=no" in captured.out
    assert load_official_resolution_report(str(output)).status == (
        "new_kernel_generation_allowed"
    )


def test_complete_gaps_candidate_receipt_blocks_before_model_or_skill(
    tmp_path, monkeypatch, capsys
):
    from numpilot import __main__ as cli
    from numpilot.gaps import official_resolution as resolution_module

    source, preparation = _static_cbrt_preparation(tmp_path)
    report = resolve_official_operators(
        preparation,
        cann_root=str(tmp_path / "cann-9.0.0"),
        ops_nn_root=str(tmp_path / "ops-nn"),
        accept_cann_osl=True,
        inventory_loader=lambda root: _inventory(tmp_path),
        source_resolver_factory=_source_factory(
            {
                "status": "ambiguous",
                "candidates": [{"candidate_id": "2" * 64}],
            },
            [],
        ),
    )
    receipt = tmp_path / "candidate-resolution.json"
    write_official_resolution_report(report, str(receipt))
    monkeypatch.setattr(
        resolution_module,
        "resolve_official_operators",
        lambda _preparation, **_kwargs: report,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("candidate route crossed the model/Skill boundary")

    monkeypatch.setattr(cli, "_load_llm_or_report", forbidden)
    monkeypatch.setattr(cli, "resolve_skill_bundle", forbidden)
    output = tmp_path / "must-not-exist"

    result = cli.main([
        "complete-gaps",
        str(source),
        "--out", str(output),
        "--target-soc", "Ascend910B4",
        "--cann-version", "9.0.0",
        "--provider", "llm",
        "--official-resolution-report", str(receipt),
    ])

    captured = capsys.readouterr()
    assert result == 3
    assert "不得调用模型手写 kernel" in captured.err
    assert "actual_remote_generation_call=no" in captured.err
    _assert_only_prepared_taskbooks(output)


def test_complete_gaps_rejects_tampered_resolution_before_model(tmp_path, monkeypatch):
    from numpilot import __main__ as cli

    source, preparation = _static_cbrt_preparation(tmp_path)
    report = resolve_official_operators(
        preparation,
        cann_root=str(tmp_path / "cann-9.0.0"),
        ops_nn_root=str(tmp_path / "ops-nn"),
        accept_cann_osl=True,
        inventory_loader=lambda root: _inventory(tmp_path),
        source_resolver_factory=_source_factory({"status": "no_match"}, []),
    )
    payload = report.to_dict()
    payload["decisions"][0]["new_kernel_generation_allowed"] = False
    receipt = tmp_path / "tampered.json"
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    receipt.chmod(0o600)

    monkeypatch.setattr(
        cli,
        "_load_llm_or_report",
        lambda: pytest.fail("tampered receipt must be rejected before model access"),
    )
    result = cli.main([
        "complete-gaps",
        str(source),
        "--out", str(tmp_path / "out"),
        "--target-soc", "Ascend910B4",
        "--cann-version", "9.0.0",
        "--provider", "llm",
        "--official-resolution-report", str(receipt),
    ])

    assert result == 2
    _assert_only_prepared_taskbooks((tmp_path / "out"))


def test_complete_gaps_rechecks_current_official_evidence_before_model(
    tmp_path, monkeypatch, capsys
):
    from numpilot import __main__ as cli
    from numpilot.gaps import official_resolution as resolution_module

    source, preparation = _static_cbrt_preparation(tmp_path)
    original_report = resolve_official_operators(
        preparation,
        cann_root=str(tmp_path / "cann-9.0.0"),
        ops_nn_root=str(tmp_path / "ops-nn"),
        accept_cann_osl=True,
        inventory_loader=lambda root: _inventory(tmp_path),
        source_resolver_factory=_source_factory({"status": "no_match"}, []),
    )
    changed_report = resolve_official_operators(
        preparation,
        cann_root=str(tmp_path / "cann-9.0.0"),
        ops_nn_root=str(tmp_path / "ops-nn"),
        accept_cann_osl=True,
        inventory_loader=lambda root: _inventory(tmp_path),
        source_resolver_factory=_source_factory(
            {
                "status": "ambiguous",
                "candidates": [{"candidate_id": "3" * 64}],
            },
            [],
        ),
    )
    receipt = tmp_path / "stale-resolution.json"
    write_official_resolution_report(original_report, str(receipt))
    monkeypatch.setattr(
        resolution_module,
        "resolve_official_operators",
        lambda _preparation, **_kwargs: changed_report,
    )
    monkeypatch.setattr(
        cli,
        "_load_llm_or_report",
        lambda: pytest.fail("stale official evidence must block model access"),
    )

    result = cli.main([
        "complete-gaps",
        str(source),
        "--out", str(tmp_path / "out"),
        "--target-soc", "Ascend910B4",
        "--cann-version", "9.0.0",
        "--provider", "llm",
        "--official-resolution-report", str(receipt),
    ])

    captured = capsys.readouterr()
    assert result == 2
    assert "evidence changed" in captured.err
    _assert_only_prepared_taskbooks((tmp_path / "out"))


def _assert_only_prepared_taskbooks(output):
    from numpilot.gap_requirements import validate_gap_taskbook
    files = sorted(path for path in output.rglob("*") if path.is_file())
    assert len(files) == 2
    assert {path.name for path in files} == {"GAP_TASKBOOK.json", "GAP_TASKBOOK.md"}
    assert all(path.relative_to(output).parts[0] == "gaps" for path in files)
    book = validate_gap_taskbook(json.loads(next(path for path in files if path.suffix == ".json").read_text()))
    assert book["status"] == "requirements_prepared_unverified"
    assert book["generation"]["claim_level"] == "unverified_candidate_only"
