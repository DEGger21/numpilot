import base64
import hashlib
import json
from pathlib import Path
import stat
import subprocess
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

import numpilot.bootstrap as bootstrap
from numpilot.bootstrap import (
    ASNUMPY_COMMIT,
    ASNUMPY_REPOSITORY,
    ASNUMPY_TREE,
    AsNumpyCheckout,
    BootstrapError,
    apply_bootstrap,
    build_bootstrap_plan,
    fetch_asnumpy_checkout,
    load_asnumpy_checkout,
    verify_asnumpy_installation,
    verify_bootstrap_installation,
)


_KEY_FILES = (
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "CMakeLists.txt",
    "bindings/python/CMakeLists.txt",
    "csrc/CMakeLists.txt",
)


def _git(root, *arguments):
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_repo(tmp_path):
    root = tmp_path / "asnumpy-source"
    root.mkdir()
    for relative in _KEY_FILES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture for %s\n" % relative, encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "remote", "add", "origin", ASNUMPY_REPOSITORY)
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=NumPilot Test",
        "-c",
        "user.email=numpilot@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    return root, _git(root, "rev-parse", "HEAD")


def _tool_finder(name):
    return "/test-tools/%s" % name


def _version_reader(name):
    versions = {
        "asnumpy": bootstrap.ASNUMPY_VERSION,
        "build": "1.2.2",
        "scikit-build-core": "1.0.0",
        "pybind11": "3.0.0",
        "numpy": "2.1.0",
        "loguru": "0.7.3",
        "setuptools-scm": "9.2.2",
        "towncrier": "24.8.0",
    }
    if name not in versions:
        raise LookupError(name)
    return versions[name]


def _successful_runner(calls):
    def run(arguments, **kwargs):
        assert isinstance(arguments, list)
        calls.append((list(arguments), dict(kwargs)))
        name = Path(arguments[0]).name
        if name == "cmake":
            output = "cmake version 3.26.5\n"
        elif name in {"g++", "gcc"}:
            output = "g++ (GCC) 11.2.1\n"
        elif name == "git":
            output = "git version 2.43.0\n"
        elif len(arguments) > 2 and arguments[1:3] == ["-I", "-c"]:
            module = Path(bootstrap.__file__).resolve()
            digest = hashlib.sha256(module.read_bytes()).hexdigest()
            output = "NUMPILOT_ASNUMPY_SMOKE=" + json.dumps(
                {
                    "asnumpy_version": bootstrap.ASNUMPY_VERSION,
                    "numpy_version": "2.1.0",
                    "loguru_version": "0.7.3",
                    "module_path": str(module),
                    "module_sha256": digest,
                    "native_modules": [
                        {"path": module.name, "sha256": digest}
                    ],
                    "shape": [1000, 1000],
                    "dtype": "float32",
                },
                sort_keys=True,
            )
        else:
            output = "installed\n"
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    return run


def _apply_environment(tmp_path):
    toolkit = tmp_path / "cann"
    opp = toolkit / "opp"
    opp.mkdir(parents=True)
    return {
        "ASCEND_TOOLKIT_HOME": str(toolkit),
        "ASCEND_OPP_PATH": str(opp),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
    }


def _wheel(
    tmp_path,
    payload=b"verified local wheel fixture",
    distribution="asnumpy",
    version=None,
    tag="py3-none-any",
):
    version = version or bootstrap.ASNUMPY_VERSION
    path = tmp_path / ("%s-%s-%s.whl" % (distribution, version, tag))
    dist_info = "%s-%s.dist-info" % (distribution, version)
    metadata_payload = (
        "Metadata-Version: 2.1\n"
        "Name: %s\n"
        "Version: %s\n"
        "Requires-Python: >=3.10\n"
        "Requires-Dist: numpy>=2.0\n"
        "Requires-Dist: loguru>=0.7.0\n\n" % (distribution, version)
    ).encode("utf-8")
    wheel_payload = (
        "Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: %s\n" % tag
    ).encode("utf-8")
    bound_payload = Path(bootstrap.__file__).resolve().read_bytes()
    members = {
        dist_info + "/METADATA": metadata_payload,
        dist_info + "/WHEEL": wheel_payload,
        "asnumpy/__init__.py": bound_payload,
        "asnumpy/_core.fixture.so": bound_payload,
        "asnumpy/fixture.bin": payload,
    }
    record_name = dist_info + "/RECORD"
    record_lines = []
    for name, content in members.items():
        encoded = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=")
        record_lines.append(
            "%s,sha256=%s,%d" % (name, encoded.decode("ascii"), len(content))
        )
    record_lines.append(record_name + ",,")
    with ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
        archive.writestr(record_name, "\n".join(record_lines) + "\n")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_default_plan_is_json_safe_and_never_fetches(monkeypatch, tmp_path):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("read-only plan attempted acquisition")

    monkeypatch.setattr(bootstrap, "fetch_asnumpy_checkout", forbidden)
    monkeypatch.setattr(bootstrap, "fetch_pinned_skill", forbidden)
    calls = []
    plan = build_bootstrap_plan(
        environment=_apply_environment(tmp_path),
        runner=_successful_runner(calls),
        tool_finder=_tool_finder,
        version_reader=_version_reader,
    )

    json.dumps(plan)
    assert plan["mode"] == "read-only-plan"
    assert plan["python"]["minimum"] == "3.10"
    assert plan["asnumpy_installation"]["version_matches_request"] is False
    assert set(plan["caches"]) == {"asnumpy", "cannbot"}
    assert plan["build_tools"]["cmake"]["ok"] is True
    assert plan["build_tools"]["gcc"]["ok"] is True
    assert plan["python_packages"]["runtime"]["numpy"]["ok"] is True
    assert plan["python_packages"]["runtime"]["loguru"]["ok"] is True
    assert all(
        item["ok"] for item in plan["python_packages"]["source_build"].values()
    )
    assert plan["npu_smi"]["available"] is True
    assert plan["network_required"] is False
    assert not any(
        command[1:2] in (["fetch"], ["clone"])
        or "pip" in command
        for command, _kwargs in calls
    )


@pytest.mark.parametrize("value", ["1.0rc1", "0.12.1.dev0", "build 999 only"])
def test_version_parser_rejects_prerelease_or_unqualified_banners(value):
    assert bootstrap._parse_version(value) is None


def test_version_parser_uses_a_dotted_tool_version_not_an_unrelated_integer():
    assert bootstrap._parse_version("build 999 / version 3.1.0") == (3, 1, 0)


def test_source_build_plan_marks_missing_fetch_and_build_as_network(tmp_path):
    calls = []
    plan = build_bootstrap_plan(
        build_asnumpy=True,
        environment=_apply_environment(tmp_path),
        runner=_successful_runner(calls),
        tool_finder=_tool_finder,
        version_reader=_version_reader,
    )
    action = next(
        item
        for item in plan["planned_actions"]
        if item["action"] == "verify_or_fetch_pinned_asnumpy_source"
    )
    assert action["network"] is True
    build = next(
        item
        for item in plan["planned_actions"]
        if item["action"] == "build_asnumpy_wheel"
    )
    assert build["network"] is True
    assert plan["network_required"] is True


def test_cached_source_build_is_still_conservatively_marked_network(tmp_path):
    cache = tmp_path / "cached-source"
    cache.mkdir()
    plan = build_bootstrap_plan(
        build_asnumpy=True,
        asnumpy_cache=cache,
        environment=_apply_environment(tmp_path),
        runner=_successful_runner([]),
        tool_finder=_tool_finder,
        version_reader=_version_reader,
    )

    fetch = next(
        item
        for item in plan["planned_actions"]
        if item["action"] == "verify_or_fetch_pinned_asnumpy_source"
    )
    build = next(
        item
        for item in plan["planned_actions"]
        if item["action"] == "build_asnumpy_wheel"
    )
    assert fetch["network"] is False
    assert build["network"] is True
    assert plan["network_required"] is True


def test_load_asnumpy_checkout_verifies_commit_origin_tree_and_hashes(tmp_path):
    root, commit = _source_repo(tmp_path)
    checkout = load_asnumpy_checkout(root, expected_commit=commit)

    assert checkout.commit == commit
    assert checkout.tree == _git(root, "rev-parse", "HEAD^{tree}")
    assert checkout.manifest()["files_sha256"]["pyproject.toml"] == hashlib.sha256(
        (root / "pyproject.toml").read_bytes()
    ).hexdigest()
    assert str(root) not in json.dumps(checkout.manifest())


@pytest.mark.parametrize("failure", ["commit", "origin", "dirty"])
def test_load_asnumpy_checkout_rejects_wrong_identity_or_dirty_tree(tmp_path, failure):
    root, commit = _source_repo(tmp_path)
    expected_commit = commit
    expected_repository = ASNUMPY_REPOSITORY
    if failure == "commit":
        expected_commit = "0" * 40
    elif failure == "origin":
        _git(root, "remote", "set-url", "origin", "https://example.invalid/asnumpy.git")
    else:
        (root / "pyproject.toml").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(BootstrapError, match=failure if failure != "origin" else "origin"):
        load_asnumpy_checkout(
            root,
            expected_commit=expected_commit,
            expected_repository=expected_repository,
        )


def test_load_asnumpy_checkout_rejects_symlink(tmp_path):
    root, commit = _source_repo(tmp_path)
    link = tmp_path / "source-link"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(BootstrapError, match="real directory"):
        load_asnumpy_checkout(link, expected_commit=commit)


def test_fetch_uses_exact_commit_and_atomically_publishes(monkeypatch, tmp_path):
    calls = []
    target = tmp_path / "cache" / ASNUMPY_COMMIT

    def fake_load(path, **_kwargs):
        return AsNumpyCheckout(
            root=str(path),
            repository=ASNUMPY_REPOSITORY,
            commit=ASNUMPY_COMMIT,
            tree=ASNUMPY_TREE,
            license="Apache-2.0",
            files_sha256={},
        )

    monkeypatch.setattr(bootstrap, "load_asnumpy_checkout", fake_load)
    checkout = fetch_asnumpy_checkout(
        target, runner=_successful_runner(calls)
    )

    assert target.is_dir()
    assert checkout.commit == ASNUMPY_COMMIT
    commands = [command for command, _kwargs in calls]
    fetch = next(command for command in commands if "fetch" in command)
    assert fetch[-1] == ASNUMPY_COMMIT
    assert "master" not in fetch
    assert all(isinstance(command, list) for command in commands)
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700


def test_apply_rejects_cannbot_before_acquisition_without_license(tmp_path):
    wheel, digest = _wheel(tmp_path)
    called = []
    with pytest.raises(BootstrapError, match="license acceptance"):
        apply_bootstrap(
            receipt_path=tmp_path / "receipt.json",
            asnumpy_wheel=wheel,
            expected_wheel_sha256=digest,
            use_cannbot=True,
            cannbot_acquirer=lambda _path: called.append(True),
            runner=_successful_runner([]),
        )

    assert called == []
    receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["ready"] is False


def test_apply_rejects_wheel_hash_before_pip(tmp_path):
    wheel, _digest = _wheel(tmp_path)
    calls = []
    with pytest.raises(BootstrapError, match="SHA-256 mismatch"):
        apply_bootstrap(
            receipt_path=tmp_path / "receipt.json",
            asnumpy_wheel=wheel,
            expected_wheel_sha256="0" * 64,
            environment=_apply_environment(tmp_path),
            runner=_successful_runner(calls),
            tool_finder=_tool_finder,
            version_reader=_version_reader,
        )

    assert not any("pip" in command for command, _kwargs in calls)
    receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["ready"] is False


def test_apply_rejects_non_asnumpy_wheel_before_pip(tmp_path):
    wheel, digest = _wheel(tmp_path, distribution="not_asnumpy")
    calls = []

    with pytest.raises(BootstrapError, match="distribution is not AsNumpy"):
        apply_bootstrap(
            receipt_path=tmp_path / "receipt.json",
            asnumpy_wheel=wheel,
            expected_wheel_sha256=digest,
            environment=_apply_environment(tmp_path),
            runner=_successful_runner(calls),
            tool_finder=_tool_finder,
            version_reader=_version_reader,
        )

    assert not any("pip" in command for command, _kwargs in calls)


def test_apply_rejects_non_archive_wheel_before_pip(tmp_path):
    wheel = tmp_path / "asnumpy-0-py3-none-any.whl"
    wheel.write_bytes(b"not a wheel archive")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    calls = []

    with pytest.raises(BootstrapError, match="readable wheel archive"):
        apply_bootstrap(
            receipt_path=tmp_path / "receipt.json",
            asnumpy_wheel=wheel,
            expected_wheel_sha256=digest,
            environment=_apply_environment(tmp_path),
            runner=_successful_runner(calls),
            tool_finder=_tool_finder,
            version_reader=_version_reader,
        )

    assert not any("pip" in command for command, _kwargs in calls)


def test_wheel_hash_can_select_an_audited_profile(monkeypatch, tmp_path):
    wheel, digest = _wheel(tmp_path, tag="cp310-cp310-linux_aarch64")
    profile = {
        "profile_id": "fixture-profile",
        "version": bootstrap.ASNUMPY_VERSION,
        "required_tag": "cp310-cp310-linux_aarch64",
        "qualification_scope": "fixture-only",
    }
    monkeypatch.setitem(bootstrap._AUDITED_WHEEL_PROFILES, digest, profile)

    identity = bootstrap._verify_wheel(wheel, digest)

    assert identity["trust"]["kind"] == "numpilot-audited-profile"
    assert identity["trust"]["profile_id"] == "fixture-profile"
    assert identity["installed_files"]["module"]["path"] == "asnumpy/__init__.py"
    assert len(identity["installed_files"]["native_modules"]) == 1


def test_audited_profile_requires_exact_runtime_versions(monkeypatch, tmp_path):
    wheel, digest = _wheel(tmp_path, tag="cp310-cp310-linux_aarch64")
    profile = {
        "profile_id": "fixture-profile",
        "version": bootstrap.ASNUMPY_VERSION,
        "required_tag": "cp310-cp310-linux_aarch64",
        "runtime_constraints": {
            "python_implementation": bootstrap.platform.python_implementation(),
            "python_major_minor": "%d.%d" % bootstrap.sys.version_info[:2],
            "machine": bootstrap.platform.machine(),
            "packages": {"numpy": "1.26.4", "loguru": "0.7.3"},
        },
        "qualification_scope": "fixture-only",
    }
    monkeypatch.setitem(bootstrap._AUDITED_WHEEL_PROFILES, digest, profile)
    plan = build_bootstrap_plan(
        asnumpy_wheel=wheel,
        expected_wheel_sha256=digest,
        environment=_apply_environment(tmp_path),
        runner=_successful_runner([]),
        tool_finder=_tool_finder,
        version_reader=_version_reader,
    )

    assert any(
        "numpy == 1.26.4" in issue
        for issue in bootstrap._environment_issues(plan, require_venv=False)
    )


def test_apply_requires_active_venv_without_injected_runner(monkeypatch, tmp_path):
    wheel, digest = _wheel(tmp_path)
    monkeypatch.setattr(bootstrap.sys, "prefix", "/system")
    monkeypatch.setattr(bootstrap.sys, "base_prefix", "/system")

    with pytest.raises(BootstrapError, match="active virtual environment"):
        apply_bootstrap(
            receipt_path=tmp_path / "receipt.json",
            asnumpy_wheel=wheel,
            expected_wheel_sha256=digest,
            environment=_apply_environment(tmp_path),
            tool_finder=lambda name: "/bin/true" if name == "npu-smi" else None,
            version_reader=_version_reader,
        )

    receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["ready"] is False


def test_apply_verified_wheel_uses_current_interpreter_runner_and_receipt(tmp_path):
    wheel, digest = _wheel(tmp_path)
    calls = []
    receipt_path = tmp_path / "receipts" / "bootstrap.json"

    result = apply_bootstrap(
        receipt_path=receipt_path,
        asnumpy_wheel=wheel,
        expected_wheel_sha256=digest,
        environment=_apply_environment(tmp_path),
        runner=_successful_runner(calls),
        tool_finder=_tool_finder,
        version_reader=_version_reader,
    )

    pip_command = next(command for command, _kwargs in calls if "pip" in command)
    assert pip_command[:5] == [
        bootstrap.sys.executable,
        "-m",
        "pip",
        "--isolated",
        "install",
    ]
    assert "--no-index" in pip_command
    assert "--no-deps" in pip_command
    assert "--force-reinstall" in pip_command
    assert pip_command[-1].endswith(wheel.name)
    assert result["status"] == "ready"
    assert result["ready"] is True
    assert result["wheel"]["kind"] == "provided"
    assert result["wheel"]["sha256"] == digest
    assert result["wheel"]["distribution"] == "asnumpy"
    assert result["wheel"]["version"] == bootstrap.ASNUMPY_VERSION
    assert result["wheel"]["trust"]["kind"] == "caller-supplied-hash"
    assert result["repositories"]["asnumpy"]["source_repository_claimed"] is False
    assert result["verification"]["asnumpy"]["status"] == "passed"
    assert result["verification"]["asnumpy"]["wheel_file_identity_verified"] is True
    smoke = next(
        command
        for command, _kwargs in calls
        if command[:3] == [bootstrap.sys.executable, "-I", "-c"]
    )
    assert "ap.ones((1000, 1000), dtype=ap.float32)" in smoke[3]
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == result
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
    assert list(receipt_path.parent.glob(".*.tmp")) == []


def test_failed_pip_atomically_replaces_applying_receipt_with_failed(tmp_path):
    wheel, digest = _wheel(tmp_path)

    def runner(arguments, **_kwargs):
        if "pip" in arguments:
            return SimpleNamespace(returncode=9, stdout="secret output", stderr="token=x")
        name = Path(arguments[0]).name
        output = {
            "cmake": "cmake version 3.26.1",
            "g++": "g++ (GCC) 11.2.0",
            "git": "git version 2.40.0",
        }.get(name, "")
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    receipt_path = tmp_path / "receipt.json"
    with pytest.raises(BootstrapError, match="subprocess failed"):
        apply_bootstrap(
            receipt_path=receipt_path,
            asnumpy_wheel=wheel,
            expected_wheel_sha256=digest,
            environment=_apply_environment(tmp_path),
            runner=runner,
            tool_finder=_tool_finder,
            version_reader=_version_reader,
        )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["ready"] is False
    assert "secret output" not in json.dumps(receipt)
    assert "token=x" not in json.dumps(receipt)


def test_source_build_fails_closed_before_acquisition_when_tools_are_missing(tmp_path):
    acquired = []
    with pytest.raises(BootstrapError, match="source build requires"):
        apply_bootstrap(
            receipt_path=tmp_path / "receipt.json",
            build_asnumpy=True,
            environment=_apply_environment(tmp_path),
            runner=_successful_runner([]),
            tool_finder=lambda name: "/test/npu-smi" if name == "npu-smi" else None,
            version_reader=lambda _name: (_ for _ in ()).throw(LookupError()),
            asnumpy_acquirer=lambda _path: acquired.append(True),
        )

    assert acquired == []
    receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["ready"] is False


def test_source_build_fails_before_acquisition_when_python_build_deps_are_missing(
    tmp_path,
):
    acquired = []

    def only_runtime_packages(name):
        versions = {"numpy": "2.1.0", "loguru": "0.7.3"}
        if name not in versions:
            raise LookupError(name)
        return versions[name]

    with pytest.raises(BootstrapError, match="source build requires build"):
        apply_bootstrap(
            receipt_path=tmp_path / "receipt.json",
            build_asnumpy=True,
            environment=_apply_environment(tmp_path),
            runner=_successful_runner([]),
            tool_finder=_tool_finder,
            version_reader=only_runtime_packages,
            asnumpy_acquirer=lambda _path: acquired.append(True),
        )

    assert acquired == []


def test_accepted_cannbot_bundle_is_bound_into_success_receipt(tmp_path):
    wheel, digest = _wheel(tmp_path)

    class Bundle:
        def manifest(self):
            return {
                "repository": bootstrap.CANNBOT_REPOSITORY,
                "commit": bootstrap.CANNBOT_COMMIT,
                "license": bootstrap.CANNBOT_LICENSE,
                "bundle_sha256": "1" * 64,
                "files": {
                    item["path"]: "2" * 64
                    for item in bootstrap.CANNBOT_ASCENDC_SKILL_CATALOG
                },
            }

    result = apply_bootstrap(
        receipt_path=tmp_path / "receipt.json",
        asnumpy_wheel=wheel,
        expected_wheel_sha256=digest,
        use_cannbot=True,
        accept_cannbot_license=True,
        environment=_apply_environment(tmp_path),
        runner=_successful_runner([]),
        tool_finder=_tool_finder,
        version_reader=_version_reader,
        cannbot_acquirer=lambda _path: Bundle(),
    )

    assert result["repositories"]["cannbot"]["license_accepted"] is True
    assert result["repositories"]["cannbot"]["bundle_sha256"] == "1" * 64
    assert result["verification"]["cannbot"]["status"] == "passed"
    assert result["verification"]["cannbot"]["skills"][0]["name"] == (
        "ascendc-registry-invoke-template"
    )


def test_asnumpy_verification_runs_official_readme_smoke():
    calls = []
    result = verify_asnumpy_installation(runner=_successful_runner(calls))

    assert result["status"] == "passed"
    assert result["shape"] == [1000, 1000]
    assert calls[0][0][:3] == [bootstrap.sys.executable, "-I", "-c"]
    assert "arr.shape" in calls[0][0][3]
    assert "arr.dtype != ap.float32" in calls[0][0][3]


def test_asnumpy_verification_rejects_files_not_bound_to_verified_wheel():
    expected = {
        "version": bootstrap.ASNUMPY_VERSION,
        "installed_files": {
            "module": {"path": "asnumpy/__init__.py", "sha256": "0" * 64},
            "native_modules": [
                {"path": "asnumpy/_core.fixture.so", "sha256": "0" * 64}
            ],
        },
    }
    with pytest.raises(BootstrapError, match="verified wheel RECORD"):
        verify_asnumpy_installation(
            runner=_successful_runner([]),
            expected_wheel=expected,
        )


def test_asnumpy_verification_reports_missing_distribution_before_subprocess(
    monkeypatch,
):
    def missing(_name):
        raise bootstrap.metadata.PackageNotFoundError("asnumpy")

    monkeypatch.setattr(bootstrap.metadata, "version", missing)
    with pytest.raises(BootstrapError, match="not installed in the current interpreter"):
        verify_asnumpy_installation()


def test_bootstrap_verification_answers_cannbot_skill_question_without_fetch(
    tmp_path,
):
    root = tmp_path / "cannbot"
    root.mkdir()

    class Bundle:
        def manifest(self):
            return {
                "repository": bootstrap.CANNBOT_REPOSITORY,
                "commit": bootstrap.CANNBOT_COMMIT,
                "license": bootstrap.CANNBOT_LICENSE,
                "bundle_sha256": "3" * 64,
                "files": {
                    item["path"]: "4" * 64
                    for item in bootstrap.CANNBOT_ASCENDC_SKILL_CATALOG
                },
            }

    result = verify_bootstrap_installation(
        use_cannbot=True,
        accept_cannbot_license=True,
        cannbot_cache=root,
        runner=_successful_runner([]),
        cannbot_loader=lambda path: Bundle() if path == root else None,
    )

    assert result["network_requested_by_numpilot"] is False
    assert result["asnumpy"]["status"] == "passed"
    assert result["cannbot"]["status"] == "passed"
    assert "哪些能参考" in result["cannbot"]["question"]
    assert {item["name"] for item in result["cannbot"]["skills"]} >= {
        "ascendc-api-best-practices",
        "ascendc-ut-develop",
        "ascendc-runtime-debug",
        "ascendc-perf-optimize",
    }


def test_bootstrap_verification_never_fetches_missing_cannbot_cache(tmp_path):
    with pytest.raises(BootstrapError, match="cache is missing"):
        verify_bootstrap_installation(
            use_cannbot=True,
            accept_cannbot_license=True,
            cannbot_cache=tmp_path / "missing",
            runner=_successful_runner([]),
        )
