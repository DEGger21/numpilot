import hashlib
import os
from pathlib import Path

import pytest

from numpilot.asnumpy_adapter_context import (
    AsnumpyAdapterContextError,
    EXISTING_MODULE_MODE,
    LEGACY_ARCHITECTURE,
    MANIFEST_ARCHITECTURE,
    MANIFEST_MODE,
    MANUAL_MODE,
    OFFICIAL_ACLNN_ARCHITECTURE,
    infer_asnumpy_adapter_api,
    plan_asnumpy_adapter_context,
    plan_official_aclnn_adapter_context,
)


MANIFEST_CONTEXT_PATHS = (
    "scripts/gen_custom_ops.py",
    "csrc/custom_ops/registry/angle_v2/op.yaml",
    "tests/asnumpy_tests/custom_ops_tests/test_angle_v2.py",
    "CMakeLists.txt",
    "cmake/CustomOps.cmake",
    "csrc/custom_ops/CMakeLists.txt",
    "include/asnumpy/custom_ops/loader.hpp",
    "csrc/custom_ops/loader.cpp",
)

LEGACY_CONTEXT_PATHS = (
    "include/asnumpy/nn/activation.hpp",
    "csrc/nn/activation.cpp",
    "bindings/python/bind_nn.cpp",
    "bindings/python/module.cpp",
    "src/asnumpy/nn.py",
    "src/asnumpy/__init__.py",
    "bindings/python/CMakeLists.txt",
    "tests/asnumpy_tests/nn_tests/test_activation.py",
)

OFFICIAL_MATMUL_CONTEXT_PATHS = (
    "include/asnumpy/linalg/product.hpp",
    "csrc/linalg/product.cpp",
    "csrc/linalg/CMakeLists.txt",
    "bindings/python/bind_linalg.cpp",
    "bindings/python/CMakeLists.txt",
    "src/asnumpy/linalg/direct.py",
    "src/asnumpy/linalg/__init__.py",
    "tests/asnumpy_tests/linalg_tests/test_product.py",
    "src/asnumpy/utils.py",
    "bindings/python/bind_utils.cpp",
)
OFFICIAL_MATMUL_LIBRARY = "aarch64-linux/lib64/libopapi.so"
OFFICIAL_MATMUL_LIBRARY_SHA256 = "a" * 64
OFFICIAL_MATMUL_LINK_BLOCK = (
    "target_link_libraries(_core\n"
    "    PUBLIC\n"
    "    ascendcl\n"
    "    runtime\n"
    "    nnopbase\n"
    "    opapi\n"
    "    Python::NumPy\n"
    ")\n"
    "target_link_libraries(_core\n"
    "    PUBLIC\n"
    "    array\n"
    "    linalg\n"
    ")\n"
)


def _write_context(root: Path, paths) -> None:
    for index, relative in enumerate(paths, start=1):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "fixture %d for %s\n" % (index, relative),
            encoding="utf-8",
        )


def _manifest_tree(tmp_path: Path) -> Path:
    root = tmp_path / "manifest-asnumpy"
    _write_context(root, MANIFEST_CONTEXT_PATHS)
    return root


def _legacy_tree(tmp_path: Path) -> Path:
    root = tmp_path / "legacy-asnumpy"
    _write_context(root, LEGACY_CONTEXT_PATHS)
    return root


def _official_matmul_tree(tmp_path: Path) -> Path:
    root = tmp_path / "official-matmul-asnumpy"
    _write_context(root, OFFICIAL_MATMUL_CONTEXT_PATHS)
    (root / "bindings/python/CMakeLists.txt").write_text(
        OFFICIAL_MATMUL_LINK_BLOCK,
        encoding="utf-8",
    )
    return root


def test_manifest_plan_matches_existing_cli_context_and_overlay_contract(tmp_path):
    root = _manifest_tree(tmp_path)

    plan = plan_asnumpy_adapter_context(root, "cbrt")

    assert plan.architecture == MANIFEST_ARCHITECTURE
    assert plan.mode == MANIFEST_MODE
    assert tuple(plan.context) == MANIFEST_CONTEXT_PATHS
    assert plan.expected_overlay_paths == (
        "csrc/custom_ops/registry/cbrt/op.yaml",
        "tests/asnumpy_tests/custom_ops_tests/test_cbrt.py",
    )
    assert plan.total_bytes == sum(
        len(content.encode("utf-8")) for content in plan.context.values()
    )
    assert len(plan.plan_sha256) == 64


def test_legacy_plan_covers_complete_manual_call_chain_with_eight_files(tmp_path):
    root = _legacy_tree(tmp_path)

    plan = plan_asnumpy_adapter_context(root, "smooth_l1_loss")

    assert plan.architecture == LEGACY_ARCHITECTURE
    assert plan.mode == MANUAL_MODE
    assert tuple(plan.context) == LEGACY_CONTEXT_PATHS
    assert len(plan.context) == 8
    assert plan.expected_overlay_paths == (
        "include/asnumpy/custom_ops/smooth_l1_loss.hpp",
        "csrc/custom_ops/smooth_l1_loss.cpp",
        "bindings/python/bind_custom_ops.cpp",
        "bindings/python/module.cpp",
        "src/asnumpy/custom_ops.py",
        "src/asnumpy/__init__.py",
        "bindings/python/CMakeLists.txt",
        "tests/asnumpy_tests/custom_ops_tests/test_smooth_l1_loss.py",
    )
    assert len(plan.expected_overlay_paths) == 8
    assert not (root / "include/asnumpy/custom_ops/smooth_l1_loss.hpp").exists()
    assert not (root / "src/asnumpy/custom_ops.py").exists()


def test_official_aclnn_matmul_uses_reviewed_existing_module_plan(tmp_path):
    root = _official_matmul_tree(tmp_path)

    plan = plan_official_aclnn_adapter_context(
        root,
        "matmul",
        selected_library_relative_path=OFFICIAL_MATMUL_LIBRARY,
        selected_library_sha256=OFFICIAL_MATMUL_LIBRARY_SHA256,
    )

    assert plan.architecture == OFFICIAL_ACLNN_ARCHITECTURE
    assert plan.mode == EXISTING_MODULE_MODE
    assert plan.schema_version == 3
    assert tuple(plan.context) == OFFICIAL_MATMUL_CONTEXT_PATHS
    assert plan.expected_overlay_paths == (
        "csrc/linalg/product.cpp",
        "src/asnumpy/linalg/direct.py",
        "tests/asnumpy_tests/linalg_tests/test_product.py",
    )
    assert not any("custom_ops" in path for path in plan.expected_overlay_paths)
    assert plan.manifest()["prompt_dependencies"] == {
        "csrc/linalg/product.cpp": [
            {
                "source": "reference",
                "path": "include/asnumpy/linalg/product.hpp",
            },
            {
                "source": "signed_official_header",
                "path": "SIGNED_OFFICIAL_HEADER/aclnn_matmul.h",
            },
        ],
        "src/asnumpy/linalg/direct.py": [
            {
                "source": "reference",
                "path": "bindings/python/bind_linalg.cpp",
            },
        ],
        "tests/asnumpy_tests/linalg_tests/test_product.py": [
            {
                "source": "generated_overlay",
                "path": "src/asnumpy/linalg/direct.py",
            },
        ],
    }
    assert plan.manifest()["official_library_binding"] == {
        "schema_version": 1,
        "proof_kind": "reviewed_exact_cmake_target_link_item",
        "cmake_path": "bindings/python/CMakeLists.txt",
        "cmake_sha256": plan.context_sha256[
            "bindings/python/CMakeLists.txt"
        ],
        "target": "_core",
        "link_item": "opapi",
        "selected_library_relative_path": OFFICIAL_MATMUL_LIBRARY,
        "selected_library_basename": "libopapi.so",
        "selected_library_sha256": OFFICIAL_MATMUL_LIBRARY_SHA256,
    }
    assert len(plan.plan_sha256) == 64


def test_official_aclnn_planner_rejects_unreviewed_api_before_reading_tree(
    tmp_path,
):
    root = _official_matmul_tree(tmp_path)

    with pytest.raises(AsnumpyAdapterContextError, match="not reviewed"):
        plan_official_aclnn_adapter_context(
            root,
            "cbrt",
            selected_library_relative_path=OFFICIAL_MATMUL_LIBRARY,
            selected_library_sha256=OFFICIAL_MATMUL_LIBRARY_SHA256,
        )


def test_official_aclnn_planner_rejects_symbol_compatible_unlinked_dso(
    tmp_path,
):
    root = _official_matmul_tree(tmp_path)

    with pytest.raises(AsnumpyAdapterContextError, match="link graph"):
        plan_official_aclnn_adapter_context(
            root,
            "matmul",
            selected_library_relative_path=(
                "aarch64-linux/lib64/libopapi_nn.so"
            ),
            selected_library_sha256="b" * 64,
        )


def test_official_aclnn_planner_rejects_reviewed_link_block_drift(tmp_path):
    root = _official_matmul_tree(tmp_path)
    (root / "bindings/python/CMakeLists.txt").write_text(
        OFFICIAL_MATMUL_LINK_BLOCK.replace("    opapi\n", "    opapi_nn\n"),
        encoding="utf-8",
    )

    with pytest.raises(AsnumpyAdapterContextError, match="link graph"):
        plan_official_aclnn_adapter_context(
            root,
            "matmul",
            selected_library_relative_path=OFFICIAL_MATMUL_LIBRARY,
            selected_library_sha256=OFFICIAL_MATMUL_LIBRARY_SHA256,
        )


def test_custom_opp_planner_does_not_silently_select_official_strategy(tmp_path):
    root = _official_matmul_tree(tmp_path)

    with pytest.raises(AsnumpyAdapterContextError):
        plan_asnumpy_adapter_context(root, "matmul")


def test_context_content_and_plan_are_exactly_hash_bound_and_immutable(tmp_path):
    root = _legacy_tree(tmp_path)
    plan = plan_asnumpy_adapter_context(root, "erf")

    for relative, content in plan.context.items():
        payload = (root / relative).read_bytes()
        assert content.encode("utf-8") == payload
        assert plan.context_sha256[relative] == hashlib.sha256(payload).hexdigest()
    with pytest.raises(TypeError):
        plan.context["new"] = "unsafe"
    with pytest.raises(TypeError):
        plan.context_sha256["new"] = "0" * 64
    receipt = plan.manifest()
    assert "context" not in receipt
    assert receipt["context_sha256"] == dict(plan.context_sha256)
    assert receipt["plan_sha256"] == plan.plan_sha256


def test_plan_digest_changes_when_target_context_changes(tmp_path):
    root = _legacy_tree(tmp_path)
    before = plan_asnumpy_adapter_context(root, "erf")
    source = root / "src/asnumpy/nn.py"
    source.write_text(source.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")

    after = plan_asnumpy_adapter_context(root, "erf")

    assert before.context_sha256["src/asnumpy/nn.py"] != (
        after.context_sha256["src/asnumpy/nn.py"]
    )
    assert before.plan_sha256 != after.plan_sha256


def test_incomplete_manifest_architecture_fails_closed(tmp_path):
    root = _legacy_tree(tmp_path)
    generator = root / "scripts/gen_custom_ops.py"
    generator.parent.mkdir(parents=True)
    generator.write_text("fixture\n", encoding="utf-8")

    with pytest.raises(AsnumpyAdapterContextError, match="incomplete"):
        plan_asnumpy_adapter_context(root, "cbrt")


def test_symlink_reference_root_and_context_file_are_rejected(tmp_path):
    root = _legacy_tree(tmp_path)
    root_link = tmp_path / "asnumpy-link"
    root_link.symlink_to(root, target_is_directory=True)
    with pytest.raises(AsnumpyAdapterContextError, match="real directory"):
        plan_asnumpy_adapter_context(root_link, "cbrt")

    outside = tmp_path / "outside.hpp"
    outside.write_text("outside\n", encoding="utf-8")
    header = root / "include/asnumpy/nn/activation.hpp"
    header.unlink()
    header.symlink_to(outside)
    with pytest.raises(AsnumpyAdapterContextError, match="symlink"):
        plan_asnumpy_adapter_context(root, "cbrt")


def test_existing_symlink_on_allowed_overlay_path_is_rejected(tmp_path):
    root = _legacy_tree(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("outside\n", encoding="utf-8")
    target = root / "src/asnumpy/custom_ops.py"
    target.symlink_to(outside)

    with pytest.raises(AsnumpyAdapterContextError, match="overlay.*symlink"):
        plan_asnumpy_adapter_context(root, "cbrt")


def test_non_regular_entry_on_allowed_overlay_path_is_rejected(tmp_path):
    root = _legacy_tree(tmp_path)
    (root / "src/asnumpy/custom_ops.py").mkdir()

    with pytest.raises(AsnumpyAdapterContextError, match="unsafe entry"):
        plan_asnumpy_adapter_context(root, "cbrt")


@pytest.mark.parametrize(
    "api_name",
    ("../cbrt", "Cbrt", "numpy.cbrt", "a" * 65, "cbrt\\escape", ""),
)
def test_unsafe_api_names_cannot_expand_overlay_allow_list(tmp_path, api_name):
    root = _legacy_tree(tmp_path)

    with pytest.raises(AsnumpyAdapterContextError, match="API name"):
        plan_asnumpy_adapter_context(root, api_name)


def test_missing_or_oversized_required_context_fails_closed(tmp_path):
    missing_root = _legacy_tree(tmp_path / "missing")
    (missing_root / "bindings/python/module.cpp").unlink()
    with pytest.raises(AsnumpyAdapterContextError):
        plan_asnumpy_adapter_context(missing_root, "cbrt")

    oversized_root = _legacy_tree(tmp_path / "oversized")
    (oversized_root / "include/asnumpy/nn/activation.hpp").write_text(
        "x" * (96 * 1024 + 1),
        encoding="utf-8",
    )
    with pytest.raises(AsnumpyAdapterContextError, match="too large"):
        plan_asnumpy_adapter_context(oversized_root, "cbrt")


def test_required_context_fifo_is_rejected_without_blocking(tmp_path):
    root = _legacy_tree(tmp_path)
    fifo = root / "include/asnumpy/nn/activation.hpp"
    fifo.unlink()
    os.mkfifo(fifo)

    with pytest.raises(AsnumpyAdapterContextError, match="regular file"):
        plan_asnumpy_adapter_context(root, "cbrt")


@pytest.mark.parametrize(
    ("paths", "expected"),
    (
        (
            (
                "csrc/custom_ops/registry/cbrt/op.yaml",
                "tests/asnumpy_tests/custom_ops_tests/test_cbrt.py",
            ),
            "cbrt",
        ),
        (
            (
                "include/asnumpy/custom_ops/smooth_l1_loss.hpp",
                "csrc/custom_ops/smooth_l1_loss.cpp",
                "bindings/python/module.cpp",
                "tests/asnumpy_tests/custom_ops_tests/test_smooth_l1_loss.py",
            ),
            "smooth_l1_loss",
        ),
    ),
)
def test_overlay_paths_recover_one_layout_independent_api(paths, expected):
    assert infer_asnumpy_adapter_api(paths) == expected


@pytest.mark.parametrize(
    "paths",
    (
        ("bindings/python/module.cpp",),
        (
            "csrc/custom_ops/registry/cbrt/op.yaml",
            "tests/asnumpy_tests/custom_ops_tests/test_erf.py",
        ),
        ("../cbrt.cpp",),
        (
            "csrc/custom_ops/registry/cbrt/op.yaml",
            "csrc/custom_ops/registry/cbrt/op.yaml",
        ),
    ),
)
def test_overlay_api_inference_fails_closed_for_missing_conflicting_or_unsafe_paths(paths):
    with pytest.raises(AsnumpyAdapterContextError):
        infer_asnumpy_adapter_api(paths)
