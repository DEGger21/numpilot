import pytest

from numpilot.capability_catalog import (
    ASNUMPY_CAPABILITY_CATALOG,
    ASNUMPY_LINALG_SOURCE,
    ASNUMPY_RANDOM_SOURCE,
    ASNUMPY_TOP_LEVEL_SOURCE,
    CapabilityCatalog,
    CapabilityCatalogEntry,
    CapabilityCatalogError,
    canonical_key_from_source_api,
    compare_capability_catalog,
    extract_asnumpy_capability_catalog,
    source_api_from_canonical_key,
    validate_canonical_key,
    validate_catalog_source_path,
)


def _entry(key, source_file, export_target):
    return CapabilityCatalogEntry(
        canonical_key=key,
        source_api="numpy." + key,
        asnumpy_api="asnumpy." + key,
        source_file=source_file,
        export_target=export_target,
    )


def _small_expected_catalog():
    return CapabilityCatalog(
        repository="https://example.invalid/asnumpy.git",
        commit="1" * 40,
        tree="2" * 40,
        entries=(
            _entry("add", ASNUMPY_TOP_LEVEL_SOURCE, ".math"),
            _entry("linalg.det", ASNUMPY_LINALG_SOURCE, "__all__"),
            _entry("random.normal", ASNUMPY_RANDOM_SOURCE, "__all__"),
        ),
    )


def _small_sources(*, top_name="add", top_target=".math"):
    return {
        ASNUMPY_TOP_LEVEL_SOURCE: (
            "_LAZY_MAPPING = {\n"
            f"    {top_name!r}: {top_target!r},\n"
            "    'linalg': '.linalg',\n"
            "    'random': '.random',\n"
            "}\n"
        ),
        ASNUMPY_LINALG_SOURCE: "__all__ = ['det']\n",
        ASNUMPY_RANDOM_SOURCE: "__all__ = ['normal']\n",
    }


def test_fixed_catalog_is_versioned_and_has_the_exact_public_counts():
    catalog = ASNUMPY_CAPABILITY_CATALOG

    assert catalog.schema_version == 1
    assert len(catalog.top_level_keys) == 122
    assert len(catalog.keys_for_namespace("linalg")) == 6
    assert len(catalog.keys_for_namespace("random")) == 14
    assert len(catalog.canonical_keys) == 142
    assert "angle" not in catalog.canonical_keys
    assert "linalg.det" in catalog.canonical_keys
    assert "random.normal" in catalog.canonical_keys


def test_catalog_source_identity_matches_the_pinned_bootstrap_contract():
    from numpilot.bootstrap import (
        ASNUMPY_COMMIT,
        ASNUMPY_REPOSITORY,
        ASNUMPY_TREE,
    )

    assert ASNUMPY_CAPABILITY_CATALOG.repository == ASNUMPY_REPOSITORY
    assert ASNUMPY_CAPABILITY_CATALOG.commit == ASNUMPY_COMMIT
    assert ASNUMPY_CAPABILITY_CATALOG.tree == ASNUMPY_TREE


@pytest.mark.parametrize(
    "key",
    ["add", "round_", "linalg.det", "random.standard_normal"],
)
def test_canonical_key_and_source_api_round_trip(key):
    assert validate_canonical_key(key) == key
    source_api = source_api_from_canonical_key(key)
    assert source_api == "numpy." + key
    assert canonical_key_from_source_api(source_api) == key


@pytest.mark.parametrize(
    "key",
    [
        "",
        ".add",
        "add.",
        "linalg..det",
        "numpy.add",
        "linalg.__dict__",
        "linalg.det-name",
    ],
)
def test_canonical_key_rejects_unsafe_or_noncanonical_values(key):
    with pytest.raises(CapabilityCatalogError):
        validate_canonical_key(key)


@pytest.mark.parametrize(
    "path",
    [
        "/src/asnumpy/__init__.py",
        "../src/asnumpy/__init__.py",
        "src/asnumpy/../outside.py",
        "src\\asnumpy\\__init__.py",
        "src/asnumpy/__init__.txt",
    ],
)
def test_catalog_source_path_must_be_safe_relative_python(path):
    with pytest.raises(CapabilityCatalogError):
        validate_catalog_source_path(path)


def test_catalog_entry_binds_canonical_and_source_paths_exactly():
    with pytest.raises(CapabilityCatalogError, match="source API"):
        CapabilityCatalogEntry(
            canonical_key="linalg.det",
            source_api="numpy.det",
            asnumpy_api="asnumpy.linalg.det",
            source_file=ASNUMPY_LINALG_SOURCE,
            export_target="__all__",
        )

    with pytest.raises(CapabilityCatalogError, match="source file"):
        CapabilityCatalogEntry(
            canonical_key="linalg.det",
            source_api="numpy.linalg.det",
            asnumpy_api="asnumpy.linalg.det",
            source_file=ASNUMPY_RANDOM_SOURCE,
            export_target="__all__",
        )


def test_ast_extractor_never_imports_source_and_matches_exact_catalog():
    expected = _small_expected_catalog()
    sources = _small_sources()
    sources[ASNUMPY_TOP_LEVEL_SOURCE] = (
        "raise RuntimeError('AST extraction must not execute this')\n"
        + sources[ASNUMPY_TOP_LEVEL_SOURCE]
    )
    observed = extract_asnumpy_capability_catalog(
        sources,
        repository=expected.repository,
        commit=expected.commit,
        tree=expected.tree,
    )

    assert observed.to_dict() == expected.to_dict()
    assert compare_capability_catalog(observed, expected).matches


def test_drift_comparator_reports_missing_unexpected_and_changed_origin():
    expected = _small_expected_catalog()
    changed_name = extract_asnumpy_capability_catalog(
        _small_sources(top_name="subtract"),
        repository=expected.repository,
        commit=expected.commit,
        tree=expected.tree,
    )
    name_drift = compare_capability_catalog(changed_name, expected)

    assert name_drift.missing_keys == ("add",)
    assert name_drift.unexpected_keys == ("subtract",)
    assert not name_drift.matches

    changed_origin = extract_asnumpy_capability_catalog(
        _small_sources(top_target=".logic"),
        repository=expected.repository,
        commit=expected.commit,
        tree=expected.tree,
    )
    origin_drift = compare_capability_catalog(changed_origin, expected)

    assert origin_drift.changed_entries == ("add",)
    assert not origin_drift.matches


@pytest.mark.parametrize(
    "source",
    [
        "name = 'add'\n_LAZY_MAPPING = {name: '.math'}\n",
        "_LAZY_MAPPING = {'add': target}\n",
        "_LAZY_MAPPING = {'add': '.math', 'add': '.logic'}\n",
    ],
)
def test_ast_extractor_rejects_dynamic_or_duplicate_lazy_mappings(source):
    sources = _small_sources()
    sources[ASNUMPY_TOP_LEVEL_SOURCE] = source

    with pytest.raises(CapabilityCatalogError):
        extract_asnumpy_capability_catalog(sources)


def test_ast_extractor_requires_only_the_three_pinned_source_files():
    missing = _small_sources()
    del missing[ASNUMPY_RANDOM_SOURCE]
    with pytest.raises(CapabilityCatalogError, match="source files"):
        extract_asnumpy_capability_catalog(missing)

    extra = _small_sources()
    extra["src/asnumpy/other.py"] = "__all__ = []\n"
    with pytest.raises(CapabilityCatalogError, match="source files"):
        extract_asnumpy_capability_catalog(extra)
