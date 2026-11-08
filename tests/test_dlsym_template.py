import pytest

from numpilot.gaps.dlsym_template import DlsymTemplateError, DlsymTemplateSpec


def test_rendered_template_is_explicit_handle_only():
    spec = DlsymTemplateSpec(
        operator="softplus",
        get_workspace_size="aclnnSoftplusGetWorkspaceSize",
        execute="aclnnSoftplus",
    )
    rendered = spec.render_header()
    assert spec.type_name == "SoftplusDlsymSymbols"
    assert spec.resolver_name == "ResolveSoftplusDlsymSymbols"
    assert "dlopen(library, flags)" in rendered
    assert "RTLD_LOCAL" in rendered
    assert "RTLD_DEEPBIND" in rendered
    assert "dlsym(symbols.handle" in rendered
    assert "NUMPILOT_CUSTOM_OPAPI_LIBRARY" in rendered
    assert '"libcust_opapi.so"' in rendered
    assert "aclnnSoftplus(" not in rendered
    assert "LD_PRELOAD" not in rendered


def test_rendering_is_deterministic():
    spec = DlsymTemplateSpec("softplus_v2", "aclnnSoftplusV2GetWorkspaceSize", "aclnnSoftplusV2")
    assert spec.render_header() == spec.render_header()
    assert "SoftplusV2DlsymSymbols" in spec.render_header()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"operator": "../softplus"},
        {"get_workspace_size": "aclnnSoftplus"},
        {"execute": "aclnnSoftplusGetWorkspaceSize"},
        {"library_soname": "libopapi.so"},
        {"env_var": "LD_PRELOAD"},
    ],
)
def test_template_rejects_unsafe_or_ambiguous_inputs(kwargs):
    values = {
        "operator": "softplus",
        "get_workspace_size": "aclnnSoftplusGetWorkspaceSize",
        "execute": "aclnnSoftplus",
    }
    values.update(kwargs)
    with pytest.raises(DlsymTemplateError):
        DlsymTemplateSpec(**values)
