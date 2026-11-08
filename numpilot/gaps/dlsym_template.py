"""Render fail-closed explicit-handle ACLNN symbol resolution C++ headers."""

from __future__ import annotations

from dataclasses import dataclass
import re

from numpilot.gaps.generated_adapter import CUSTOM_OPAPI_SONAME, SYMBOL_RESOLUTION_STRATEGY

_OPERATOR = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_ACLNN = re.compile(r"^aclnn[A-Za-z0-9_]{1,127}$")
_CPP_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENV_VAR = "NUMPILOT_CUSTOM_OPAPI_LIBRARY"


class DlsymTemplateError(ValueError):
    """A generated dlsym template request is invalid."""


def _operator(value: str) -> str:
    if not isinstance(value, str) or _OPERATOR.fullmatch(value) is None:
        raise DlsymTemplateError("operator symbol is invalid")
    return value


def _aclnn(value: str, label: str) -> str:
    if not isinstance(value, str) or _ACLNN.fullmatch(value) is None:
        raise DlsymTemplateError(f"{label} is invalid")
    return value


def _cpp_ident(value: str, label: str) -> str:
    if not isinstance(value, str) or _CPP_IDENT.fullmatch(value) is None:
        raise DlsymTemplateError(f"{label} is invalid")
    return value


def _type_name(symbol: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in symbol.split("_"))


@dataclass(frozen=True)
class DlsymTemplateSpec:
    """Inputs for one generated custom-op explicit-handle resolver."""

    operator: str
    get_workspace_size: str
    execute: str
    library_soname: str = CUSTOM_OPAPI_SONAME
    env_var: str = _ENV_VAR

    def __post_init__(self) -> None:
        _operator(self.operator)
        get_symbol = _aclnn(self.get_workspace_size, "GetWorkspaceSize symbol")
        execute_symbol = _aclnn(self.execute, "execute symbol")
        if get_symbol == execute_symbol or not get_symbol.endswith("GetWorkspaceSize"):
            raise DlsymTemplateError("ACLNN symbols must be distinct two-stage names")
        if self.library_soname != CUSTOM_OPAPI_SONAME:
            raise DlsymTemplateError("custom OPP library must be libcust_opapi.so")
        _cpp_ident(self.env_var, "environment variable")
        if self.env_var != _ENV_VAR:
            raise DlsymTemplateError("custom OPP environment variable is unsupported")

    @property
    def type_name(self) -> str:
        return _type_name(self.operator) + "DlsymSymbols"

    @property
    def resolver_name(self) -> str:
        return "Resolve" + _type_name(self.operator) + "DlsymSymbols"

    def render_header(self) -> str:
        """Return deterministic C++17 resolver header text.

        The generated header only resolves function pointers from the custom
        OPP handle. It deliberately does not issue a global ACLNN call.
        The caller supplies operator-specific function pointer types because
        ACLNN first-stage signatures differ between operators.
        """
        return f'''#pragma once

#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>
#include <string>

namespace numpilot_generated {{

inline constexpr const char* k{_type_name(self.operator)}GetWorkspaceSizeSymbol = "{self.get_workspace_size}";
inline constexpr const char* k{_type_name(self.operator)}ExecuteSymbol = "{self.execute}";
inline constexpr const char* k{_type_name(self.operator)}LibrarySoname = "{self.library_soname}";
inline constexpr const char* k{_type_name(self.operator)}LibraryEnv = "{self.env_var}";

// Strategy: {SYMBOL_RESOLUTION_STRATEGY}. Keep this handle alive for the
// process lifetime so function pointers remain valid after resolution.
template <typename GetWorkspaceFn, typename ExecuteFn>
struct {self.type_name} {{
    void* handle = nullptr;
    GetWorkspaceFn get_workspace_size = nullptr;
    ExecuteFn execute = nullptr;
}};

template <typename GetWorkspaceFn, typename ExecuteFn>
{self.type_name}<GetWorkspaceFn, ExecuteFn>& {self.resolver_name}() {{
    static {self.type_name}<GetWorkspaceFn, ExecuteFn> symbols;
    static std::once_flag once;
    std::call_once(once, [] {{
        const char* configured = std::getenv(k{_type_name(self.operator)}LibraryEnv);
        const char* library = (configured != nullptr && configured[0] != '\\0')
            ? configured : k{_type_name(self.operator)}LibrarySoname;
        int flags = RTLD_NOW | RTLD_LOCAL;
#ifdef RTLD_DEEPBIND
        flags |= RTLD_DEEPBIND;
#endif
        symbols.handle = dlopen(library, flags);
        if (symbols.handle == nullptr) {{
            const char* detail = dlerror();
            throw std::runtime_error(std::string("cannot load custom OPP library ") + library +
                (detail != nullptr ? std::string(": ") + detail : std::string()));
        }}
        dlerror();
        symbols.get_workspace_size = reinterpret_cast<GetWorkspaceFn>(
            dlsym(symbols.handle, k{_type_name(self.operator)}GetWorkspaceSizeSymbol));
        const char* get_error = dlerror();
        symbols.execute = reinterpret_cast<ExecuteFn>(
            dlsym(symbols.handle, k{_type_name(self.operator)}ExecuteSymbol));
        const char* execute_error = dlerror();
        if (get_error != nullptr || execute_error != nullptr ||
            symbols.get_workspace_size == nullptr || symbols.execute == nullptr) {{
            std::string detail = get_error != nullptr ? get_error :
                (execute_error != nullptr ? execute_error : "missing ACLNN symbols");
            dlclose(symbols.handle);
            symbols.handle = nullptr;
            throw std::runtime_error("custom OPP ACLNN symbol resolution failed: " + detail);
        }}
    }});
    return symbols;
}}

}}  // namespace numpilot_generated
'''


__all__ = ["DlsymTemplateError", "DlsymTemplateSpec"]
