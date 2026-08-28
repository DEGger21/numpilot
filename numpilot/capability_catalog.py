"""Versioned, inert catalog of the pinned AsNumpy public compute surface.

The catalog is the single checked-in source of API-name truth.  Runtime
availability and execution provenance remain separate concerns owned by the
backend capability layer.  Regeneration is deliberately based on Python AST
only: importing AsNumpy would initialise CANN and therefore is not an
acceptable registry drift check.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import PurePosixPath
import re
from types import MappingProxyType
from typing import Any, Dict, Mapping, Tuple


CAPABILITY_CATALOG_SCHEMA_VERSION = 1
ASNUMPY_CATALOG_REPOSITORY = "https://gitcode.com/cann/asnumpy.git"
ASNUMPY_CATALOG_COMMIT = "1929a3ba44eb532a9f544f774a249cdf1c4119be"
ASNUMPY_CATALOG_TREE = "bf2f353825f60bd96f910c3a90fd5b75d4c3c8d0"

ASNUMPY_TOP_LEVEL_SOURCE = "src/asnumpy/__init__.py"
ASNUMPY_LINALG_SOURCE = "src/asnumpy/linalg/__init__.py"
ASNUMPY_RANDOM_SOURCE = "src/asnumpy/random/__init__.py"
ASNUMPY_CATALOG_SOURCE_PATHS = frozenset({
    ASNUMPY_TOP_LEVEL_SOURCE,
    ASNUMPY_LINALG_SOURCE,
    ASNUMPY_RANDOM_SOURCE,
})

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")
_EXPORT_TARGET = re.compile(
    r"^(?:__all__|\.[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*)$"
)
_MAX_CANONICAL_KEY_CHARS = 512
_MAX_CANONICAL_KEY_PARTS = 8
_MAX_SOURCE_CHARS = 4 * 1024 * 1024


class CapabilityCatalogError(ValueError):
    """The checked-in catalog or one static source extraction is invalid."""


def validate_canonical_key(value: str) -> str:
    """Validate an internal key such as ``add`` or ``linalg.det``."""

    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > _MAX_CANONICAL_KEY_CHARS
    ):
        raise CapabilityCatalogError("canonical capability key is invalid")
    parts = value.split(".")
    if not parts or len(parts) > _MAX_CANONICAL_KEY_PARTS:
        raise CapabilityCatalogError("canonical capability key depth is invalid")
    if parts[0] in {"numpy", "asnumpy", "operator"}:
        raise CapabilityCatalogError(
            "canonical capability key must not contain a source namespace"
        )
    for part in parts:
        if (
            not _IDENTIFIER.fullmatch(part)
            or part.startswith("__")
            or part.endswith("__")
        ):
            raise CapabilityCatalogError(
                "canonical capability key contains an unsafe component"
            )
    return value


def source_api_from_canonical_key(key: str) -> str:
    """Return the external NumPy spelling for one canonical catalog key."""

    return "numpy." + validate_canonical_key(key)


def asnumpy_api_from_canonical_key(key: str) -> str:
    """Return the public AsNumpy spelling for one canonical catalog key."""

    return "asnumpy." + validate_canonical_key(key)


def canonical_key_from_source_api(source_api: str) -> str:
    """Convert an exact ``numpy.*`` source API to its canonical catalog key."""

    if (
        not isinstance(source_api, str)
        or not source_api.startswith("numpy.")
        or source_api != source_api.strip()
    ):
        raise CapabilityCatalogError("source API must be an exact numpy.* path")
    return validate_canonical_key(source_api[len("numpy."):])


def validate_catalog_source_path(value: str) -> str:
    """Require a bounded, repository-relative AsNumpy Python source path."""

    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 512
        or "\\" in value
    ):
        raise CapabilityCatalogError("catalog source path is invalid")
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise CapabilityCatalogError("catalog source path is unsafe")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or tuple(path.parts[:2]) != ("src", "asnumpy")
        or path.suffix != ".py"
    ):
        raise CapabilityCatalogError(
            "catalog source path must be a relative AsNumpy Python file"
        )
    return value


@dataclass(frozen=True)
class CapabilityCatalogEntry:
    """One canonical API and the pinned source export that declares it."""

    canonical_key: str
    source_api: str
    asnumpy_api: str
    source_file: str
    export_target: str

    def __post_init__(self) -> None:
        key = validate_canonical_key(self.canonical_key)
        if self.source_api != source_api_from_canonical_key(key):
            raise CapabilityCatalogError(
                "catalog source API does not match its canonical key"
            )
        if self.asnumpy_api != asnumpy_api_from_canonical_key(key):
            raise CapabilityCatalogError(
                "catalog AsNumpy API does not match its canonical key"
            )
        source_file = validate_catalog_source_path(self.source_file)
        namespace = key.split(".", 1)[0] if "." in key else None
        expected_source = {
            None: ASNUMPY_TOP_LEVEL_SOURCE,
            "linalg": ASNUMPY_LINALG_SOURCE,
            "random": ASNUMPY_RANDOM_SOURCE,
        }.get(namespace)
        if expected_source is None or source_file != expected_source:
            raise CapabilityCatalogError(
                "catalog source file does not match its canonical namespace"
            )
        if (
            not isinstance(self.export_target, str)
            or not _EXPORT_TARGET.fullmatch(self.export_target)
            or (namespace is None and self.export_target == "__all__")
            or (namespace is not None and self.export_target != "__all__")
        ):
            raise CapabilityCatalogError("catalog export target is invalid")

    def to_dict(self) -> Dict[str, str]:
        return {
            "canonical_key": self.canonical_key,
            "source_api": self.source_api,
            "asnumpy_api": self.asnumpy_api,
            "source_file": self.source_file,
            "export_target": self.export_target,
        }


@dataclass(frozen=True)
class CapabilityCatalog:
    """Immutable catalog bound to one exact AsNumpy source identity."""

    repository: str
    commit: str
    tree: str
    entries: Tuple[CapabilityCatalogEntry, ...]
    schema_version: int = CAPABILITY_CATALOG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CAPABILITY_CATALOG_SCHEMA_VERSION:
            raise CapabilityCatalogError("unsupported capability catalog schema")
        if (
            not isinstance(self.repository, str)
            or not self.repository
            or self.repository != self.repository.strip()
            or len(self.repository) > 512
        ):
            raise CapabilityCatalogError("capability catalog repository is invalid")
        if not isinstance(self.commit, str) or not _HEX40.fullmatch(self.commit):
            raise CapabilityCatalogError("capability catalog commit is invalid")
        if not isinstance(self.tree, str) or not _HEX40.fullmatch(self.tree):
            raise CapabilityCatalogError("capability catalog tree is invalid")
        if (
            not isinstance(self.entries, tuple)
            or not self.entries
            or any(
                not isinstance(entry, CapabilityCatalogEntry)
                for entry in self.entries
            )
        ):
            raise CapabilityCatalogError("capability catalog entries are invalid")
        keys = tuple(entry.canonical_key for entry in self.entries)
        if len(set(keys)) != len(keys):
            raise CapabilityCatalogError("capability catalog contains duplicate keys")
        if keys != tuple(sorted(keys)):
            raise CapabilityCatalogError(
                "capability catalog entries must use canonical key order"
            )

    @property
    def canonical_keys(self) -> frozenset[str]:
        return frozenset(entry.canonical_key for entry in self.entries)

    @property
    def top_level_keys(self) -> frozenset[str]:
        return frozenset(key for key in self.canonical_keys if "." not in key)

    @property
    def entry_map(self) -> Mapping[str, CapabilityCatalogEntry]:
        return MappingProxyType({
            entry.canonical_key: entry for entry in self.entries
        })

    def keys_for_namespace(self, namespace: str) -> frozenset[str]:
        namespace = validate_canonical_key(namespace)
        if "." in namespace:
            raise CapabilityCatalogError("catalog namespace must be top-level")
        prefix = namespace + "."
        return frozenset(
            key for key in self.canonical_keys if key.startswith(prefix)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": {
                "repository": self.repository,
                "commit": self.commit,
                "tree": self.tree,
            },
            "entries": [entry.to_dict() for entry in self.entries],
        }


@dataclass(frozen=True)
class CapabilityCatalogDrift:
    """Deterministic difference between an extracted and expected catalog."""

    missing_keys: Tuple[str, ...] = ()
    unexpected_keys: Tuple[str, ...] = ()
    changed_entries: Tuple[str, ...] = ()
    identity_mismatches: Tuple[str, ...] = ()

    @property
    def matches(self) -> bool:
        return not any((
            self.missing_keys,
            self.unexpected_keys,
            self.changed_entries,
            self.identity_mismatches,
        ))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matches": self.matches,
            "missing_keys": list(self.missing_keys),
            "unexpected_keys": list(self.unexpected_keys),
            "changed_entries": list(self.changed_entries),
            "identity_mismatches": list(self.identity_mismatches),
        }


# Every other _LAZY_MAPPING key is conservatively treated as a compute API.
# This allowlist prevents known modules/types/I/O helpers from entering the
# compatibility view while ensuring a new unknown export produces CI drift.
_NON_COMPUTE_LAZY_EXPORTS = {
    "linalg": ".linalg",
    "random": ".random",
    "save": ".io",
    "savez": ".io",
    "savez_compressed": ".io",
    "load": ".io",
    "ArrayLike": "._types",
    "DTypeLike": "._types",
    "ShapeLike": "._types",
    "AxisLike": "._types",
    "AxisOptional": "._types",
    "ScalarLike": "._types",
    "broadcast_shape": ".utils",
    "ndarray": ".utils",
}


_TOP_LEVEL_EXPORTS = {
    ".array": """
        empty empty_like eye full full_like identity linspace ones ones_like
        zeros zeros_like
    """.split(),
    ".linalg.direct": "dot einsum inner matmul outer vdot".split(),
    ".logic": """
        all any equal greater greater_equal isfinite isinf isneginf isposinf
        less less_equal logical_and logical_not logical_or logical_xor not_equal
    """.split(),
    ".math": """
        absolute add amax amin around arccos arccosh arcsin arcsinh arctan
        arctan2 arctanh ceil clip copysign cos cosh cross cumprod cumsum
        deg2rad degrees divide divmod exp exp2 expm1 fabs fix floor
        floor_divide fmax fmin fmod float_power gelu gcd heaviside hypot lcm
        ldexp log log10 log1p log2 logaddexp logaddexp2 max maximum min minimum
        mod modf multiply nan_to_num nancumprod nancumsum nanmax nanprod nansum
        negative power positive prod rad2deg radians real reciprocal relu
        remainder rint round_ sign signbit sin sinc sinh sqrt square subtract
        sum tan tanh true_divide trunc
    """.split(),
    ".sorting": ["sort"],
    ".statistics": ["mean"],
    ".nn": ["softmax"],
}

_LINALG_EXPORTS = (
    "det",
    "inv",
    "matrix_power",
    "norm",
    "qr",
    "slogdet",
)
_RANDOM_EXPORTS = (
    "binomial",
    "exponential",
    "geometric",
    "gumbel",
    "laplace",
    "logistic",
    "lognormal",
    "normal",
    "pareto",
    "rayleigh",
    "standard_cauchy",
    "standard_normal",
    "uniform",
    "weibull",
)


def _entry(
    canonical_key: str,
    source_file: str,
    export_target: str,
) -> CapabilityCatalogEntry:
    return CapabilityCatalogEntry(
        canonical_key=canonical_key,
        source_api=source_api_from_canonical_key(canonical_key),
        asnumpy_api=asnumpy_api_from_canonical_key(canonical_key),
        source_file=source_file,
        export_target=export_target,
    )


def _fixed_entries() -> Tuple[CapabilityCatalogEntry, ...]:
    entries = []
    for target, names in _TOP_LEVEL_EXPORTS.items():
        entries.extend(
            _entry(name, ASNUMPY_TOP_LEVEL_SOURCE, target)
            for name in names
        )
    entries.extend(
        _entry("linalg." + name, ASNUMPY_LINALG_SOURCE, "__all__")
        for name in _LINALG_EXPORTS
    )
    entries.extend(
        _entry("random." + name, ASNUMPY_RANDOM_SOURCE, "__all__")
        for name in _RANDOM_EXPORTS
    )
    return tuple(sorted(entries, key=lambda item: item.canonical_key))


ASNUMPY_CAPABILITY_CATALOG = CapabilityCatalog(
    repository=ASNUMPY_CATALOG_REPOSITORY,
    commit=ASNUMPY_CATALOG_COMMIT,
    tree=ASNUMPY_CATALOG_TREE,
    entries=_fixed_entries(),
)


def _parse_module(source: str, label: str) -> ast.Module:
    if not isinstance(source, str) or len(source) > _MAX_SOURCE_CHARS:
        raise CapabilityCatalogError("%s source is invalid or too large" % label)
    try:
        source_bytes = source.encode("utf-8")
    except UnicodeError as exc:
        raise CapabilityCatalogError("%s source is not UTF-8" % label) from exc
    if len(source_bytes) > _MAX_SOURCE_CHARS:
        raise CapabilityCatalogError("%s source is invalid or too large" % label)
    try:
        return ast.parse(source, filename=label)
    except SyntaxError as exc:
        raise CapabilityCatalogError("%s source has invalid syntax" % label) from exc


def _assignment_value(tree: ast.Module, name: str, label: str) -> ast.AST:
    matches = []
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if (
            len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == name
        ):
            matches.append(statement.value)
    if len(matches) != 1:
        raise CapabilityCatalogError(
            "%s must define exactly one literal %s" % (label, name)
        )
    return matches[0]


def _literal_string_mapping(node: ast.AST, label: str) -> Dict[str, str]:
    if not isinstance(node, ast.Dict):
        raise CapabilityCatalogError("%s must be a literal dictionary" % label)
    result = {}
    for key_node, value_node in zip(node.keys, node.values):
        if not (
            isinstance(key_node, ast.Constant)
            and type(key_node.value) is str
            and isinstance(value_node, ast.Constant)
            and type(value_node.value) is str
        ):
            raise CapabilityCatalogError(
                "%s must contain only literal string pairs" % label
            )
        key = key_node.value
        if key in result:
            raise CapabilityCatalogError("%s contains a duplicate key" % label)
        result[key] = value_node.value
    return result


def _literal_string_sequence(node: ast.AST, label: str) -> Tuple[str, ...]:
    if not isinstance(node, (ast.List, ast.Tuple)):
        raise CapabilityCatalogError("%s must be a literal string sequence" % label)
    result = []
    for item in node.elts:
        if not isinstance(item, ast.Constant) or type(item.value) is not str:
            raise CapabilityCatalogError(
                "%s must contain only literal strings" % label
            )
        result.append(item.value)
    if len(set(result)) != len(result):
        raise CapabilityCatalogError("%s contains duplicate exports" % label)
    return tuple(result)


def extract_asnumpy_capability_catalog(
    sources: Mapping[str, str],
    *,
    repository: str = ASNUMPY_CATALOG_REPOSITORY,
    commit: str = ASNUMPY_CATALOG_COMMIT,
    tree: str = ASNUMPY_CATALOG_TREE,
) -> CapabilityCatalog:
    """Extract the public compute catalog without importing any source.

    ``sources`` must contain exactly the three pinned, repository-relative
    files used by the catalog.  This makes an accidental scan of a checkout or
    an unrelated source file impossible at the API boundary.
    """

    if not isinstance(sources, Mapping):
        raise CapabilityCatalogError("catalog sources must be a mapping")
    normalized_sources = {}
    for raw_path, source in sources.items():
        path = validate_catalog_source_path(raw_path)
        if path in normalized_sources:
            raise CapabilityCatalogError("catalog sources contain duplicate paths")
        normalized_sources[path] = source
    if frozenset(normalized_sources) != ASNUMPY_CATALOG_SOURCE_PATHS:
        raise CapabilityCatalogError(
            "catalog source files must match the three pinned inputs"
        )

    top_tree = _parse_module(
        normalized_sources[ASNUMPY_TOP_LEVEL_SOURCE],
        ASNUMPY_TOP_LEVEL_SOURCE,
    )
    lazy_mapping = _literal_string_mapping(
        _assignment_value(
            top_tree,
            "_LAZY_MAPPING",
            ASNUMPY_TOP_LEVEL_SOURCE,
        ),
        "AsNumpy _LAZY_MAPPING",
    )
    for required_module in ("linalg", "random"):
        expected_target = _NON_COMPUTE_LAZY_EXPORTS[required_module]
        if lazy_mapping.get(required_module) != expected_target:
            raise CapabilityCatalogError(
                "AsNumpy lazy namespace export drifted: %s" % required_module
            )
    for name, expected_target in _NON_COMPUTE_LAZY_EXPORTS.items():
        if name in lazy_mapping and lazy_mapping[name] != expected_target:
            raise CapabilityCatalogError(
                "AsNumpy non-compute export drifted: %s" % name
            )

    entries = []
    for name, target in lazy_mapping.items():
        if name in _NON_COMPUTE_LAZY_EXPORTS:
            continue
        key = validate_canonical_key(name)
        if "." in key:
            raise CapabilityCatalogError(
                "top-level AsNumpy export contains a nested key"
            )
        entries.append(_entry(key, ASNUMPY_TOP_LEVEL_SOURCE, target))

    for namespace, path in (
        ("linalg", ASNUMPY_LINALG_SOURCE),
        ("random", ASNUMPY_RANDOM_SOURCE),
    ):
        tree_node = _parse_module(normalized_sources[path], path)
        exports = _literal_string_sequence(
            _assignment_value(tree_node, "__all__", path),
            "%s __all__" % namespace,
        )
        for name in exports:
            if "." in name:
                raise CapabilityCatalogError(
                    "%s __all__ contains a nested export" % namespace
                )
            entries.append(
                _entry(namespace + "." + name, path, "__all__")
            )

    return CapabilityCatalog(
        repository=repository,
        commit=commit,
        tree=tree,
        entries=tuple(sorted(entries, key=lambda item: item.canonical_key)),
    )


def compare_capability_catalog(
    observed: CapabilityCatalog,
    expected: CapabilityCatalog = ASNUMPY_CAPABILITY_CATALOG,
) -> CapabilityCatalogDrift:
    """Compare one AST extraction with the checked-in catalog."""

    if not isinstance(observed, CapabilityCatalog) or not isinstance(
        expected, CapabilityCatalog
    ):
        raise TypeError("catalog comparison requires CapabilityCatalog values")
    observed_map = observed.entry_map
    expected_map = expected.entry_map
    observed_keys = set(observed_map)
    expected_keys = set(expected_map)
    changed = tuple(sorted(
        key
        for key in observed_keys & expected_keys
        if observed_map[key] != expected_map[key]
    ))
    identity_mismatches = tuple(
        name
        for name in ("schema_version", "repository", "commit", "tree")
        if getattr(observed, name) != getattr(expected, name)
    )
    return CapabilityCatalogDrift(
        missing_keys=tuple(sorted(expected_keys - observed_keys)),
        unexpected_keys=tuple(sorted(observed_keys - expected_keys)),
        changed_entries=changed,
        identity_mismatches=identity_mismatches,
    )


def require_no_capability_catalog_drift(
    observed: CapabilityCatalog,
    expected: CapabilityCatalog = ASNUMPY_CAPABILITY_CATALOG,
) -> None:
    """Raise one bounded diagnostic when a static extraction has drifted."""

    drift = compare_capability_catalog(observed, expected)
    if drift.matches:
        return
    raise CapabilityCatalogError(
        "AsNumpy capability catalog drift: missing=%r unexpected=%r "
        "changed=%r identity=%r"
        % (
            drift.missing_keys,
            drift.unexpected_keys,
            drift.changed_entries,
            drift.identity_mismatches,
        )
    )
