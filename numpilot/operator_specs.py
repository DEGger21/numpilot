"""声明式领域算子契约及其安全静态解析。

``operator_spec`` 在运行时只给 NumPy oracle 附加不可变元数据；迁移分析不会
导入或执行用户源码，而是只接受由真实 ``numpilot.operator_spec`` 导入绑定的、
全部参数均为 Python 字面量的装饰器。
"""

import ast
from dataclasses import dataclass
import re
from typing import Any, Dict, Mapping, Tuple


OPERATOR_SPEC_SCHEMA_VERSION = 1
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}$")
_TARGET_API = re.compile(r"^asnumpy\.([A-Za-z][A-Za-z0-9_]{0,127})$")
_ORACLE = re.compile(
    r"^(?:self|[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*)$"
)
_ALLOWED_PROVENANCE = frozenset({
    "ops_nn_reuse_candidate",
    "cann_reuse_candidate",
    "asnumpy_composite_candidate",
    "custom_generation_candidate",
})
_ALLOWED_ROUTES = frozenset({
    "cann_existing_reuse_candidate",
    "asnumpy_composite_candidate",
    "ascendc_registry_candidate",
})
_FIELDS = frozenset({
    "name",
    "version",
    "numpy_oracle",
    "target_api",
    "provenance",
    "route",
    "signature_variants",
    "dtype_rules",
    "shape_rules",
    "source",
    "license",
    "target_soc",
    "cann_version",
})


def _text(value: Any, label: str, *, limit: int = 512) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError("operator spec %s must be non-empty bounded text" % label)
    return value


def _rules(value: Any, label: str) -> Tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or not value or len(value) > 64:
        raise ValueError("operator spec %s must be a non-empty sequence" % label)
    rules = tuple(_text(item, label, limit=128) for item in value)
    if len(set(rules)) != len(rules):
        raise ValueError("operator spec %s contains duplicates" % label)
    return rules


def _signatures(value: Any) -> Tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (tuple, list)) or not value or len(value) > 64:
        raise ValueError("operator spec signature_variants must be non-empty")
    result = []
    identities = set()
    for raw in value:
        if not isinstance(raw, Mapping) or frozenset(raw) != {
            "positional_args", "keyword_args"
        }:
            raise ValueError("operator spec signature variant fields are invalid")
        positional = raw["positional_args"]
        keywords = raw["keyword_args"]
        if (
            isinstance(positional, bool)
            or not isinstance(positional, int)
            or positional < 0
            or positional > 64
        ):
            raise ValueError("operator spec positional_args is invalid")
        if not isinstance(keywords, (tuple, list)) or len(keywords) > 64:
            raise ValueError("operator spec keyword_args is invalid")
        normalized = tuple(sorted(_text(item, "keyword", limit=128) for item in keywords))
        if len(set(normalized)) != len(normalized):
            raise ValueError("operator spec keyword_args contains duplicates")
        identity = (positional, normalized)
        if identity in identities:
            raise ValueError("operator spec signature_variants contains duplicates")
        identities.add(identity)
        result.append({
            "positional_args": positional,
            "keyword_args": list(normalized),
        })
    return tuple(sorted(
        result,
        key=lambda item: (item["positional_args"], tuple(item["keyword_args"])),
    ))


@dataclass(frozen=True)
class OperatorSpec:
    """一个领域 helper 到一个公开 AsNumpy API 的显式、可哈希契约。"""

    name: str
    version: str
    numpy_oracle: str
    target_api: str
    provenance: str
    route: str
    signature_variants: Tuple[Mapping[str, Any], ...]
    dtype_rules: Tuple[str, ...]
    shape_rules: Tuple[str, ...]
    source: str
    license: str
    target_soc: str
    cann_version: str
    schema_version: int = OPERATOR_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OPERATOR_SPEC_SCHEMA_VERSION:
            raise ValueError("unsupported operator spec schema")
        if not isinstance(self.name, str) or not _NAME.fullmatch(self.name):
            raise ValueError("operator spec name is invalid")
        if not isinstance(self.version, str) or not _VERSION.fullmatch(self.version):
            raise ValueError("operator spec version is invalid")
        if not isinstance(self.numpy_oracle, str) or not _ORACLE.fullmatch(
            self.numpy_oracle
        ):
            raise ValueError("operator spec numpy_oracle is invalid")
        target = _TARGET_API.fullmatch(self.target_api or "")
        if target is None or target.group(1) != self.name:
            raise ValueError("operator spec target_api must be asnumpy.<name>")
        if self.provenance not in _ALLOWED_PROVENANCE:
            raise ValueError("operator spec provenance is invalid")
        if self.route not in _ALLOWED_ROUTES:
            raise ValueError("operator spec route is invalid")
        object.__setattr__(
            self,
            "signature_variants",
            _signatures(self.signature_variants),
        )
        object.__setattr__(self, "dtype_rules", _rules(self.dtype_rules, "dtype_rules"))
        object.__setattr__(self, "shape_rules", _rules(self.shape_rules, "shape_rules"))
        for field_name in ("source", "license", "target_soc", "cann_version"):
            _text(getattr(self, field_name), field_name)

    @property
    def qualified_api(self) -> str:
        return "operator." + self.name

    def admits_call(self, positional_args: int, keyword_args: Tuple[str, ...]) -> bool:
        identity = (int(positional_args), tuple(sorted(keyword_args)))
        return any(
            identity == (
                int(item["positional_args"]),
                tuple(item["keyword_args"]),
            )
            for item in self.signature_variants
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "version": self.version,
            "numpy_oracle": self.numpy_oracle,
            "target_api": self.target_api,
            "provenance": self.provenance,
            "route": self.route,
            "signature_variants": [dict(item) for item in self.signature_variants],
            "dtype_rules": list(self.dtype_rules),
            "shape_rules": list(self.shape_rules),
            "source": self.source,
            "license": self.license,
            "target_soc": self.target_soc,
            "cann_version": self.cann_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OperatorSpec":
        if not isinstance(data, Mapping):
            raise ValueError("operator spec must be an object")
        if frozenset(data) != _FIELDS | {"schema_version"}:
            raise ValueError("operator spec fields are invalid")
        return cls(
            schema_version=data["schema_version"],
            name=data["name"],
            version=data["version"],
            numpy_oracle=data["numpy_oracle"],
            target_api=data["target_api"],
            provenance=data["provenance"],
            route=data["route"],
            signature_variants=tuple(data["signature_variants"]),
            dtype_rules=tuple(data["dtype_rules"]),
            shape_rules=tuple(data["shape_rules"]),
            source=data["source"],
            license=data["license"],
            target_soc=data["target_soc"],
            cann_version=data["cann_version"],
        )


def operator_spec(**kwargs):
    """Attach a validated :class:`OperatorSpec` to a NumPy oracle function."""

    spec = OperatorSpec(**kwargs)

    def decorate(function):
        if not callable(function):
            raise TypeError("operator_spec can decorate only a callable")
        oracle_names = {
            "self",
            function.__name__,
            function.__module__ + "." + function.__name__,
        }
        if spec.numpy_oracle not in oracle_names:
            raise ValueError("operator spec numpy_oracle does not name its function")
        setattr(function, "__numpilot_operator_spec__", spec)
        return function

    return decorate


@dataclass(frozen=True)
class StaticOperatorSpecs:
    specs: Mapping[str, OperatorSpec]
    errors: Mapping[str, str]


def _literal_kwargs(call: ast.Call) -> Dict[str, Any]:
    if call.args:
        raise ValueError("operator_spec accepts keyword arguments only")
    values = {}
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg in values:
            raise ValueError("operator_spec contains invalid keyword arguments")
        if keyword.arg not in _FIELDS:
            raise ValueError("operator_spec contains an unknown field")
        try:
            values[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError, MemoryError, RecursionError) as exc:
            raise ValueError("operator_spec fields must be literals") from exc
    if frozenset(values) != _FIELDS:
        missing = ", ".join(sorted(_FIELDS - set(values)))
        raise ValueError("operator_spec is missing fields: " + missing)
    return values


def _decorator_bindings(tree: ast.Module) -> Tuple[set, set]:
    modules = set()
    functions = set()
    rebound = set()
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                local = alias.asname or alias.name.split(".", 1)[0]
                if alias.name == "numpilot":
                    modules.add(local)
                else:
                    rebound.add(local)
        elif isinstance(statement, ast.ImportFrom):
            for alias in statement.names:
                local = alias.asname or alias.name
                if statement.level == 0 and statement.module == "numpilot" and alias.name == "operator_spec":
                    functions.add(local)
                else:
                    rebound.add(local)
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            rebound.add(statement.name)
        else:
            for node in ast.walk(statement):
                if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                    rebound.add(node.id)
    return modules - rebound, functions - rebound


def collect_static_operator_specs(tree: ast.Module) -> StaticOperatorSpecs:
    """Parse trusted decorator syntax without importing or executing the source."""

    modules, functions = _decorator_bindings(tree)
    specs = {}
    errors = {}
    semantic_names = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        recognized = []
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            target = decorator.func
            direct = isinstance(target, ast.Name) and target.id in functions
            qualified = (
                isinstance(target, ast.Attribute)
                and target.attr == "operator_spec"
                and isinstance(target.value, ast.Name)
                and target.value.id in modules
            )
            if direct or qualified:
                recognized.append(decorator)
        if not recognized:
            continue
        if len(recognized) != 1:
            errors[node.name] = "helper 必须恰有一个 numpilot.operator_spec 装饰器"
            continue
        try:
            values = _literal_kwargs(recognized[0])
            spec = OperatorSpec(**values)
            allowed_oracles = {"self", node.name}
            if spec.numpy_oracle not in allowed_oracles and not spec.numpy_oracle.endswith(
                "." + node.name
            ):
                raise ValueError("operator spec numpy_oracle does not name its helper")
            previous = semantic_names.get(spec.name)
            if previous is not None and previous != node.name:
                raise ValueError("operator spec semantic name is duplicated")
            semantic_names[spec.name] = node.name
            specs[node.name] = spec
        except (TypeError, ValueError) as exc:
            errors[node.name] = str(exc)
    return StaticOperatorSpecs(specs=specs, errors=errors)
