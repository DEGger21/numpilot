"""Open-source friendly LLM configuration managed by NumPilot itself.

The repository never contains a real authentication token.  Interactive users
configure the provider through ``python -m numpilot configure-llm``; NumPilot
then stores a small JSON document in the user's config directory with mode
``0600``.  CI can continue to inject the existing ``NUMPILOT_LLM_*``
environment variables, which always take precedence over saved values.
"""
from dataclasses import dataclass
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Dict, Iterable, Optional, Tuple
from urllib.parse import urlsplit


class LlmConfigError(RuntimeError):
    """A local LLM configuration is missing, malformed, or insecure."""


_FIELD_TO_ENV = {
    "base_url": "NUMPILOT_LLM_BASE_URL",
    "auth_token": "NUMPILOT_LLM_AUTH_TOKEN",
    "model": "NUMPILOT_LLM_MODEL",
    "provider": "NUMPILOT_LLM_PROVIDER",
    "protocol": "NUMPILOT_LLM_PROTOCOL",
}
_ALIASES_ENV = "NUMPILOT_LLM_RESPONSE_MODEL_ALIASES"
REQUIRED_LLM_ENV = (
    "NUMPILOT_LLM_BASE_URL",
    "NUMPILOT_LLM_AUTH_TOKEN",
    "NUMPILOT_LLM_MODEL",
)

OPENAI_CHAT_COMPLETIONS_PROTOCOL = "openai-chat-completions"
DEFAULT_PROVIDER_LABEL = "openai-compatible"
MIN_LLM_TIMEOUT_SECONDS = 10.0
MAX_LLM_TIMEOUT_SECONDS = 3600.0
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+@-]{0,511}$")
_PROVIDER_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,127}$")

_LLM_CONFIG_TEMPLATE = {
    "schema_version": 2,
    "base_url": "",
    "auth_token": "",
    "model": "",
    "provider": DEFAULT_PROVIDER_LABEL,
    "protocol": OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    "response_model_aliases": [],
}


def validate_model_id(value: object, *, field: str = "MODEL") -> str:
    """Return an exact request identity, separate from display sanitization."""

    if not isinstance(value, str):
        raise LlmConfigError(f"{field} 必须是字符串")
    model = value
    if (
        not model
        or model != model.strip()
        or _MODEL_ID.fullmatch(model) is None
    ):
        raise LlmConfigError(
            f"{field} 必须是 1～512 字符且不含空白或控制字符"
        )
    return model


def validate_llm_timeout(value: object) -> float:
    """Return the one timeout profile accepted by check, probe, and generation."""

    if isinstance(value, bool):
        raise LlmConfigError("LLM timeout 必须在 10～3600 秒之间")
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise LlmConfigError("LLM timeout 必须在 10～3600 秒之间") from exc
    if not MIN_LLM_TIMEOUT_SECONDS <= timeout <= MAX_LLM_TIMEOUT_SECONDS:
        raise LlmConfigError("LLM timeout 必须在 10～3600 秒之间")
    return timeout


def validate_provider_label(value: object) -> str:
    """Validate an audit label without treating it as an authorization gate."""

    if not isinstance(value, str):
        raise LlmConfigError("PROVIDER 审计标签必须是字符串")
    provider = value.strip()
    if (
        not provider or _PROVIDER_LABEL.fullmatch(provider) is None
    ):
        raise LlmConfigError("PROVIDER 审计标签无效")
    return provider


def validate_protocol(value: object) -> str:
    """Validate the deliberately small protocol contract supported this sprint."""

    if value != OPENAI_CHAT_COMPLETIONS_PROTOCOL:
        raise LlmConfigError(
            "protocol 必须是 %s" % OPENAI_CHAT_COMPLETIONS_PROTOCOL
        )
    return OPENAI_CHAT_COMPLETIONS_PROTOCOL


def normalize_response_model_aliases(
    aliases: object,
    *,
    configured_model: str,
) -> Tuple[str, ...]:
    """Return a bounded, explicit response-model allowlist."""

    if aliases is None:
        values: Iterable[object] = ()
    elif isinstance(aliases, (list, tuple)):
        values = aliases
    else:
        raise LlmConfigError("response_model_aliases 必须是 JSON 字符串数组")
    normalized = []
    for raw in values:
        alias = validate_model_id(raw, field="response_model_aliases")
        if alias == configured_model:
            raise LlmConfigError(
                "response_model_aliases 不应重复 configured model"
            )
        if alias in normalized:
            raise LlmConfigError("response_model_aliases 不能包含重复值")
        normalized.append(alias)
        if len(normalized) > 32:
            raise LlmConfigError("response_model_aliases 最多允许 32 项")
    return tuple(normalized)


def response_model_aliases_from_environment(
    *, configured_model: str
) -> Tuple[str, ...]:
    raw = os.getenv(_ALIASES_ENV, "").strip()
    if not raw:
        return ()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LlmConfigError(
            f"{_ALIASES_ENV} 必须是 JSON 字符串数组"
        ) from exc
    return normalize_response_model_aliases(
        parsed, configured_model=configured_model
    )


def model_identity_sha256(
    *,
    protocol: str,
    provider: str,
    configured_model: str,
    requested_model: str,
    response_model: Optional[str],
    response_model_aliases: object = (),
    response_model_identity: Optional[str] = None,
) -> str:
    """Bind exact identity fields without using display-sanitized values."""

    aliases = normalize_response_model_aliases(
        response_model_aliases, configured_model=configured_model
    )
    if response_model_identity is None:
        if response_model is None:
            response_model_identity = "missing"
        elif response_model == configured_model:
            response_model_identity = "exact"
        elif response_model in aliases:
            response_model_identity = "explicit_alias"
        else:
            response_model_identity = "mismatch"
    identity = {
        "protocol": protocol,
        "provider": provider,
        "configured_model": configured_model,
        "requested_model": requested_model,
        "response_model": response_model,
        "response_model_aliases": list(aliases),
        "response_model_identity": response_model_identity,
    }
    payload = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _env_has(name: str) -> bool:
    return bool((os.getenv(name) or "").strip())


@dataclass(frozen=True)
class LlmConfigStatus:
    path: Path
    configured: bool
    missing: Tuple[str, ...]
    source: str
    model: Optional[str]
    provider: Optional[str]
    token_present: bool
    protocol: str = OPENAI_CHAT_COMPLETIONS_PROTOCOL
    response_model_aliases: Tuple[str, ...] = ()

    def summary(self) -> str:
        """Return a display-safe status string that never contains the token."""
        if not self.configured:
            return "LLM 未配置，缺少: " + ", ".join(self.missing)
        provider = self.provider or "未标注"
        return (
            f"LLM 已配置(source={self.source}, provider={provider}, "
            f"protocol={self.protocol}, model={self.model}, token=已设置)"
        )


def llm_config_path(path: Optional[str] = None) -> Path:
    """Resolve the private per-user JSON path, with test/automation overrides."""
    if path:
        return Path(path).expanduser()
    explicit = os.getenv("NUMPILOT_LLM_CONFIG_FILE")
    if explicit:
        return Path(explicit).expanduser()
    from numpilot.paths import config_home

    return config_home() / "llm.json"


def _read_saved(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    if os.name == "posix":
        mode = path.stat().st_mode & 0o777
        if mode & 0o077:
            raise LlmConfigError(
                f"LLM 配置权限过宽({mode:03o}): {path}；请执行 chmod 600"
            )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LlmConfigError(f"无法读取 LLM 配置 {path}: {type(exc).__name__}") from exc
    if not isinstance(raw, dict):
        raise LlmConfigError(f"LLM 配置必须是 JSON object: {path}")

    schema_version = raw.get("schema_version", 1)
    if schema_version not in (1, 2) or isinstance(schema_version, bool):
        raise LlmConfigError(f"LLM 配置 schema_version 不受支持: {path}")

    data = {}
    for field in _FIELD_TO_ENV:
        value = raw.get(field)
        if value is None:
            continue
        if not isinstance(value, str):
            raise LlmConfigError(f"LLM 配置字段 {field} 必须是字符串: {path}")
        if field == "model" and value != value.strip():
            raise LlmConfigError(f"LLM 配置字段 model 不允许两端空白: {path}")
        if value.strip():
            data[field] = value.strip()
    if schema_version == 1:
        data.setdefault("protocol", OPENAI_CHAT_COMPLETIONS_PROTOCOL)
        data["response_model_aliases"] = ()
    else:
        data["response_model_aliases"] = normalize_response_model_aliases(
            raw.get("response_model_aliases", ()),
            configured_model=data.get("model", ""),
        ) if data.get("model") else ()
    return data


def _status(path: Path, source: str) -> LlmConfigStatus:
    missing = tuple(key for key in REQUIRED_LLM_ENV if not _env_has(key))
    model = os.getenv("NUMPILOT_LLM_MODEL")
    protocol = os.getenv(
        "NUMPILOT_LLM_PROTOCOL", OPENAI_CHAT_COMPLETIONS_PROTOCOL
    ).strip()
    aliases: Tuple[str, ...] = ()
    if model:
        validate_model_id(model)
        validate_protocol(protocol)
        aliases = response_model_aliases_from_environment(
            configured_model=model
        )
    provider = os.getenv("NUMPILOT_LLM_PROVIDER", DEFAULT_PROVIDER_LABEL)
    validate_provider_label(provider)
    if not missing:
        _validate_values(
            os.environ["NUMPILOT_LLM_BASE_URL"],
            os.environ["NUMPILOT_LLM_AUTH_TOKEN"],
            model or "",
            provider,
            protocol,
            aliases,
        )
        try:
            max_tokens = int(os.getenv("NUMPILOT_LLM_MAX_TOKENS", "8192"))
        except ValueError as exc:
            raise LlmConfigError("LLM max_tokens 必须是数字") from exc
        if not 1 <= max_tokens <= 1_000_000:
            raise LlmConfigError("LLM max_tokens 必须在 1～1000000 之间")
        validate_llm_timeout(os.getenv("NUMPILOT_LLM_TIMEOUT", "120"))
        if os.getenv("NUMPILOT_LLM_INSECURE", "0") not in {"0", "1"}:
            raise LlmConfigError("NUMPILOT_LLM_INSECURE 只能是 0 或 1")
    return LlmConfigStatus(
        path=path,
        configured=not missing,
        missing=missing,
        source=source,
        model=model,
        provider=provider,
        token_present=_env_has("NUMPILOT_LLM_AUTH_TOKEN"),
        protocol=protocol,
        response_model_aliases=aliases,
    )


def load_llm_config(path: Optional[str] = None) -> LlmConfigStatus:
    """Load missing values from the private JSON file into ``os.environ``.

    Existing environment values win field by field.  A complete environment
    bypasses the file entirely, which keeps CI independent of a stale local
    configuration.
    """
    target = llm_config_path(path)
    present_before = {key for key in _FIELD_TO_ENV.values() if _env_has(key)}
    if all(_env_has(key) for key in REQUIRED_LLM_ENV):
        return _status(target, "environment")

    saved = _read_saved(target)
    loaded = False
    for field, env_name in _FIELD_TO_ENV.items():
        value = saved.get(field)
        if value and not _env_has(env_name):
            os.environ[env_name] = value
            loaded = True
    aliases = saved.get("response_model_aliases")
    model_was_explicit = "NUMPILOT_LLM_MODEL" in present_before
    if aliases and not _env_has(_ALIASES_ENV) and not model_was_explicit:
        os.environ[_ALIASES_ENV] = json.dumps(
            list(aliases), ensure_ascii=True, separators=(",", ":")
        )
        loaded = True
    if loaded and present_before:
        source = "environment+user-config"
    elif loaded:
        source = "user-config"
    elif present_before:
        source = "environment(partial)"
    else:
        source = "unconfigured"
    return _status(target, source)


def _validate_values(
    base_url: str,
    auth_token: str,
    model: str,
    provider: str,
    protocol: str,
    response_model_aliases: object,
) -> Tuple[str, str, str, Tuple[str, ...]]:
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
        hostname = parsed.hostname
    except ValueError as exc:
        raise LlmConfigError("BASE_URL 必须是完整的 http(s) URL") from exc
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or not hostname
        or (port is not None and not 1 <= port <= 65535)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or any(ord(character) <= 32 or ord(character) == 127 for character in base_url)
    ):
        raise LlmConfigError("BASE_URL 必须是完整的 http(s) URL")
    if (
        not auth_token
        or len(auth_token) > 16 * 1024
        or any(ord(character) <= 32 or ord(character) == 127 for character in auth_token)
    ):
        raise LlmConfigError("AUTH_TOKEN 不能为空或包含空白/控制字符")
    model = validate_model_id(model)
    provider = validate_provider_label(provider)
    protocol = validate_protocol(protocol)
    aliases = normalize_response_model_aliases(
        response_model_aliases, configured_model=model
    )
    return model, provider, protocol, aliases


def init_llm_config_template(path: Optional[str] = None) -> Tuple[Path, bool]:
    """Atomically create a private, manually editable config template.

    Return ``(path, created)``.  An existing file, directory, or symlink is
    never changed, so repeated setup commands cannot overwrite credentials.
    """
    target = llm_config_path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name == "posix":
            os.chmod(target.parent, 0o700)
    except OSError as exc:
        raise LlmConfigError(
            f"无法创建 LLM 配置目录 {target.parent}: {type(exc).__name__}"
        ) from exc

    if os.path.lexists(target):
        return target, False

    fd = -1
    temp_name = ""
    try:
        fd, temp_name = tempfile.mkstemp(
            prefix=".llm-template-", suffix=".tmp", dir=str(target.parent)
        )
        if os.name == "posix":
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            json.dump(_LLM_CONFIG_TEMPLATE, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        # A hard-link publishes the complete file atomically and fails if a
        # concurrent process created the destination first.
        try:
            os.link(temp_name, target)
        except FileExistsError:
            return target, False
    except OSError as exc:
        raise LlmConfigError(
            f"无法生成 LLM 配置模板 {target}: {type(exc).__name__}"
        ) from exc
    finally:
        if fd >= 0:
            os.close(fd)
        if temp_name:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
    return target, True


def save_llm_config(
    base_url: str,
    auth_token: str,
    model: str,
    provider: str = DEFAULT_PROVIDER_LABEL,
    path: Optional[str] = None,
    *,
    protocol: str = OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    response_model_aliases: object = (),
) -> Path:
    """Atomically save a validated config with a private directory/file mode."""
    base_url = base_url.strip().rstrip("/")
    auth_token = auth_token.strip()
    model = model.strip()
    provider = provider.strip() or DEFAULT_PROVIDER_LABEL
    model, provider, protocol, aliases = _validate_values(
        base_url,
        auth_token,
        model,
        provider,
        protocol,
        response_model_aliases,
    )

    target = llm_config_path(path)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        os.chmod(target.parent, 0o700)
    payload = {
        "schema_version": 2,
        "base_url": base_url,
        "auth_token": auth_token,
        "model": model,
        "provider": provider,
        "protocol": protocol,
        "response_model_aliases": list(aliases),
    }

    fd = -1
    temp_name = ""
    try:
        fd, temp_name = tempfile.mkstemp(prefix=".llm-", suffix=".tmp", dir=str(target.parent))
        if os.name == "posix":
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
        temp_name = ""
        if os.name == "posix":
            os.chmod(target, 0o600)
    except OSError as exc:
        raise LlmConfigError(f"无法保存 LLM 配置 {target}: {type(exc).__name__}") from exc
    finally:
        if fd >= 0:
            os.close(fd)
        if temp_name:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
    return target


def configure_llm_interactive(
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    protocol: Optional[str] = None,
    response_model_aliases: Optional[Tuple[str, ...]] = None,
    auth_token: Optional[str] = None,
    path: Optional[str] = None,
    input_fn: Callable[[str], str] = input,
    secret_fn: Callable[[str], str] = getpass.getpass,
) -> LlmConfigStatus:
    """Prompt inside NumPilot, hiding the token and preserving it on updates."""
    target = llm_config_path(path)
    saved = _read_saved(target)

    old_base = saved.get("base_url") or os.getenv("NUMPILOT_LLM_BASE_URL", "")
    old_model = saved.get("model") or os.getenv("NUMPILOT_LLM_MODEL", "")
    old_provider = saved.get("provider") or os.getenv(
        "NUMPILOT_LLM_PROVIDER", DEFAULT_PROVIDER_LABEL
    )
    old_protocol = saved.get("protocol") or os.getenv(
        "NUMPILOT_LLM_PROTOCOL", OPENAI_CHAT_COMPLETIONS_PROTOCOL
    )
    old_aliases = saved.get("response_model_aliases") or ()

    if base_url is None:
        suffix = f" [{old_base}]" if old_base else ""
        base_url = input_fn(f"LLM Base URL{suffix}: ").strip() or old_base
    if model is None:
        suffix = f" [{old_model}]" if old_model else ""
        model = input_fn(f"LLM Model{suffix}: ").strip() or old_model
    provider = (provider or old_provider or DEFAULT_PROVIDER_LABEL).strip()
    protocol = (protocol or old_protocol or OPENAI_CHAT_COMPLETIONS_PROTOCOL).strip()
    aliases = (
        old_aliases
        if response_model_aliases is None and (model or "").strip() == old_model
        else () if response_model_aliases is None
        else response_model_aliases
    )

    if auth_token is None:
        keep_hint = "（留空则保留现有值）" if saved.get("auth_token") else ""
        auth_token = secret_fn(f"LLM Auth Token{keep_hint}: ").strip()
        if not auth_token:
            auth_token = saved.get("auth_token", "")

    save_llm_config(
        base_url or "",
        auth_token,
        model or "",
        provider,
        str(target),
        protocol=protocol,
        response_model_aliases=aliases,
    )
    values = {
        "NUMPILOT_LLM_BASE_URL": (base_url or "").strip().rstrip("/"),
        "NUMPILOT_LLM_AUTH_TOKEN": auth_token,
        "NUMPILOT_LLM_MODEL": (model or "").strip(),
        "NUMPILOT_LLM_PROVIDER": provider,
        "NUMPILOT_LLM_PROTOCOL": protocol,
        _ALIASES_ENV: json.dumps(
            list(aliases), ensure_ascii=True, separators=(",", ":")
        ),
    }
    os.environ.update(values)
    return _status(target, "user-config")
