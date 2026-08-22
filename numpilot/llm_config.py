"""Open-source friendly LLM configuration managed by NumPilot itself.

The repository never contains a real authentication token.  Interactive users
configure the provider through ``python -m numpilot configure-llm``; NumPilot
then stores a small JSON document in the user's config directory with mode
``0600``.  CI can continue to inject the existing ``NUMPILOT_LLM_*``
environment variables, which always take precedence over saved values.
"""
from dataclasses import dataclass
import getpass
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Dict, Optional, Tuple
from urllib.parse import urlsplit


class LlmConfigError(RuntimeError):
    """A local LLM configuration is missing, malformed, or insecure."""


_FIELD_TO_ENV = {
    "base_url": "NUMPILOT_LLM_BASE_URL",
    "auth_token": "NUMPILOT_LLM_AUTH_TOKEN",
    "model": "NUMPILOT_LLM_MODEL",
    "provider": "NUMPILOT_LLM_PROVIDER",
}
REQUIRED_LLM_ENV = (
    "NUMPILOT_LLM_BASE_URL",
    "NUMPILOT_LLM_AUTH_TOKEN",
    "NUMPILOT_LLM_MODEL",
)

_LLM_CONFIG_TEMPLATE = {
    "schema_version": 1,
    "base_url": "",
    "auth_token": "",
    "model": "",
    "provider": "glm",
}


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

    def summary(self) -> str:
        """Return a display-safe status string that never contains the token."""
        if not self.configured:
            return "LLM 未配置，缺少: " + ", ".join(self.missing)
        provider = self.provider or "未标注"
        return (
            f"LLM 已配置(source={self.source}, provider={provider}, "
            f"model={self.model}, token=已设置)"
        )


def llm_config_path(path: Optional[str] = None) -> Path:
    """Resolve the private per-user JSON path, with test/automation overrides."""
    if path:
        return Path(path).expanduser()
    explicit = os.getenv("NUMPILOT_LLM_CONFIG_FILE")
    if explicit:
        return Path(explicit).expanduser()
    config_home = os.getenv("NUMPILOT_CONFIG_HOME")
    if config_home:
        return Path(config_home).expanduser() / "llm.json"
    xdg_home = os.getenv("XDG_CONFIG_HOME")
    base = Path(xdg_home).expanduser() if xdg_home else Path.home() / ".config"
    return base / "numpilot" / "llm.json"


def _read_saved(path: Path) -> Dict[str, str]:
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

    data = {}
    for field in _FIELD_TO_ENV:
        value = raw.get(field)
        if value is None:
            continue
        if not isinstance(value, str):
            raise LlmConfigError(f"LLM 配置字段 {field} 必须是字符串: {path}")
        if value.strip():
            data[field] = value.strip()
    return data


def _status(path: Path, source: str) -> LlmConfigStatus:
    missing = tuple(key for key in REQUIRED_LLM_ENV if not _env_has(key))
    return LlmConfigStatus(
        path=path,
        configured=not missing,
        missing=missing,
        source=source,
        model=os.getenv("NUMPILOT_LLM_MODEL"),
        provider=os.getenv("NUMPILOT_LLM_PROVIDER"),
        token_present=_env_has("NUMPILOT_LLM_AUTH_TOKEN"),
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
    if loaded and present_before:
        source = "environment+user-config"
    elif loaded:
        source = "user-config"
    elif present_before:
        source = "environment(partial)"
    else:
        source = "unconfigured"
    return _status(target, source)


def _validate_values(base_url: str, auth_token: str, model: str) -> None:
    parsed = urlsplit(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise LlmConfigError("BASE_URL 必须是完整的 http(s) URL")
    if not auth_token or "\n" in auth_token or "\r" in auth_token:
        raise LlmConfigError("AUTH_TOKEN 不能为空或包含换行")
    if not model or "\n" in model or "\r" in model:
        raise LlmConfigError("MODEL 不能为空或包含换行")


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
    provider: str = "glm",
    path: Optional[str] = None,
) -> Path:
    """Atomically save a validated config with a private directory/file mode."""
    base_url = base_url.strip().rstrip("/")
    auth_token = auth_token.strip()
    model = model.strip()
    provider = provider.strip() or "glm"
    _validate_values(base_url, auth_token, model)

    target = llm_config_path(path)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        os.chmod(target.parent, 0o700)
    payload = {
        "schema_version": 1,
        "base_url": base_url,
        "auth_token": auth_token,
        "model": model,
        "provider": provider,
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
    old_provider = saved.get("provider") or os.getenv("NUMPILOT_LLM_PROVIDER", "glm")

    if base_url is None:
        suffix = f" [{old_base}]" if old_base else ""
        base_url = input_fn(f"LLM Base URL{suffix}: ").strip() or old_base
    if model is None:
        suffix = f" [{old_model}]" if old_model else ""
        model = input_fn(f"LLM Model{suffix}: ").strip() or old_model
    provider = (provider or old_provider or "glm").strip()

    if auth_token is None:
        keep_hint = "（留空则保留现有值）" if saved.get("auth_token") else ""
        auth_token = secret_fn(f"LLM Auth Token{keep_hint}: ").strip()
        if not auth_token:
            auth_token = saved.get("auth_token", "")

    save_llm_config(base_url or "", auth_token, model or "", provider, str(target))
    values = {
        "NUMPILOT_LLM_BASE_URL": (base_url or "").strip().rstrip("/"),
        "NUMPILOT_LLM_AUTH_TOKEN": auth_token,
        "NUMPILOT_LLM_MODEL": (model or "").strip(),
        "NUMPILOT_LLM_PROVIDER": provider,
    }
    os.environ.update(values)
    return _status(target, "user-config")
