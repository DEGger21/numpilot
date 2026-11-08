"""Single, auditable boundary for configured model calls.

The gateway keeps model configuration and remote receipts behind one typed seam.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Callable, Dict, Mapping, Optional, Protocol, Tuple

from numpilot.llm_config import LlmConfigError
from numpilot.model_contract import (
    ConfiguredModelIdentity,
    ModelContractError,
    require_model_status,
    validate_remote_receipt,
)

_MAX_PROMPT_BYTES = 2 * 1024 * 1024
_MAX_ROLE_BYTES = 128
_MAX_TASK_BYTES = 128
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_TASKS = frozenset({
    "connectivity_check", "script_migration", "gap_generation",
    "gap_repair", "interactive_interpretation", "readiness_gate",
})
_THINKING_MODES = frozenset({"enabled", "disabled"})
_REASONING_EFFORTS = frozenset(
    {"max", "xhigh", "high", "medium", "low", "minimal", "none"}
)


class ModelTransport(Protocol):
    """The narrow callable implemented by production transport and tests."""

    def __call__(
        self, system: str, user: str, **options: object
    ) -> Tuple[str, Dict[str, object]]:
        ...


@dataclass(frozen=True)
class ModelRequest:
    """A bounded, task-labelled model request."""

    task: str
    system: str
    user: str
    role: str = "unknown"
    max_tokens: Optional[int] = None
    thinking_mode: Optional[str] = None
    reasoning_effort: Optional[str] = None
    stream: bool = False
    source_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        _validate_label(self.task, "task", _MAX_TASK_BYTES)
        if self.task not in _TASKS:
            raise ValueError("unsupported model task")
        _validate_label(self.role, "role", _MAX_ROLE_BYTES)
        _validate_prompt(self.system, "system prompt")
        _validate_prompt(self.user, "user prompt")
        if self.max_tokens is not None and (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or not 1 <= self.max_tokens <= 1_000_000
        ):
            raise ValueError("max_tokens must be an integer in 1..1000000")
        if self.thinking_mode not in (None, *_THINKING_MODES):
            raise ValueError("thinking_mode is invalid")
        if self.reasoning_effort not in (None, *_REASONING_EFFORTS):
            raise ValueError("reasoning_effort is invalid")
        if not isinstance(self.stream, bool):
            raise ValueError("stream must be a boolean")
        if self.source_sha256 is not None and (
            not isinstance(self.source_sha256, str)
            or _SHA256.fullmatch(self.source_sha256) is None
        ):
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class ModelResult:
    """Validated model content plus the existing redacted remote receipt.

    response_sha256 is retained as a compatibility name for the digest of
    returned content. The receipt HTTP entity hash is exposed separately.
    """

    content: str
    receipt: Mapping[str, object]
    identity: ConfiguredModelIdentity
    request_sha256: str
    response_sha256: str
    actual_remote_call: bool

    @property
    def content_sha256(self) -> str:
        """Return the digest of the model content."""

        return self.response_sha256

    @property
    def response_body_sha256(self) -> Optional[str]:
        """Return the digest of the redacted HTTP response entity."""

        value = self.receipt.get("response_body_sha256")
        return value if isinstance(value, str) else None


class ModelGatewayError(RuntimeError):
    """Safe, stable failure raised at the model boundary."""

    def __init__(
        self,
        category: str,
        message: str,
        *,
        status_code: Optional[int] = None,
        receipt: Optional[Mapping[str, object]] = None,
    ) -> None:
        if category not in {"config", "transport", "http", "response", "contract"}:
            category = "unknown"
        self.category = category
        self.status_code = status_code
        self.receipt = dict(receipt) if isinstance(receipt, Mapping) else None
        super().__init__("model gateway %s: %s" % (category, message))


def _validate_label(value: object, field: str, max_bytes: int) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("%s must be a non-empty string" % field)
    if len(value.encode("utf-8")) > max_bytes or _SAFE_LABEL.fullmatch(value) is None:
        raise ValueError("%s contains unsupported characters" % field)


def _validate_prompt(value: object, field: str) -> None:
    if not isinstance(value, str) or len(value.encode("utf-8")) > _MAX_PROMPT_BYTES:
        raise ValueError("%s exceeds the size limit" % field)


def _request_sha256(request: ModelRequest, identity: ConfiguredModelIdentity) -> str:
    payload = {
        "configured_model": identity.model,
        "protocol": identity.protocol,
        "provider": identity.provider,
        "response_model_aliases": list(identity.response_model_aliases),
        "role": request.role, "source_sha256": request.source_sha256,
        "stream": request.stream, "task": request.task,
        "thinking_mode": request.thinking_mode,
        "reasoning_effort": request.reasoning_effort,
        "max_tokens": request.max_tokens,
        "system": request.system, "user": request.user,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ModelGateway:
    """Own configuration loading, transport invocation and receipt validation."""

    def __init__(
        self,
        *,
        config_path: Optional[str] = None,
        configuration_loader: Optional[Callable[..., object]] = None,
        transport: Optional[ModelTransport] = None,
    ) -> None:
        self._config_path = config_path
        self._configuration_loader = configuration_loader
        self._transport = transport

    def _load_status(self) -> object:
        try:
            if self._configuration_loader is None:
                from numpilot.llm_config import load_llm_config

                loader = load_llm_config
            else:
                loader = self._configuration_loader
            if self._config_path is None:
                return loader()
            return loader(self._config_path)
        except LlmConfigError:
            raise ModelGatewayError(
                "config", "configuration is unavailable"
            ) from None
        except Exception:
            raise ModelGatewayError(
                "config", "configuration loader failed"
            ) from None

    def complete(self, request: ModelRequest) -> ModelResult:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a ModelRequest")
        status = self._load_status()
        if not getattr(status, "configured", False):
            raise ModelGatewayError("config", "configured model is unavailable")
        try:
            identity = require_model_status(status)
        except (ModelContractError, LlmConfigError):
            raise ModelGatewayError(
                "config", "configured model identity is invalid"
            ) from None

        digest = _request_sha256(request, identity)
        transport = self._transport
        actual_remote_call = transport is None
        if transport is None:
            try:
                from numpilot.engines.llm import LlmEngine
                transport = LlmEngine._chat_with_receipt
            except Exception:
                raise ModelGatewayError(
                    "transport", "model transport is unavailable"
                ) from None

        options: Dict[str, object] = {}
        if request.max_tokens is not None:
            options["max_tokens"] = request.max_tokens
        if request.thinking_mode is not None:
            options["thinking_mode"] = request.thinking_mode
        if request.reasoning_effort is not None:
            options["reasoning_effort"] = request.reasoning_effort
        if request.stream:
            options["stream"] = True

        try:
            content, raw_receipt = transport(
                request.system, request.user, **options
            )
        except Exception as exc:
            name = type(exc).__name__
            raw_error_receipt = getattr(exc, "receipt", None)
            if name == "LlmHttpError":
                status_code = getattr(exc, "status_code", None)
                if not isinstance(status_code, int) or isinstance(status_code, bool):
                    status_code = None
                raise ModelGatewayError(
                    "http", "remote model request failed",
                    status_code=status_code,
                ) from None
            if name == "LlmResponseError":
                safe_receipt = (
                    raw_error_receipt
                    if isinstance(raw_error_receipt, Mapping) else None
                )
                raise ModelGatewayError(
                    "response", "remote model response is invalid",
                    receipt=safe_receipt,
                ) from exc
            if name == "LlmConfigError":
                raise ModelGatewayError(
                    "config", "remote model configuration is invalid"
                ) from None
            if name == "LlmTransportError":
                raise ModelGatewayError(
                    "transport", "remote model transport failed"
                ) from None
            raise ModelGatewayError(
                "transport", "remote model transport failed"
            ) from None

        if not isinstance(content, str) or not isinstance(raw_receipt, dict):
            raise ModelGatewayError(
                "response", "remote model response has an invalid shape"
            )
        try:
            receipt = validate_remote_receipt(raw_receipt, content)
        except Exception as exc:
            raise ModelGatewayError(
                "contract", "remote model receipt violates the contract",
                receipt=raw_receipt,
            ) from exc
        return ModelResult(
            content=content,
            receipt=receipt,
            identity=identity,
            request_sha256=digest,
            response_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            actual_remote_call=actual_remote_call,
        )

    def as_chat(
        self, *, task: str = "script_migration", role: str = "unknown"
    ) -> ModelTransport:
        """Return the legacy two-string chat adapter."""

        def chat(system: str, user: str, **options: object):
            allowed = {
                "max_tokens", "thinking_mode", "reasoning_effort", "stream"
            }
            if set(options) - allowed:
                raise ValueError("unsupported model options")
            request = ModelRequest(
                task=task, role=role, system=system, user=user,
                max_tokens=options.get("max_tokens"),  # type: ignore[arg-type]
                thinking_mode=options.get("thinking_mode"),  # type: ignore[arg-type]
                reasoning_effort=options.get("reasoning_effort"),  # type: ignore[arg-type]
                stream=options.get("stream", False),  # type: ignore[arg-type]
            )
            result = self.complete(request)
            return result.content, dict(result.receipt)

        return chat
