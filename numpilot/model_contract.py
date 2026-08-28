"""Provider-neutral identity and receipt contract for configured models.

This module deliberately has no CANNBot or operator-generation imports so the
ordinary direct lane can validate model evidence without loading gap tooling.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from typing import Dict, Optional, Tuple

from numpilot.llm_config import (
    LlmConfigError,
    OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    model_identity_sha256,
    normalize_response_model_aliases,
    validate_model_id,
    validate_protocol,
    validate_provider_label,
)


class ModelContractError(RuntimeError):
    """Configuration or remote evidence violates the declared model contract."""


REMOTE_RECEIPT_V1_FIELDS = frozenset({
    "schema_version",
    "utc_start",
    "utc_end",
    "http_status",
    "requested_model",
    "response_model",
    "response_id",
    "response_created",
    "request_body_sha256",
    "response_body_sha256",
    "content_sha256",
    "request_id",
    "usage",
    "tls_verification",
})
REMOTE_RECEIPT_V2_FIELDS = REMOTE_RECEIPT_V1_FIELDS | frozenset({
    "protocol",
    "provider",
    "configured_model",
    "model_identity_sha256",
    "response_model_aliases",
    "response_model_identity",
})

_SAFE_METADATA = re.compile(r"^[A-Za-z0-9._:?-]{1,512}$")
_LEGACY_GLM_MODEL = re.compile(
    r"^glm-[A-Za-z0-9][A-Za-z0-9._:-]{0,507}$", re.IGNORECASE
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RESPONSE_NOT_PROVIDED = object()
_SENSITIVE_ENVIRONMENT_NAMES = (
    "NUMPILOT_LLM_AUTH_TOKEN",
    "NUMPILOT_LLM_BASE_URL",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)


@dataclass(frozen=True)
class ConfiguredModelIdentity:
    provider: str
    protocol: str
    model: str
    response_model_aliases: Tuple[str, ...]


def _raise_config(exc: Exception) -> None:
    raise ModelContractError(str(exc)) from exc


def require_model_configuration(
    provider: object,
    model: object,
    *,
    protocol: object = OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    response_model_aliases: object = (),
) -> ConfiguredModelIdentity:
    """Validate readiness without authorizing or rejecting a provider label."""

    try:
        exact_model = validate_model_id(model)
        audit_provider = validate_provider_label(provider)
        exact_protocol = validate_protocol(protocol)
        aliases = normalize_response_model_aliases(
            response_model_aliases, configured_model=exact_model
        )
    except LlmConfigError as exc:
        _raise_config(exc)
    return ConfiguredModelIdentity(
        provider=audit_provider,
        protocol=exact_protocol,
        model=exact_model,
        response_model_aliases=aliases,
    )


def require_model_status(status: object) -> ConfiguredModelIdentity:
    if status is None or not getattr(status, "configured", False):
        raise ModelContractError("configured model is unavailable")
    return require_model_configuration(
        getattr(status, "provider", None),
        getattr(status, "model", None),
        protocol=getattr(
            status, "protocol", OPENAI_CHAT_COMPLETIONS_PROTOCOL
        ),
        response_model_aliases=getattr(
            status, "response_model_aliases", ()
        ),
    )


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _parse_utc(value: object, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z") or len(value) > 64:
        raise ModelContractError(
            "remote model receipt field is invalid: %s" % name
        )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ModelContractError(
            "remote model receipt field is invalid: %s" % name
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ModelContractError(
            "remote model receipt field is invalid: %s" % name
        )
    return parsed


def validate_remote_receipt(
    receipt: object,
    response: object = _RESPONSE_NOT_PROVIDED,
) -> Dict[str, object]:
    """Validate schema, bounds, redaction and response-content binding."""

    if not isinstance(receipt, dict):
        raise ModelContractError("remote model call returned an incomplete receipt")
    schema = receipt.get("schema_version")
    expected_fields = (
        REMOTE_RECEIPT_V1_FIELDS if schema == 1
        else REMOTE_RECEIPT_V2_FIELDS if schema == 2
        else None
    )
    if expected_fields is None or isinstance(schema, bool):
        raise ModelContractError("remote model receipt schema is invalid")
    if frozenset(receipt) != expected_fields:
        raise ModelContractError("remote model call returned an incomplete receipt")

    status = receipt.get("http_status")
    if (
        isinstance(status, bool)
        or not isinstance(status, int)
        or status < 200
        or status >= 300
    ):
        raise ModelContractError("remote model receipt HTTP status is invalid")
    for name in ("request_body_sha256", "response_body_sha256"):
        if not _is_sha256(receipt.get(name)):
            raise ModelContractError(
                "remote model receipt hash is invalid: %s" % name
            )

    if response is _RESPONSE_NOT_PROVIDED:
        if receipt.get("content_sha256") is not None and not _is_sha256(
            receipt.get("content_sha256")
        ):
            raise ModelContractError("remote model receipt content hash is invalid")
    elif response is None:
        if receipt.get("content_sha256") is not None:
            raise ModelContractError(
                "failed remote model receipt claims response content"
            )
    elif (
        not isinstance(response, str)
        or not _is_sha256(receipt.get("content_sha256"))
        or receipt["content_sha256"]
        != hashlib.sha256(response.encode("utf-8")).hexdigest()
    ):
        raise ModelContractError("remote model receipt content hash mismatch")

    started = _parse_utc(receipt.get("utc_start"), "utc_start")
    ended = _parse_utc(receipt.get("utc_end"), "utc_end")
    if ended < started:
        raise ModelContractError("remote model receipt time range is invalid")

    for name in ("response_id", "request_id"):
        value = receipt.get(name)
        if value is not None and (
            not isinstance(value, str) or _SAFE_METADATA.fullmatch(value) is None
        ):
            raise ModelContractError(
                "remote model receipt field is invalid: %s" % name
            )

    if schema == 1:
        for name in ("requested_model", "response_model"):
            value = receipt.get(name)
            if name == "requested_model" and value is None:
                raise ModelContractError(
                    "remote model receipt field is invalid: %s" % name
                )
            if value is not None and (
                not isinstance(value, str)
                or _SAFE_METADATA.fullmatch(value) is None
            ):
                raise ModelContractError(
                    "remote model receipt field is invalid: %s" % name
                )
    else:
        try:
            configured = validate_model_id(
                receipt.get("configured_model"), field="configured_model"
            )
            requested = validate_model_id(
                receipt.get("requested_model"), field="requested_model"
            )
            actual_raw = receipt.get("response_model")
            actual = (
                None
                if actual_raw is None
                else validate_model_id(actual_raw, field="response_model")
            )
            protocol = validate_protocol(receipt.get("protocol"))
            provider = validate_provider_label(receipt.get("provider"))
            aliases = normalize_response_model_aliases(
                receipt.get("response_model_aliases"),
                configured_model=configured,
            )
        except LlmConfigError as exc:
            _raise_config(exc)
        response_identity = receipt.get("response_model_identity")
        expected_response_identity = (
            "missing" if actual is None
            else "exact" if actual == configured
            else "explicit_alias" if actual in aliases
            else "mismatch"
        )
        if response_identity != expected_response_identity:
            raise ModelContractError(
                "remote model receipt response identity status is invalid"
            )
        expected_identity_hash = model_identity_sha256(
            protocol=protocol,
            provider=provider,
            configured_model=configured,
            requested_model=requested,
            response_model=actual,
            response_model_aliases=aliases,
            response_model_identity=response_identity,
        )
        if receipt.get("model_identity_sha256") != expected_identity_hash:
            raise ModelContractError(
                "remote model receipt identity hash is invalid"
            )

    created = receipt.get("response_created")
    if created is not None and (
        isinstance(created, bool)
        or not isinstance(created, int)
        or created < 0
        or created > 2 ** 63 - 1
    ):
        raise ModelContractError(
            "remote model receipt created value is invalid"
        )
    usage = receipt.get("usage")
    if (
        not isinstance(usage, dict)
        or len(usage) > 32
        or any(
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", name) is None
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value > 2 ** 63 - 1
            for name, value in usage.items()
        )
    ):
        raise ModelContractError("remote model receipt usage is invalid")
    if receipt.get("tls_verification") not in {"enabled", "disabled"}:
        raise ModelContractError("remote model receipt TLS state is invalid")
    serialized = json.dumps(receipt, ensure_ascii=False, sort_keys=True)
    if len(serialized.encode("utf-8")) > 64 * 1024:
        raise ModelContractError("remote model receipt exceeds the size limit")
    for env_name in _SENSITIVE_ENVIRONMENT_NAMES:
        sensitive = os.getenv(env_name, "")
        if sensitive and sensitive in serialized:
            raise ModelContractError("remote model receipt contains sensitive data")
    return dict(receipt)


def validate_configured_model_receipt(
    receipt: object,
    response: object = _RESPONSE_NOT_PROVIDED,
    *,
    expected_model: object,
    expected_provider: object,
    expected_protocol: object = OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    response_model_aliases: object = (),
) -> Dict[str, object]:
    """Prove configured=requested and exact/explicit-alias response identity."""

    identity = require_model_configuration(
        expected_provider,
        expected_model,
        protocol=expected_protocol,
        response_model_aliases=response_model_aliases,
    )
    validated = validate_remote_receipt(receipt, response)
    schema = validated["schema_version"]
    requested = validated.get("requested_model")
    actual = validated.get("response_model")

    if schema == 1:
        if (
            _LEGACY_GLM_MODEL.fullmatch(identity.model) is None
            or identity.provider.lower() != "glm"
            or identity.protocol != OPENAI_CHAT_COMPLETIONS_PROTOCOL
            or identity.response_model_aliases
        ):
            raise ModelContractError(
                "schema 1 is read-only compatibility for an exact legacy GLM identity"
            )
        if requested != identity.model or actual != identity.model:
            raise ModelContractError(
                "legacy GLM receipt does not prove one exact configured identity"
            )
        return validated

    if validated.get("configured_model") != identity.model:
        raise ModelContractError(
            "remote receipt configured_model does not match local configuration"
        )
    if requested != identity.model:
        raise ModelContractError(
            "remote receipt requested_model does not match configured model"
        )
    if validated.get("provider") != identity.provider:
        raise ModelContractError("remote receipt provider audit label is inconsistent")
    if validated.get("protocol") != identity.protocol:
        raise ModelContractError("remote receipt protocol is inconsistent")
    if tuple(validated.get("response_model_aliases", ())) != (
        identity.response_model_aliases
    ):
        raise ModelContractError(
            "remote receipt response-model alias policy is inconsistent"
        )
    if actual is None and (
        response is None
        or (
            response is _RESPONSE_NOT_PROVIDED
            and validated.get("content_sha256") is None
        )
    ):
        return validated
    if not isinstance(actual, str):
        raise ModelContractError("remote receipt response_model is missing")
    if actual != identity.model and actual not in identity.response_model_aliases:
        raise ModelContractError(
            "remote receipt response_model is neither exact nor explicitly allowed"
        )
    expected_state = "exact" if actual == identity.model else "explicit_alias"
    if validated.get("response_model_identity") != expected_state:
        raise ModelContractError(
            "remote receipt response identity decision is inconsistent"
        )
    return validated


def validate_configured_model_attempt_receipt(
    receipt: object,
    response: object = _RESPONSE_NOT_PROVIDED,
    *,
    expected_model: object,
    expected_provider: object,
    expected_protocol: object = OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    response_model_aliases: object = (),
) -> Dict[str, object]:
    """Validate historical attempt evidence without authorizing its response.

    Schema 2 still binds configured/requested identity, provider, protocol,
    alias policy, content and the self-declared response decision.  A missing
    or mismatched response identity may be retained only as failed-attempt
    evidence; callers must apply :func:`validate_configured_model_receipt` to
    the final accepted attempt.
    """

    identity = require_model_configuration(
        expected_provider,
        expected_model,
        protocol=expected_protocol,
        response_model_aliases=response_model_aliases,
    )
    validated = validate_remote_receipt(receipt, response)
    if validated["schema_version"] == 1:
        return validate_configured_model_receipt(
            validated,
            response,
            expected_model=identity.model,
            expected_provider=identity.provider,
            expected_protocol=identity.protocol,
            response_model_aliases=identity.response_model_aliases,
        )
    if validated.get("configured_model") != identity.model:
        raise ModelContractError(
            "remote receipt configured_model does not match local configuration"
        )
    if validated.get("requested_model") != identity.model:
        raise ModelContractError(
            "remote receipt requested_model does not match configured model"
        )
    if validated.get("provider") != identity.provider:
        raise ModelContractError("remote receipt provider audit label is inconsistent")
    if validated.get("protocol") != identity.protocol:
        raise ModelContractError("remote receipt protocol is inconsistent")
    if tuple(validated.get("response_model_aliases", ())) != (
        identity.response_model_aliases
    ):
        raise ModelContractError(
            "remote receipt response-model alias policy is inconsistent"
        )
    return validated
