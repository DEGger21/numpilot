# This file is part of the NumPilot project at Harbin Institute of Technology (HIT).
#
# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# SPDX-License-Identifier: Apache-2.0
#
# Authors (accounts):
# - Su Tonghua <@sutonghua>
# - Li Xing <@li-xingHIT>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Safe, framework-owned connectivity check for the configured LLM.

The check deliberately sends one tiny, deterministic prompt and validates the
same provider/model/receipt contract used by migration and operator-generation
workflows.  It never returns or renders the response body, endpoint, token, or
proxy URL.  A successful result therefore means that a real, attributable
remote response was received; it is not a model-quality evaluation.
"""

from dataclasses import dataclass
import sys
from typing import Callable, Dict, Mapping, Optional, TextIO, Tuple

from numpilot.cli_ui import flow_print
from numpilot.engines.llm import (
    LlmEngine,
    LlmHttpError,
    LlmResponseError,
    LlmTransportError,
    llm_proxy_route_status,
)
from numpilot.llm_config import LlmConfigError, load_llm_config
from numpilot.model_contract import (
    ModelContractError,
    require_model_status,
    validate_configured_model_receipt,
)


LLM_CHECK_MARKER = "NUMPILOT_LLM_READY"
LLM_CHECK_SYSTEM_MESSAGE = (
    "You are a connectivity checker. Return one short acknowledgement only."
)
LLM_CHECK_USER_MESSAGE = "Reply with exactly the ASCII token NUMPILOT_LLM_READY and nothing else. Do not use XML tags, markup, or code fences."
# Some reasoning-capable gateways consume a small completion budget internally
# and occasionally return an empty visible acknowledgement at 64 tokens.  The
# request itself remains a one-line connectivity probe; 256 keeps that single
# call reliable without turning it into a model-quality workload.
LLM_CHECK_MAX_TOKENS = 256


_ERROR_MESSAGES = {
    "config_invalid": "本机 LLM 配置无效，请检查 llm.json。",
    "not_configured": "尚未配置可用的 LLM，请先填写 llm.json。",
    "proxy_invalid": "代理策略配置无效，未发起模型请求。",
    "proxy_blocked": "代理门禁未通过，未发起模型请求。",
    "http_error": "模型服务返回了非成功 HTTP 状态。",
    "transport_error": "未能连接模型服务或请求超时。",
    "response_invalid": "收到响应，但无法通过模型身份和回执校验。",
    "unexpected_error": "检查过程中出现未知本地错误。",
}


class LlmConnectivityCheckError(RuntimeError):
    """A bounded error that is safe to render in an interactive terminal."""

    def __init__(
        self,
        category: str,
        *,
        remote_attempted: bool = False,
        actual_remote_call: bool = False,
        http_status: Optional[int] = None,
    ) -> None:
        if category not in _ERROR_MESSAGES:
            category = "unexpected_error"
        safe_status = (
            http_status
            if (
                not isinstance(http_status, bool)
                and isinstance(http_status, int)
                and 100 <= http_status <= 599
            )
            else None
        )
        self.category = category
        self.remote_attempted = bool(remote_attempted)
        self.actual_remote_call = bool(actual_remote_call)
        self.http_status = safe_status
        super().__init__(_ERROR_MESSAGES[category])

    def to_safe_dict(self) -> Dict[str, object]:
        """Return only stable, non-secret failure facts."""

        return {
            "schema_version": 1,
            "check": "llm_connectivity",
            "status": "fail",
            "category": self.category,
            "message": _ERROR_MESSAGES[self.category],
            "remote_attempted": self.remote_attempted,
            "actual_remote_call": self.actual_remote_call,
            "http_status": self.http_status,
        }


@dataclass(frozen=True)
class LlmConnectivityResult:
    """Public proof that the configured model returned a valid acknowledgement."""

    config_source: str
    provider: str
    protocol: str
    configured_model: str
    response_model: Optional[str]
    http_status: int
    tls_verification: str
    total_tokens: Optional[int]
    content_sha256: str
    proxy_policy: str
    proxy_route: str
    proxy_matched: bool
    proxy_bypassed: bool

    def to_safe_dict(self) -> Dict[str, object]:
        """Serialize without response content, endpoint, credentials or proxy URL."""

        return {
            "schema_version": 1,
            "check": "llm_connectivity",
            "status": "pass",
            "actual_remote_call": True,
            "response_body_disclosed": False,
            "answer_marker": LLM_CHECK_MARKER,
            "answer_marker_present": True,
            "config_source": self.config_source,
            "provider": self.provider,
            "protocol": self.protocol,
            "configured_model": self.configured_model,
            "response_model": self.response_model,
            "http_status": self.http_status,
            "tls_verification": self.tls_verification,
            "total_tokens": self.total_tokens,
            "content_sha256": self.content_sha256,
            "proxy": {
                "policy": self.proxy_policy,
                "route": self.proxy_route,
                "matched": self.proxy_matched,
                "bypassed": self.proxy_bypassed,
            },
        }


def _safe_http_status(receipt: object) -> Optional[int]:
    if not isinstance(receipt, Mapping):
        return None
    value = receipt.get("http_status")
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 100
        or value > 599
    ):
        return None
    return value


def _safe_total_tokens(usage: object) -> Optional[int]:
    if not isinstance(usage, Mapping):
        return None
    value = usage.get("total_tokens")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _required_proxy_is_blocked(proxy: Mapping[str, object]) -> bool:
    return proxy.get("policy") == "required" and proxy.get("route") != "enforced"


def run_llm_connectivity_check(
    config_path: Optional[str] = None,
    *,
    chat: Optional[
        Callable[..., Tuple[str, Dict[str, object]]]
    ] = None,
) -> LlmConnectivityResult:
    """Make one real bounded call and validate its configured-model evidence.

    ``chat`` exists solely as a test seam.  Production callers should omit it,
    which routes through :meth:`LlmEngine._chat_with_receipt` and the same
    transport used by migration.  Environment proxy variables remain optional;
    if ``NUMPILOT_LLM_REQUIRE_PROXY=1`` is selected, the existing fail-closed
    proxy policy must be satisfied before any remote request is attempted.
    """

    try:
        status = load_llm_config(config_path)
    except LlmConfigError as exc:
        raise LlmConnectivityCheckError("config_invalid") from exc
    except Exception as exc:
        raise LlmConnectivityCheckError("unexpected_error") from exc
    if not status.configured:
        raise LlmConnectivityCheckError("not_configured")

    try:
        identity = require_model_status(status)
    except ModelContractError as exc:
        raise LlmConnectivityCheckError("config_invalid") from exc
    except Exception as exc:
        raise LlmConnectivityCheckError("unexpected_error") from exc

    try:
        proxy = llm_proxy_route_status()
    except LlmConfigError as exc:
        raise LlmConnectivityCheckError("proxy_invalid") from exc
    except Exception as exc:
        raise LlmConnectivityCheckError("unexpected_error") from exc
    if _required_proxy_is_blocked(proxy):
        raise LlmConnectivityCheckError("proxy_blocked")

    transport = chat or LlmEngine._chat_with_receipt
    try:
        content, receipt = transport(
            LLM_CHECK_SYSTEM_MESSAGE,
            LLM_CHECK_USER_MESSAGE,
            max_tokens=LLM_CHECK_MAX_TOKENS,
        )
    except LlmHttpError as exc:
        raise LlmConnectivityCheckError(
            "http_error",
            remote_attempted=True,
            http_status=exc.status_code,
        ) from exc
    except LlmTransportError as exc:
        raise LlmConnectivityCheckError(
            "transport_error", remote_attempted=True
        ) from exc
    except LlmResponseError as exc:
        http_status = _safe_http_status(exc.receipt)
        raise LlmConnectivityCheckError(
            "response_invalid",
            remote_attempted=True,
            actual_remote_call=(
                http_status is not None and 200 <= http_status < 300
            ),
            http_status=http_status,
        ) from exc
    except LlmConfigError as exc:
        # The transport re-validates the hydrated configuration immediately
        # before network I/O, so a concurrent mutation remains a local failure.
        raise LlmConnectivityCheckError("config_invalid") from exc
    except Exception as exc:
        # Never interpolate arbitrary exceptions: HTTP libraries can include a
        # full endpoint, proxy URL, or credential-bearing diagnostic string.
        raise LlmConnectivityCheckError(
            "unexpected_error", remote_attempted=True
        ) from exc

    try:
        validated = validate_configured_model_receipt(
            receipt,
            content,
            expected_model=identity.model,
            expected_provider=identity.provider,
            expected_protocol=identity.protocol,
            response_model_aliases=identity.response_model_aliases,
        )
    except ModelContractError as exc:
        raise LlmConnectivityCheckError(
            "response_invalid",
            remote_attempted=True,
            actual_remote_call=True,
            http_status=_safe_http_status(receipt),
        ) from exc

    if not isinstance(content, str) or LLM_CHECK_MARKER not in content:
        raise LlmConnectivityCheckError(
            "response_invalid",
            remote_attempted=True,
            actual_remote_call=True,
            http_status=_safe_http_status(validated),
        )

    return LlmConnectivityResult(
        config_source=status.source,
        provider=identity.provider,
        protocol=identity.protocol,
        configured_model=identity.model,
        response_model=validated.get("response_model"),
        http_status=validated["http_status"],
        tls_verification=validated["tls_verification"],
        total_tokens=_safe_total_tokens(validated.get("usage")),
        content_sha256=validated["content_sha256"],
        proxy_policy=str(proxy["policy"]),
        proxy_route=str(proxy["route"]),
        proxy_matched=bool(proxy["matched"]),
        proxy_bypassed=bool(proxy["bypassed"]),
    )


def render_llm_connectivity_result(result: LlmConnectivityResult) -> str:
    """Render a compact terminal summary without exposing the response body."""

    model = result.response_model or result.configured_model
    tokens = result.total_tokens if result.total_tokens is not None else "unknown"
    return "\n".join((
        "[LLM 检查] 成功：已配置模型返回了可验证应答，可以开始迁移。",
        (
            "[LLM 检查] actual_remote_call=yes · http_status=%s · "
            "model=%s · total_tokens=%s"
        ) % (result.http_status, model, tokens),
        (
            "[LLM 检查] 应答正文不显示 · marker=verified · "
            "content_sha256=%s"
        ) % result.content_sha256,
        (
            "[LLM 代理] policy=%s · route=%s · matched=%s · bypass=%s"
        ) % (
            result.proxy_policy,
            result.proxy_route,
            "yes" if result.proxy_matched else "no",
            "yes" if result.proxy_bypassed else "no",
        ),
    ))


def render_llm_connectivity_error(error: LlmConnectivityCheckError) -> str:
    """Render stable failure facts without forwarding exception details."""

    status = "unknown" if error.http_status is None else str(error.http_status)
    return "\n".join((
        "[LLM 检查] 失败：%s" % _ERROR_MESSAGES[error.category],
        (
            "[LLM 检查] remote_attempted=%s · actual_remote_call=%s · "
            "http_status=%s"
        ) % (
            "yes" if error.remote_attempted else "no",
            "yes" if error.actual_remote_call else "no",
            status,
        ),
    ))


def run_llm_check_command(
    config_path: Optional[str] = None,
    *,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
    chat: Optional[
        Callable[..., Tuple[str, Dict[str, object]]]
    ] = None,
) -> int:
    """Execute the terminal ``check llm`` action with stable exit semantics.

    Local configuration/proxy failures return ``1``.  A request that is sent
    but cannot produce a verified model acknowledgement returns ``4``, matching
    the existing ``configure-llm --probe`` convention.  The injected ``chat``
    hook is for tests only.
    """

    output = stdout if stdout is not None else sys.stdout
    errors = stderr if stderr is not None else sys.stderr
    flow_print(
        "[LLM 检查] 正在向当前配置的模型发送轻量验证消息……",
        stream=output,
        tone="progress",
    )
    try:
        result = run_llm_connectivity_check(config_path, chat=chat)
    except LlmConnectivityCheckError as exc:
        flow_print(
            render_llm_connectivity_error(exc),
            stream=errors,
            tone="error",
        )
        return 4 if exc.remote_attempted else 1
    flow_print(
        render_llm_connectivity_result(result),
        stream=output,
        tone="success",
    )
    return 0
