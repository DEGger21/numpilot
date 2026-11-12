"""可插拔 LLM 迁移引擎(provider 无关, OpenAI 兼容协议)。

CLI 会优先读取以下环境变量，并可通过 ``numpilot configure-llm`` 自动加载
框架管理的私有用户配置:
  NUMPILOT_LLM_BASE_URL    OpenAI 兼容 endpoint, 例: https://host:port/v1(务必带 /v1)
  NUMPILOT_LLM_AUTH_TOKEN  鉴权 token(以 Authorization: Bearer 发送)
  NUMPILOT_LLM_MODEL       精确模型名, 例: org/model
  NUMPILOT_LLM_PROVIDER    (可选)仅作标签, 例: pangu | glm | openai | deepseek
  NUMPILOT_LLM_MAX_TOKENS  (可选)显式生成上限; 未设置时不发送该字段,
                           由所接入模型/网关使用自身默认上限
  NUMPILOT_LLM_INSECURE    (可选)"1"=显式跳过 TLS 证书校验(仅限受信内网), 默认 "0"
  NUMPILOT_LLM_TIMEOUT     (可选)单次请求超时秒, 默认 120
  NUMPILOT_LLM_REQUIRE_PROXY (可选)"1"=强制请求经标准 HTTP(S) 代理, 默认 "0"

``strict`` 参数仅为 API 兼容保留。所有 configured-model 路径现在均 fail
closed；配置缺失、调用失败或响应不合格时绝不回退成本地 AST 产物。
真实凭据不入库；框架管理的用户配置以 0600 权限保存在用户配置目录。
"""
import base64
import hashlib
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from numpilot.contracts import OffloadPlan
from numpilot.asnumpy_guide import render_asnumpy_guide
from numpilot.engines.base import MigrationEngine
from numpilot.llm_config import (
    DEFAULT_PROVIDER_LABEL,
    OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    LlmConfigError,
    load_llm_config,
    model_identity_sha256,
    response_model_aliases_from_environment,
    validate_llm_timeout,
    validate_model_id,
    validate_protocol,
    validate_provider_label,
)
from numpilot.model_contract import (
    require_model_configuration,
    validate_configured_model_receipt,
)


class LlmError(RuntimeError):
    """strict 模式下 LLM 未配置/无响应/返回非法迁移码时抛出, 由 CLI 捕获并终止。"""


class LlmHttpError(RuntimeError):
    """A redacted HTTP failure that never includes the URL or response body."""

    def __init__(self, status_code: int):
        if (
            isinstance(status_code, bool)
            or not isinstance(status_code, int)
            or status_code < 100
            or status_code > 599
        ):
            status_code = 0
        self.status_code = status_code
        super().__init__("remote LLM HTTP status %d" % status_code)


class LlmTransportError(RuntimeError):
    """A redacted transport failure that does not expose endpoint details."""

    def __init__(self, category: str):
        safe_category = re.sub(r"[^A-Za-z0-9_.-]", "?", category)[:128]
        super().__init__(
            "remote LLM transport failed: %s" % (safe_category or "unknown")
        )


class LlmResponseError(RuntimeError):
    """A safe parse failure carrying the receipt for a successful HTTP response."""

    def __init__(self, reason: str, receipt: Dict[str, object]):
        self.receipt = dict(receipt)
        super().__init__("remote LLM response is invalid: %s" % reason)


_REQUIRED_ENV = (
    "NUMPILOT_LLM_BASE_URL",
    "NUMPILOT_LLM_AUTH_TOKEN",
    "NUMPILOT_LLM_MODEL",
)

_REQUEST_ID_HEADERS = (
    "x-request-id",
    "request-id",
    "x-openai-request-id",
    "x-zhipu-request-id",
    "x-trace-id",
)
# Optional transport safety ceiling.  ``None`` means no NumPilot-imposed
# response-body limit: operator reasoning/output can be long and must be
# bounded by the configured model/gateway, not by a small framework
# constant.  A deployment may set a positive integer to enforce a ceiling.
_MAX_RESPONSE_BODY_BYTES: Optional[int] = None
# Keep individual non-stream reads small enough that the monotonic deadline
# can be re-checked regularly.  ``HTTPResponse.read(amt)`` may otherwise wait
# for all requested bytes while a peer slowly dribbles a response body.
_MAX_RESPONSE_READ_CHUNK_BYTES = 64 * 1024
# Transport sanity bound, not a model output limit.  One million SSE
# events is far beyond any real operator response and still rejects a
# malicious infinite stream.
_MAX_STREAM_EVENTS = 1_000_000
_MIN_PER_CALL_MAX_TOKENS = 1
_MAX_PER_CALL_MAX_TOKENS = 1_000_000
_REQUIRE_PROXY_ENV = "NUMPILOT_LLM_REQUIRE_PROXY"
_SENSITIVE_METADATA_ENV = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)


def _proxy_is_required() -> bool:
    """Return the explicit fail-closed proxy policy for remote LLM calls."""

    value = os.getenv(_REQUIRE_PROXY_ENV, "0")
    if value not in ("0", "1"):
        raise LlmConfigError(
            "%s must be exactly 0 or 1" % _REQUIRE_PROXY_ENV
        )
    return value == "1"


def _environment_proxy_snapshot() -> Dict[str, str]:
    """Capture standard proxy variables once for one local decision."""

    return dict(urllib.request.getproxies_environment())


def _proxy_bypassed(host: str, proxies: Dict[str, str]) -> bool:
    """Evaluate NO_PROXY from the same environment snapshot as the proxy."""

    return bool(urllib.request.proxy_bypass_environment(host, proxies))


class _PinnedProxyHandler(urllib.request.ProxyHandler):
    """ProxyHandler that cannot re-read a changed bypass environment.

    The required-proxy path has already evaluated ``NO_PROXY`` against its
    captured mapping.  This mirrors the standard library handler except for
    deliberately omitting its second, process-global bypass lookup.
    """

    def proxy_open(self, request, proxy, _proxy_type):
        try:
            original_type = request.type
            parsed_type, user, password, hostport = urllib.request._parse_proxy(
                proxy
            )
            if parsed_type is None:
                parsed_type = original_type
            if user and password:
                user_pass = "%s:%s" % (
                    urllib.parse.unquote(user),
                    urllib.parse.unquote(password),
                )
                credentials = base64.b64encode(user_pass.encode()).decode(
                    "ascii"
                )
                request.add_header(
                    "Proxy-authorization", "Basic " + credentials
                )
            hostport = urllib.parse.unquote(hostport)
            if not hostport:
                raise ValueError("empty proxy host")
            request.set_proxy(hostport, parsed_type)
        except (TypeError, UnicodeError, ValueError):
            raise LlmConfigError(
                "matching HTTP(S) proxy configuration is invalid"
            ) from None
        if original_type == parsed_type or original_type == "https":
            return None
        return self.parent.open(request, timeout=request.timeout)


def llm_proxy_route_status(base_url: Optional[str] = None) -> Dict[str, object]:
    """Return a redacted, network-free summary of the current proxy route.

    No endpoint hostname, proxy URL, credentials, or ``NO_PROXY`` values are
    returned.  The result is therefore safe for CLI diagnostics and receipts.
    """

    required = _proxy_is_required()
    if base_url is None:
        endpoint = os.getenv("NUMPILOT_LLM_BASE_URL", "")
    else:
        try:
            endpoint = os.fspath(base_url)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise LlmConfigError("LLM endpoint is invalid") from exc
        if not isinstance(endpoint, str):
            raise LlmConfigError("LLM endpoint is invalid")
    try:
        parsed = urllib.parse.urlsplit(endpoint)
    except (TypeError, UnicodeError, ValueError):
        parsed = urllib.parse.SplitResult("", "", "", "", "")
    scheme = parsed.scheme.lower()
    supported_scheme = scheme in {"http", "https"}
    proxies = _environment_proxy_snapshot()
    selected = proxies.get(scheme) if supported_scheme else None
    matched = isinstance(selected, str) and bool(selected.strip())
    try:
        bypassed = bool(
            supported_scheme
            and parsed.netloc
            and _proxy_bypassed(parsed.netloc, proxies)
        )
    except (OSError, RuntimeError, TypeError, UnicodeError, ValueError):
        # An uninspectable bypass policy cannot prove a safe direct route.
        # Treat it as bypassed so a required-proxy call is blocked.
        bypassed = True
    if required:
        route = (
            "enforced"
            if supported_scheme and matched and not bypassed
            else "blocked"
        )
    elif supported_scheme and matched and not bypassed:
        route = "environment"
    else:
        route = "direct"
    return {
        "policy": "required" if required else "optional",
        "scheme": scheme if supported_scheme else "unknown",
        "matched": matched,
        "bypassed": bypassed,
        "route": route,
    }


def _open_remote_request(
    request: urllib.request.Request,
    *,
    timeout: float,
    context: Optional[ssl.SSLContext],
):
    """Open one request, optionally proving and pinning its proxy route.

    The default path preserves urllib's standard environment-proxy behaviour.
    When the caller opts into the fail-closed policy, take one proxy snapshot,
    reject missing or bypassed routes before transport, and build a per-request
    opener from that same snapshot.  This avoids relying on a process-global
    opener that may have cached an older environment.
    """

    if not _proxy_is_required():
        return urllib.request.urlopen(
            request,
            timeout=timeout,
            context=context,
        )

    scheme = request.type.lower()
    host = request.host
    proxies = _environment_proxy_snapshot()
    selected_proxy = proxies.get(scheme)
    if not isinstance(selected_proxy, str) or not selected_proxy.strip():
        raise LlmConfigError(
            "%s=1 but no matching HTTP(S) proxy is configured"
            % _REQUIRE_PROXY_ENV
        )
    if not host or _proxy_bypassed(host, proxies):
        raise LlmConfigError(
            "%s=1 but the LLM endpoint is excluded by proxy bypass policy"
            % _REQUIRE_PROXY_ENV
        )

    handlers = [_PinnedProxyHandler({scheme: selected_proxy})]
    if scheme == "https":
        handlers.append(urllib.request.HTTPSHandler(context=context))
    opener = urllib.request.build_opener(*handlers)
    return opener.open(request, timeout=timeout)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_receipt_text(
    value: object,
    *,
    token: str,
    base_url: str,
    limit: int = 512,
) -> Optional[str]:
    """Bound untrusted response metadata and redact known client secrets."""

    if value is None:
        return None
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    text = str(value).strip()
    sensitive_values = {token, base_url}
    sensitive_values.update(
        os.getenv(name, "") for name in _SENSITIVE_METADATA_ENV
    )
    for sensitive in sorted(sensitive_values, key=len, reverse=True):
        if sensitive:
            text = text.replace(sensitive, "redacted")
    text = "".join(
        character
        if (
            32 <= ord(character) < 127
            and (character.isalnum() or character in "._:-")
        )
        else "?"
        for character in text
    )
    return text[:limit] or None


def _exact_receipt_model_id(
    value: object,
    *,
    token: str,
    base_url: str,
) -> Optional[str]:
    """Keep an exact safe model identity; never compare a display sanitizer."""

    sensitive_values = {token, base_url}
    sensitive_values.update(
        os.getenv(name, "") for name in _SENSITIVE_METADATA_ENV
    )
    if not isinstance(value, str) or any(
        sensitive and sensitive in value for sensitive in sensitive_values
    ):
        return None
    try:
        return validate_model_id(value, field="response model")
    except LlmConfigError:
        return None


def _request_id(headers: object, *, token: str, base_url: str) -> Optional[str]:
    try:
        normalized = {
            str(name).lower(): value
            for name, value in headers.items()
        }
    except (AttributeError, TypeError, ValueError):
        return None
    for name in _REQUEST_ID_HEADERS:
        if name in normalized:
            return _safe_receipt_text(
                normalized[name], token=token, base_url=base_url
            )
    return None


def _integer_usage_items(
    value: object, *, token: str, base_url: str
) -> Dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result = {}
    sensitive_values = {token, base_url}
    sensitive_values.update(
        os.getenv(name, "") for name in _SENSITIVE_METADATA_ENV
    )
    for raw_name, raw_value in sorted(value.items(), key=lambda item: str(item[0])):
        name = str(raw_name)
        if (
            len(result) >= 32
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", name)
            or any(
                sensitive and sensitive in name
                for sensitive in sensitive_values
            )
            or isinstance(raw_value, bool)
            or not isinstance(raw_value, int)
            or raw_value < 0
            or raw_value > 2 ** 63 - 1
        ):
            continue
        result[name] = raw_value
    return result


def _set_response_socket_timeout(response: object, timeout: float) -> None:
    """Best-effort bound for one blocking urllib response read.

    ``urlopen(timeout=...)`` configures an inactivity timeout.  Streaming
    responses can otherwise keep resetting that timeout forever.  CPython's
    HTTPResponse exposes the connected socket through ``fp.raw._sock``; keep
    this lookup deliberately narrow and retain monotonic checks as a fallback
    for response doubles and alternate implementations.
    """

    fp = getattr(response, "fp", None)
    raw = getattr(fp, "raw", None)
    sock = getattr(raw, "_sock", None)
    if sock is None:
        return
    try:
        sock.settimeout(max(timeout, 0.001))
    except (AttributeError, OSError, TypeError, ValueError):
        return


def _read_response_body(
    response: object,
    *,
    deadline: Optional[float] = None,
) -> bytes:
    """Read a bounded non-stream body while enforcing a total deadline.

    ``urlopen(timeout=...)`` only limits socket inactivity.  A gateway that
    sends a few bytes before each inactivity interval can therefore keep a
    non-stream request alive indefinitely if the body is read in one call.
    Read bounded chunks, apply the remaining deadline to the underlying
    socket when available, and check the monotonic clock around every read.
    """

    payload = bytearray()
    while True:
        if deadline is not None:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise LlmTransportError("request_timeout")
            _set_response_socket_timeout(response, remaining_seconds)

        body_limit = _MAX_RESPONSE_BODY_BYTES
        if body_limit is None:
            read_size = _MAX_RESPONSE_READ_CHUNK_BYTES
        else:
            remaining_bytes = body_limit - len(payload)
            if remaining_bytes < 0:
                raise LlmTransportError("response_too_large")
            read_size = min(
                _MAX_RESPONSE_READ_CHUNK_BYTES,
                remaining_bytes + 1,
            )
        try:
            chunk = response.read(read_size)
        except TimeoutError:
            if deadline is not None:
                raise LlmTransportError("request_timeout") from None
            raise
        if not isinstance(chunk, bytes):
            raise LlmTransportError("invalid_response_read")
        if (
            body_limit is not None
            and len(chunk) > body_limit - len(payload)
        ):
            raise LlmTransportError("response_too_large")
        payload.extend(chunk)

        if deadline is not None and time.monotonic() > deadline:
            raise LlmTransportError("request_timeout")
        if not chunk:
            return bytes(payload)


def _read_stream_response_body(
    response: object,
    *,
    deadline: Optional[float] = None,
) -> bytes:
    """Read bounded SSE bytes through the terminal event, not HTTP EOF.

    OpenAI-compatible gateways commonly keep an HTTP/1.1 chunked connection
    alive after sending ``data: [DONE]``.  Waiting for EOF in that case turns a
    complete response into a socket timeout.  Read bounded lines so one
    complete SSE event can terminate the response without accepting a partial
    marker.  The returned bytes are exactly the response-body bytes consumed
    and remain the input to both strict parsing and the audit hash.
    """

    payload = bytearray()
    event_data_line_count = 0
    event_single_data_is_done = False
    while True:
        if deadline is not None:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise LlmTransportError("request_timeout")
            _set_response_socket_timeout(response, remaining_seconds)
        body_limit = _MAX_RESPONSE_BODY_BYTES
        try:
            if body_limit is None:
                line = response.readline()
            else:
                remaining = body_limit - len(payload)
                if remaining < 0:
                    raise LlmTransportError("response_too_large")
                line = response.readline(remaining + 1)
        except TimeoutError:
            if deadline is not None:
                raise LlmTransportError("request_timeout") from None
            raise
        if not isinstance(line, bytes):
            raise LlmTransportError("invalid_stream_read")
        if not line:
            return bytes(payload)
        payload.extend(line)
        if body_limit is not None and len(payload) > body_limit:
            raise LlmTransportError("response_too_large")
        if deadline is not None and time.monotonic() > deadline:
            raise LlmTransportError("request_timeout")

        content = line
        if content.endswith(b"\n"):
            content = content[:-1]
            if content.endswith(b"\r"):
                content = content[:-1]
        if content:
            if content.startswith(b"data:"):
                value = content[5:]
                if value.startswith(b" "):
                    value = value[1:]
                event_data_line_count += 1
                event_single_data_is_done = (
                    event_data_line_count == 1 and value == b"[DONE]"
                )
            continue
        if event_data_line_count == 1 and event_single_data_is_done:
            return bytes(payload)
        event_data_line_count = 0
        event_single_data_is_done = False


def _decode_stream_completion(payload: bytes) -> Dict[str, object]:
    """Strictly aggregate one bounded OpenAI-compatible SSE completion."""

    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError("invalid_stream_utf8") from exc
    events = re.split(r"\r?\n\r?\n", text)
    chunks = []
    saw_done = False
    for event in events:
        if not event.strip():
            continue
        data_lines = []
        for line in event.splitlines():
            if not line or line.startswith(":"):
                continue
            if line.startswith("event:"):
                if line[6:].strip() not in {"", "message"}:
                    raise ValueError("invalid_stream_event")
                continue
            if not line.startswith("data:"):
                raise ValueError("invalid_stream_line")
            value = line[5:]
            data_lines.append(value[1:] if value.startswith(" ") else value)
        if not data_lines:
            continue
        data_text = "\n".join(data_lines)
        if data_text == "[DONE]":
            if saw_done:
                raise ValueError("duplicate_stream_done")
            saw_done = True
            continue
        if saw_done:
            raise ValueError("stream_data_after_done")
        if len(chunks) >= _MAX_STREAM_EVENTS:
            raise ValueError("too_many_stream_events")
        try:
            chunk = json.loads(data_text)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid_stream_json") from exc
        if not isinstance(chunk, dict):
            raise ValueError("non_object_stream_json")
        chunks.append(chunk)
    if not saw_done or not chunks:
        raise ValueError("incomplete_stream")

    response_id = None
    response_model = None
    response_created = None
    usage = None
    content_parts = []
    for chunk in chunks:
        for field, current in (
            ("id", response_id),
            ("model", response_model),
            ("created", response_created),
        ):
            value = chunk.get(field)
            if value is None:
                continue
            if current is not None and value != current:
                raise ValueError("inconsistent_stream_metadata")
            if field == "id":
                response_id = value
            elif field == "model":
                response_model = value
            else:
                response_created = value
        if chunk.get("usage") is not None:
            if not isinstance(chunk["usage"], dict):
                raise ValueError("invalid_stream_usage")
            usage = chunk["usage"]
        choices = chunk.get("choices")
        if choices == []:
            continue
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("invalid_stream_choices")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ValueError("invalid_stream_choice")
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            raise ValueError("invalid_stream_delta")
        reasoning = delta.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError("invalid_stream_reasoning")
        content = delta.get("content")
        if content is not None:
            if not isinstance(content, str):
                raise ValueError("invalid_stream_content")
            content_parts.append(content)
    return {
        "id": response_id,
        "model": response_model,
        "created": response_created,
        "usage": usage or {},
        "choices": [{"message": {"content": "".join(content_parts)}}],
    }

_SYSTEM_PROMPT = (
    "你是 NumPilot 的 Code Migrator。把用户给的 NumPy 程序迁移为 AsNumpy(NPU) 版本。\n"
    "严格遵守下面的 AsNumpy 代码格式；Planner 任务书中的逐调用点事实优先于一般规则。\n"
    "只输出完整的迁移后 Python 源码, 用 ```python 代码块包裹, 不要任何解释文字。\n\n"
    + render_asnumpy_guide()
)


def _taskbook_block(taskbook) -> str:
    """Render the Planner hand-off as a stable, model-readable block."""

    if not taskbook:
        return ""
    return (
        "\n\n迁移任务书（Planner 生成，逐调用点事实；含已完成的整程序准备）:\n"
        "```json\n"
        + json.dumps(taskbook, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```"
    )


def _diagnosis_block(diagnosis) -> str:
    if hasattr(diagnosis, "to_dict"):
        value = diagnosis.to_dict()
    else:
        value = diagnosis
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)


class LlmEngine(MigrationEngine):
    name = "llm"

    def __init__(self, strict: bool = False):
        self.strict = strict
        self.last_receipt: Dict[str, object] = {}
        self.last_identity: Dict[str, object] = {}

    def is_configured(self) -> bool:
        return all(os.getenv(k) for k in _REQUIRED_ENV)

    def rewrite(
        self, source: str, plan: OffloadPlan, taskbook=None
    ) -> str:
        self.last_receipt = {}
        self.last_identity = {}
        npu_ops = sorted(op for op, p in plan.placement.items() if p == "NPU")
        user_msg = (
            f"需上 NPU 的算子: {npu_ops}\n\n"
            f"源码:\n```python\n{source}\n```"
            + _taskbook_block(taskbook)
        )
        return self._request_code(_SYSTEM_PROMPT, user_msg)

    def repair(
        self, source: str, plan: OffloadPlan, previous_source: str, diagnosis,
        taskbook=None,
    ) -> str:
        """One bounded repair attempt driven by the Diagnostics Agent."""

        self.last_receipt = {}
        self.last_identity = {}
        npu_ops = sorted(op for op, p in plan.placement.items() if p == "NPU")
        user_msg = (
            "上一版迁移没有通过确定性验证。请依据诊断修复，并输出完整的新源码。\n"
            f"需上 NPU 的算子: {npu_ops}\n\n"
            f"诊断:\n```json\n{_diagnosis_block(diagnosis)}\n```\n\n"
            f"上一版迁移源码:\n```python\n{previous_source}\n```\n\n"
            f"原始（已准备）源码:\n```python\n{source}\n```"
            + _taskbook_block(taskbook)
        )
        return self._request_code(_SYSTEM_PROMPT, user_msg)

    def _request_code(self, system: str, user_msg: str) -> str:
        if not self.is_configured():
            missing = [k for k in _REQUIRED_ENV if not os.getenv(k)]
            raise LlmError(
                "LLM 引擎未配置, 缺环境变量: " + ", ".join(missing)
                + "。请在仓库根目录执行 `install -m 600 llm.example.json llm.json` "
                  "后填写，或运行 `python -m numpilot configure-llm` 后重试。"
            )

        model = os.environ["NUMPILOT_LLM_MODEL"]
        try:
            identity = require_model_configuration(
                os.getenv("NUMPILOT_LLM_PROVIDER", DEFAULT_PROVIDER_LABEL),
                model,
                protocol=os.getenv(
                    "NUMPILOT_LLM_PROTOCOL",
                    OPENAI_CHAT_COMPLETIONS_PROTOCOL,
                ),
                response_model_aliases=(
                    response_model_aliases_from_environment(
                        configured_model=model
                    )
                ),
            )
            content, receipt = self._chat_with_receipt(system, user_msg)
            validated_receipt = validate_configured_model_receipt(
                receipt,
                content,
                expected_model=identity.model,
                expected_provider=identity.provider,
                expected_protocol=identity.protocol,
                response_model_aliases=identity.response_model_aliases,
            )
            code = self._extract_code(content)
        except Exception as e:
            raise LlmError(f"调用模型 {model} 失败(无响应/网络/解析错误): "
                           f"{type(e).__name__}: {str(e)[:200]}") from e

        if not code or "ap." not in code:
            raise LlmError(f"模型 {model} 返回未包含有效迁移代码(缺少 ap.* 调用), "
                           "无法确认迁移成功。")
        self.last_receipt = validated_receipt
        self.last_identity = {
            "generated_by": "configured_model",
            "provider": identity.provider,
            "protocol": identity.protocol,
            "configured_model": identity.model,
            "response_model": validated_receipt["response_model"],
            "response_model_identity": validated_receipt[
                "response_model_identity"
            ],
            "response_model_aliases": list(identity.response_model_aliases),
            "artifact_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
            "response_content_sha256": validated_receipt["content_sha256"],
        }
        return code

    @staticmethod
    def _chat(system: str, user: str) -> str:
        """Compatibility wrapper returning only model content."""

        content, _receipt = LlmEngine._chat_with_receipt(system, user)
        return content

    @staticmethod
    def _chat_with_receipt(
        system: str,
        user: str,
        *,
        max_tokens: Optional[int] = None,
        thinking_mode: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        stream: bool = False,
    ) -> Tuple[str, Dict[str, object]]:
        """Perform one real request and return content plus a redacted receipt.

        ``max_tokens`` is an optional explicit bound.  When it is omitted and
        ``NUMPILOT_LLM_MAX_TOKENS`` is unset, the request omits ``max_tokens``
        so the configured model/gateway applies its own output limit;
        operator generation must not be capped by a small framework default.
        ``thinking_mode`` and ``reasoning_effort`` expose GLM-compatible
        reasoning controls only when a caller deliberately selects them.
        Omitting them preserves the historical request body.  ``stream=True``
        strictly aggregates a bounded SSE response while retaining a hash of
        the consumed HTTP entity-body bytes through its terminal event.
        """

        if max_tokens is None:
            configured_max_tokens = os.getenv(
                "NUMPILOT_LLM_MAX_TOKENS", ""
            ).strip()
            if configured_max_tokens:
                try:
                    request_max_tokens = int(configured_max_tokens)
                except ValueError as exc:
                    raise LlmConfigError(
                        "LLM max_tokens 必须是数字"
                    ) from exc
                if not (
                    _MIN_PER_CALL_MAX_TOKENS
                    <= request_max_tokens
                    <= _MAX_PER_CALL_MAX_TOKENS
                ):
                    raise LlmConfigError(
                        "LLM max_tokens must be an integer in 1..1000000"
                    )
            else:
                request_max_tokens = None
        elif (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < _MIN_PER_CALL_MAX_TOKENS
            or max_tokens > _MAX_PER_CALL_MAX_TOKENS
        ):
            raise LlmConfigError(
                "per-call LLM max_tokens must be an integer in 1..1000000"
            )
        else:
            request_max_tokens = max_tokens
        if thinking_mode not in (None, "enabled", "disabled"):
            raise LlmConfigError(
                "per-call LLM thinking_mode must be enabled or disabled"
            )
        if reasoning_effort not in (
            None, "max", "xhigh", "high", "medium", "low", "minimal", "none"
        ):
            raise LlmConfigError(
                "per-call LLM reasoning_effort is invalid"
            )
        if not isinstance(stream, bool):
            raise LlmConfigError("per-call LLM stream must be a boolean")

        # Validate the shared timeout profile before touching required
        # environment keys.  Direct callers (including the probe path) may
        # provide a private config file instead of preloading its values into
        # ``os.environ``; indexing the mapping first used to leak ``KeyError``
        # and made an invalid timeout impossible to diagnose cleanly.
        timeout = validate_llm_timeout(
            os.getenv("NUMPILOT_LLM_TIMEOUT", "120")
        )
        status = load_llm_config()
        if not status.configured:
            raise LlmConfigError(
                "LLM 未配置，缺少: " + ", ".join(status.missing)
            )

        # ``load_llm_config`` is the single configuration boundary.  Use
        # ``get`` after it succeeds so a concurrent environment edit still
        # becomes a stable configuration error rather than a raw KeyError.
        base = os.getenv("NUMPILOT_LLM_BASE_URL", "").rstrip("/")
        token = os.getenv("NUMPILOT_LLM_AUTH_TOKEN", "")
        configured_model = validate_model_id(
            os.getenv("NUMPILOT_LLM_MODEL", "")
        )
        requested_model = configured_model
        provider = validate_provider_label(
            os.getenv("NUMPILOT_LLM_PROVIDER", DEFAULT_PROVIDER_LABEL)
        )
        protocol = validate_protocol(
            os.getenv(
                "NUMPILOT_LLM_PROTOCOL", OPENAI_CHAT_COMPLETIONS_PROTOCOL
            )
        )
        response_model_aliases = response_model_aliases_from_environment(
            configured_model=configured_model
        )
        url = base + "/chat/completions"
        request_data = {
            "model": requested_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
        }
        if request_max_tokens is not None:
            request_data["max_tokens"] = request_max_tokens
        if thinking_mode is not None:
            request_data["thinking"] = {"type": thinking_mode}
        if reasoning_effort is not None:
            request_data["reasoning_effort"] = reasoning_effort
        if stream:
            request_data["stream"] = True
        body = json.dumps(request_data).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", "Bearer " + token)
        # 默认校验证书；只有受信内网自签网关才可显式选择 insecure。
        insecure = os.getenv("NUMPILOT_LLM_INSECURE", "0") == "1"
        tls_verification_enabled = base.lower().startswith("https://") and not insecure
        ctx = ssl._create_unverified_context() if insecure else None
        request_deadline = time.monotonic() + timeout
        started_at = _utc_now()
        try:
            response = _open_remote_request(
                req,
                timeout=timeout,
                context=ctx,
            )
        except LlmConfigError:
            raise
        except urllib.error.HTTPError as exc:
            status_code = (
                exc.code
                if (
                    not isinstance(exc.code, bool)
                    and isinstance(exc.code, int)
                )
                else 0
            )
            try:
                exc.close()
            except Exception:
                pass
            raise LlmHttpError(status_code) from None
        except TimeoutError:
            raise LlmTransportError("request_timeout") from None
        except urllib.error.URLError as exc:
            if isinstance(getattr(exc, "reason", None), TimeoutError):
                raise LlmTransportError("request_timeout") from None
            raise LlmTransportError("URLError") from None
        except Exception as exc:
            raise LlmTransportError(type(exc).__name__) from None
        try:
            with response as resp:
                status = getattr(resp, "status", None)
                if status is None:
                    status = resp.getcode()
                if isinstance(status, bool) or not isinstance(status, int):
                    raise LlmTransportError("invalid_http_status")
                if status < 200 or status >= 300:
                    raise LlmHttpError(status)
                headers = getattr(resp, "headers", {})
                response_body = (
                    _read_stream_response_body(
                        resp,
                        deadline=request_deadline,
                    )
                    if stream
                    else _read_response_body(
                        resp,
                        deadline=request_deadline,
                    )
                )
        except (LlmHttpError, LlmTransportError):
            raise
        except Exception as exc:
            raise LlmTransportError(type(exc).__name__) from None
        ended_at = _utc_now()
        receipt = {
            "schema_version": 2,
            "protocol": protocol,
            "provider": provider,
            "configured_model": configured_model,
            "utc_start": started_at,
            "utc_end": ended_at,
            "http_status": status,
            "requested_model": requested_model,
            "response_model": None,
            "response_model_aliases": list(response_model_aliases),
            "response_model_identity": "missing",
            "model_identity_sha256": None,
            "response_id": None,
            "response_created": None,
            "request_body_sha256": hashlib.sha256(body).hexdigest(),
            "response_body_sha256": hashlib.sha256(response_body).hexdigest(),
            "content_sha256": None,
            "request_id": _request_id(headers, token=token, base_url=base),
            "usage": {},
            "tls_verification": (
                "enabled" if tls_verification_enabled else "disabled"
            ),
        }
        receipt["model_identity_sha256"] = model_identity_sha256(
            protocol=protocol,
            provider=provider,
            configured_model=configured_model,
            requested_model=requested_model,
            response_model=None,
            response_model_aliases=response_model_aliases,
            response_model_identity="missing",
        )
        try:
            data = (
                _decode_stream_completion(response_body)
                if stream
                else json.loads(response_body.decode("utf-8"))
            )
        except ValueError as exc:
            reason = str(exc) if stream else "invalid_json"
            raise LlmResponseError(reason, receipt) from None
        if not isinstance(data, dict):
            raise LlmResponseError("non_object_json", receipt)
        receipt.update({
            "response_model": _exact_receipt_model_id(
                data.get("model"), token=token, base_url=base
            ),
            "response_id": _safe_receipt_text(
                data.get("id"), token=token, base_url=base
            ),
            "response_created": (
                data.get("created")
                if (
                    not isinstance(data.get("created"), bool)
                    and isinstance(data.get("created"), int)
                    and data.get("created") >= 0
                )
                else None
            ),
            "usage": _integer_usage_items(
                data.get("usage"), token=token, base_url=base
            ),
        })
        if receipt["response_model"] == configured_model:
            response_identity = "exact"
        elif receipt["response_model"] in response_model_aliases:
            response_identity = "explicit_alias"
        elif receipt["response_model"] is None:
            response_identity = "missing"
        else:
            response_identity = "mismatch"
        receipt["response_model_identity"] = response_identity
        receipt["model_identity_sha256"] = model_identity_sha256(
            protocol=protocol,
            provider=provider,
            configured_model=configured_model,
            requested_model=requested_model,
            response_model=receipt["response_model"],
            response_model_aliases=response_model_aliases,
            response_model_identity=response_identity,
        )
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LlmResponseError("missing_content", receipt) from None
        if not isinstance(content, str):
            raise LlmResponseError("non_text_content", receipt)
        receipt["content_sha256"] = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()
        return content, receipt

    @staticmethod
    def _extract_code(content: str) -> str:
        m = re.search(r"```(?:python)?\n(.*?)```", content, re.DOTALL)
        return (m.group(1) if m else content).strip()
