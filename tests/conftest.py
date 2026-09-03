"""Suite-wide isolation from developer workstation process state.

Several CLI helpers intentionally hydrate the configured LLM into
``os.environ``.  That is the right boundary for a short-lived command, but
pytest executes many commands in one interpreter and direct assignments are
not tracked by ``monkeypatch``.  Keep tests deterministic by restoring the
LLM-related variables after every test as well as clearing proxy policy.
"""

import os
import pytest


_REMOTE_ROUTE_ENV = (
    "NUMPILOT_LLM_REQUIRE_PROXY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)

_LLM_CONFIG_ENV = (
    # Runtime aliases also affect the default llm.json location.  Keep them
    # isolated so a developer's activated competition runtime cannot change
    # unrelated tests (or leak into a receipt fixture).
    "NUMPILOT_HOME",
    "NUMPILOT_RUNTIME_ROOT",
    "NUMPILOT_CONFIG_HOME",
    "NUMPILOT_LLM_BASE_URL",
    "NUMPILOT_LLM_AUTH_TOKEN",
    "NUMPILOT_LLM_MODEL",
    "NUMPILOT_LLM_PROVIDER",
    "NUMPILOT_LLM_PROTOCOL",
    "NUMPILOT_LLM_RESPONSE_MODEL_ALIASES",
    "NUMPILOT_LLM_CONFIG_FILE",
    "NUMPILOT_LLM_TIMEOUT",
    "NUMPILOT_LLM_MAX_TOKENS",
    "NUMPILOT_LLM_INSECURE",
)


@pytest.fixture(autouse=True)
def isolate_remote_route_environment(monkeypatch):
    """Make proxy behavior explicit in tests that exercise it."""

    # ``load_llm_config``/``configure_llm_interactive`` may assign directly to
    # os.environ.  Snapshot and restore manually because monkeypatch only
    # restores assignments made through its own API.
    llm_snapshot = {name: os.environ.get(name) for name in _LLM_CONFIG_ENV}
    for name in _REMOTE_ROUTE_ENV:
        monkeypatch.delenv(name, raising=False)
    for name in _LLM_CONFIG_ENV:
        monkeypatch.delenv(name, raising=False)
    yield
    for name, value in llm_snapshot.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
