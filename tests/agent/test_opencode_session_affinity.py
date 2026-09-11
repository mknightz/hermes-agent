"""x-opencode-session rides on every OpenCode request.

On 2026-09-08 OpenCode Go (opencode.ai/zen/go) started rejecting requests
without the header with ``HTTP 400 MissingSessionID``. Tiers 1+2 of both
agents' fallback chains live on opencode-go, so every turn fell through to
the paid OpenRouter tier. Backport of upstream 139396995, adapted to this
fork (see ``agent/opencode_affinity.py`` for the deviations).
"""

from __future__ import annotations

import sys
import types

import pytest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import hermes_logging
from agent import auxiliary_client as aux
from agent.chat_completion_helpers import build_api_kwargs
from agent.opencode_affinity import (
    OPENCODE_SESSION_HEADER,
    is_opencode_target,
    merge_opencode_session_headers,
    opencode_session_headers,
)
from run_agent import AIAgent

_MSGS = [{"role": "user", "content": "hi"}]
GO = "https://opencode.ai/zen/go/v1"
ZEN = "https://opencode.ai/zen/v1"
OR = "https://openrouter.ai/api/v1"


class _FakeOpenAI:
    def __init__(self, **kw):
        self.api_key = kw.get("api_key", "test")
        self.base_url = kw.get("base_url", "http://test")

    def close(self):
        pass


def _agent(monkeypatch, provider, model, base_url, session_id="sess-affinity-1"):
    monkeypatch.setattr("run_agent.get_tool_definitions", lambda **kw: [])
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda: {})
    monkeypatch.setattr("run_agent.OpenAI", _FakeOpenAI)
    return AIAgent(
        api_key="test-key",
        base_url=base_url,
        model=model,
        provider=provider,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_id=session_id,
    )


# ── target detection ──────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "provider, base_url, expected",
    [
        ("opencode-go", GO, True),
        ("opencode-zen", ZEN, True),
        ("opencode-free", ZEN, True),
        ("custom", GO, True),                       # URL-only detection
        ("custom", "https://opencode.ai/zen/go/v1/", True),
        ("opencode-go-custom", "https://proxy.example/v1", True),  # custom opencode-* id
        ("openrouter", OR, False),
        ("custom", "https://evil.com/opencode.ai/v1", False),      # substring, not host
        ("custom", "https://opencode.ai.evil/v1", False),
        (None, None, False),
    ],
)
def test_is_opencode_target(provider, base_url, expected):
    assert is_opencode_target(provider, base_url) is expected


def test_headers_empty_for_non_opencode_and_present_for_opencode():
    assert opencode_session_headers("openrouter", OR, "s1") == {}
    assert opencode_session_headers("opencode-go", GO, "s1") == {OPENCODE_SESSION_HEADER: "s1"}


def test_merge_preserves_caller_pinned_header_and_other_headers():
    kwargs = {"extra_headers": {"x-opencode-session": "pinned", "x-other": "1"}}
    merge_opencode_session_headers(kwargs, "opencode-go", GO, "s1")
    assert kwargs["extra_headers"] == {"x-opencode-session": "pinned", "x-other": "1"}
    untouched = {"model": "m"}
    merge_opencode_session_headers(untouched, "openrouter", OR, "s1")
    assert "extra_headers" not in untouched


# ── main turn ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "provider, model, base_url",
    [
        ("opencode-go", "qwen3.8-max", GO),          # José primary, chat_completions
        ("opencode-go", "kimi-k3", GO + "/"),        # tier-2 fallback, trailing slash
        ("opencode-zen", "glm-5", ZEN),
        ("custom", "glm-5", GO),                     # URL-only detection
        ("opencode-go", "minimax-m2.7", GO),         # routes to anthropic_messages
    ],
)
def test_main_turn_sends_stable_session_header(monkeypatch, provider, model, base_url):
    agent = _agent(monkeypatch, provider, model, base_url)
    first = build_api_kwargs(agent, _MSGS)["extra_headers"][OPENCODE_SESSION_HEADER]
    second = build_api_kwargs(agent, _MSGS)["extra_headers"][OPENCODE_SESSION_HEADER]
    assert first == second == "sess-affinity-1"


def test_main_turn_no_header_for_openrouter(monkeypatch):
    other = _agent(monkeypatch, "openrouter", "moonshotai/kimi-k3", OR)
    assert OPENCODE_SESSION_HEADER not in (build_api_kwargs(other, _MSGS).get("extra_headers") or {})


def test_main_turn_without_session_id_still_sends_a_stable_key(monkeypatch):
    # Go hard-rejects header-less requests, so a missing session id must not
    # mean a missing header (fork deviation from upstream — see module doc).
    agent = _agent(monkeypatch, "opencode-go", "qwen3.8-max", GO, session_id=None)
    first = build_api_kwargs(agent, _MSGS)["extra_headers"][OPENCODE_SESSION_HEADER]
    second = build_api_kwargs(agent, _MSGS)["extra_headers"][OPENCODE_SESSION_HEADER]
    assert first and first == second


# ── auxiliary calls ───────────────────────────────────────────────────────

def test_auxiliary_calls_share_the_main_turn_session_key():
    # conversation_loop sets both at the top of every turn.
    hermes_logging.set_session_context("sess-affinity-1")
    aux.set_runtime_main("opencode-go", "qwen3.8-max", session_id="sess-affinity-1")
    try:
        kwargs = aux._build_call_kwargs("opencode-go", "qwen3.8-max", _MSGS, base_url=GO)
        assert kwargs["extra_headers"][OPENCODE_SESSION_HEADER] == "sess-affinity-1"
        other = aux._build_call_kwargs("openrouter", "x", _MSGS, base_url=OR)
        assert OPENCODE_SESSION_HEADER not in (other.get("extra_headers") or {})
    finally:
        aux.clear_runtime_main()
        hermes_logging.clear_session_context()


def test_auxiliary_calls_prefer_the_turn_thread_session_over_the_process_global():
    # The gateway runs conversations concurrently on different threads; the
    # per-thread logging context is the authoritative "current turn" id.
    hermes_logging.set_session_context("thread-sess")
    aux.set_runtime_main("opencode-go", "qwen3.8-max", session_id="other-sess")
    try:
        kwargs = aux._build_call_kwargs("opencode-go", "qwen3.8-max", _MSGS, base_url=GO)
        assert kwargs["extra_headers"][OPENCODE_SESSION_HEADER] == "thread-sess"
    finally:
        aux.clear_runtime_main()
        hermes_logging.clear_session_context()


def test_auxiliary_calls_without_runtime_session_use_a_stable_process_key():
    aux.clear_runtime_main()
    hermes_logging.clear_session_context()
    first = aux._build_call_kwargs("opencode-go", "qwen3.8-max", _MSGS, base_url=GO)
    second = aux._build_call_kwargs("opencode-go", "qwen3.8-max", _MSGS, base_url=GO)
    k1 = first["extra_headers"][OPENCODE_SESSION_HEADER]
    k2 = second["extra_headers"][OPENCODE_SESSION_HEADER]
    assert k1 and k1 == k2


# ── auxiliary adapters forward per-request headers to the SDK ─────────────

class _Stop(Exception):
    """Short-circuit the adapter right after the SDK call is issued."""


class _Recorder:
    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        raise _Stop()


def test_aux_codex_adapter_forwards_extra_headers():
    rec = _Recorder()
    fake = types.SimpleNamespace(responses=types.SimpleNamespace(stream=rec))
    adapter = aux._CodexCompletionsAdapter(fake, "gpt-5.1-codex")
    with pytest.raises(_Stop):
        adapter.create(model="gpt-5.1-codex", messages=_MSGS, extra_headers={OPENCODE_SESSION_HEADER: "s1"})
    assert rec.kwargs["extra_headers"] == {OPENCODE_SESSION_HEADER: "s1"}


def test_aux_anthropic_adapter_forwards_extra_headers():
    rec = _Recorder()
    fake = types.SimpleNamespace(messages=types.SimpleNamespace(create=rec))
    adapter = aux._AnthropicCompletionsAdapter(fake, "minimax-m2.7")
    with pytest.raises(_Stop):
        adapter.create(model="minimax-m2.7", messages=_MSGS, max_tokens=16, extra_headers={OPENCODE_SESSION_HEADER: "s1"})
    assert rec.kwargs["extra_headers"][OPENCODE_SESSION_HEADER] == "s1"
