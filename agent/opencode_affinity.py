"""``x-opencode-session`` — OpenCode relay session-affinity header.

Backport of upstream NousResearch/hermes-agent 139396995 (2026-09-03),
adapted to this fork (``mygel/local``), which predates the helper modules
upstream leans on (``opencode_provider_family``, ``_is_opencode_endpoint``,
``get_affinity_scope``, ``_cache_scope_from_session_id``).

Why it matters here: on 2026-09-08 OpenCode Go started rejecting requests
that lack the header with ``HTTP 400 MissingSessionID`` ("Request is missing
x-opencode-session and cannot be routed efficiently"). Both José and Charlie
run tiers 1+2 of their fallback chain on opencode-go, so every turn fell
through to the paid OpenRouter tier. OpenCode also uses the value to pin a
conversation to one backend so its prompt cache stays warm.

Every OpenCode request — main turn on any transport, auxiliary calls
(compression, titles, vision), max-iteration summaries — goes through
:func:`merge_opencode_session_headers` so the header cannot drift per code
path.

Fork deviation from upstream: when no session id is known (out-of-turn
auxiliary calls, ad-hoc scripts) we still send a process-stable key instead
of omitting the header, because the Go relay hard-rejects header-less
requests. Upstream tracks the same gap as #105023.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

OPENCODE_SESSION_HEADER = "x-opencode-session"

# Built-in OpenCode provider ids plus the aliases hermes_cli maps onto them.
_OPENCODE_PROVIDER_IDS = frozenset({
    "opencode",
    "opencode-go",
    "opencode-zen",
    "opencode-free",
    "go",
    "opencode-go-sub",
})

# Stable for the life of the process — used only when no session id is known.
_PROCESS_FALLBACK_KEY = f"hermes-proc-{uuid.uuid4().hex}"


def is_opencode_target(provider: Optional[str], base_url: Optional[str]) -> bool:
    """True when *provider* or *base_url* addresses the OpenCode relay.

    Matches the built-in opencode-zen/go/free providers, custom
    ``opencode-*`` providers, and any base_url hosted on opencode.ai
    (hostname match, not substring — see ``utils.base_url_host_matches``).
    """
    p = (provider or "").strip().lower()
    if p in _OPENCODE_PROVIDER_IDS or p.startswith("opencode-"):
        return True
    try:
        from utils import base_url_host_matches

        return base_url_host_matches(str(base_url or ""), "opencode.ai")
    except Exception:
        return False


def opencode_session_key(session_id: Optional[str] = None) -> str:
    """Opaque, per-conversation key: the Hermes session id, else a process key."""
    sid = str(session_id or "").strip()
    return sid or _PROCESS_FALLBACK_KEY


def opencode_session_headers(
    provider: Optional[str],
    base_url: Optional[str],
    session_id: Optional[str] = None,
) -> Dict[str, str]:
    """Return ``{"x-opencode-session": <key>}`` for OpenCode targets, else ``{}``."""
    if not is_opencode_target(provider, base_url):
        return {}
    return {OPENCODE_SESSION_HEADER: opencode_session_key(session_id)}


def merge_opencode_session_headers(
    kwargs: Dict[str, Any],
    provider: Optional[str],
    base_url: Optional[str],
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Merge the affinity header into ``kwargs["extra_headers"]`` (in place).

    Existing per-request headers win, so a caller-pinned value is preserved.
    Non-OpenCode targets are left untouched. Returns *kwargs* for chaining.
    """
    headers = opencode_session_headers(provider, base_url, session_id)
    if headers:
        existing = kwargs.get("extra_headers")
        merged = dict(existing) if isinstance(existing, dict) else {}
        for key, value in headers.items():
            merged.setdefault(key, value)
        kwargs["extra_headers"] = merged
    return kwargs
