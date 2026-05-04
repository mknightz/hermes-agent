"""Tests for the bundled observability/langfuse plugin."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "observability" / "langfuse"


# ---------------------------------------------------------------------------
# Manifest + layout
# ---------------------------------------------------------------------------

class TestManifest:
    def test_plugin_directory_exists(self):
        assert PLUGIN_DIR.is_dir()
        assert (PLUGIN_DIR / "plugin.yaml").exists()
        assert (PLUGIN_DIR / "__init__.py").exists()

    def test_manifest_fields(self):
        data = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text())
        assert data["name"] == "langfuse"
        assert data["version"]
        # All six hooks the plugin implements.
        assert set(data["hooks"]) == {
            "pre_api_request", "post_api_request",
            "pre_llm_call", "post_llm_call",
            "pre_tool_call", "post_tool_call",
        }
        # Required env vars are the user-facing HERMES_ prefixed keys.
        assert "HERMES_LANGFUSE_PUBLIC_KEY" in data["requires_env"]
        assert "HERMES_LANGFUSE_SECRET_KEY" in data["requires_env"]


# ---------------------------------------------------------------------------
# Plugin discovery: langfuse is opt-in (not loaded unless explicitly enabled).
# This guards against someone accidentally re-introducing a per-hook
# load_config() gate or making the plugin auto-load.
# ---------------------------------------------------------------------------

class TestDiscovery:
    def test_plugin_is_discovered_as_standalone_opt_in(self, tmp_path, monkeypatch):
        """Scanner should find the plugin but NOT load it by default."""
        from hermes_cli import plugins as plugins_mod

        # Isolated HERMES_HOME so we don't read the developer's config.yaml.
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        manager = plugins_mod.PluginManager()
        manager.discover_and_load()

        # observability/langfuse appears in the plugin registry …
        loaded = manager._plugins.get("observability/langfuse")
        assert loaded is not None, "plugin not discovered"
        # … but is not loaded (opt-in default → no config.yaml means nothing enabled)
        assert loaded.enabled is False
        assert "not enabled" in (loaded.error or "").lower()


# ---------------------------------------------------------------------------
# Runtime gate: _get_langfuse() returns None and caches _INIT_FAILED when
# credentials are missing. Guards against regressing toward the rejected
# per-hook load_config() design.
# ---------------------------------------------------------------------------

class TestRuntimeGate:
    def _fresh_plugin(self):
        """Import the plugin module fresh (clears any cached client)."""
        mod_name = "plugins.observability.langfuse"
        sys.modules.pop(mod_name, None)
        return importlib.import_module(mod_name)

    def test_get_langfuse_returns_none_without_credentials(self, monkeypatch):
        for k in (
            "HERMES_LANGFUSE_PUBLIC_KEY", "HERMES_LANGFUSE_SECRET_KEY",
            "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
        ):
            monkeypatch.delenv(k, raising=False)

        langfuse_plugin = self._fresh_plugin()
        assert langfuse_plugin._get_langfuse() is None

    def test_get_langfuse_caches_failure_no_config_load(self, monkeypatch):
        """A miss must be cached — no per-hook config.yaml reads, no env re-reads."""
        for k in (
            "HERMES_LANGFUSE_PUBLIC_KEY", "HERMES_LANGFUSE_SECRET_KEY",
            "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
        ):
            monkeypatch.delenv(k, raising=False)

        langfuse_plugin = self._fresh_plugin()

        # Prime the cache with one call.
        assert langfuse_plugin._get_langfuse() is None

        # Now block os.environ.get — a correctly-cached plugin must not
        # touch env again.
        import os
        called = {"n": 0}
        real_get = os.environ.get

        def tracking_get(key, default=None):
            if key.startswith(("HERMES_LANGFUSE_", "LANGFUSE_")):
                called["n"] += 1
            return real_get(key, default)

        monkeypatch.setattr(os.environ, "get", tracking_get)

        for _ in range(20):
            assert langfuse_plugin._get_langfuse() is None

        assert called["n"] == 0, (
            f"_get_langfuse() re-read env {called['n']} times after cache miss — "
            "it should short-circuit via _INIT_FAILED"
        )

    def test_get_langfuse_does_not_import_hermes_config(self, monkeypatch):
        """The plugin must not re-read config.yaml per hook."""
        for k in (
            "HERMES_LANGFUSE_PUBLIC_KEY", "HERMES_LANGFUSE_SECRET_KEY",
            "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
        ):
            monkeypatch.delenv(k, raising=False)

        # Drop any cached import of hermes_cli.config.
        sys.modules.pop("hermes_cli.config", None)

        langfuse_plugin = self._fresh_plugin()
        for _ in range(20):
            langfuse_plugin._get_langfuse()

        assert "hermes_cli.config" not in sys.modules, (
            "langfuse plugin imported hermes_cli.config — regression toward "
            "the rejected per-hook load_config() design"
        )


# ---------------------------------------------------------------------------
# Hooks are inert when the client is unavailable.
# ---------------------------------------------------------------------------

class TestHooksInert:
    def test_hooks_noop_without_client(self, monkeypatch):
        """All 6 hooks must return without raising when _get_langfuse() is None."""
        for k in (
            "HERMES_LANGFUSE_PUBLIC_KEY", "HERMES_LANGFUSE_SECRET_KEY",
            "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
        ):
            monkeypatch.delenv(k, raising=False)

        sys.modules.pop("plugins.observability.langfuse", None)
        import importlib
        mod = importlib.import_module("plugins.observability.langfuse")

        # Each hook should just return; no exceptions.
        mod.on_pre_llm_call(task_id="t", session_id="s", messages=[{"role": "user", "content": "hi"}])
        mod.on_pre_llm_request(task_id="t", session_id="s", api_call_count=1, messages=[])
        mod.on_post_llm_call(task_id="t", session_id="s", api_call_count=1)
        mod.on_pre_tool_call(tool_name="read_file", args={}, task_id="t", session_id="s")
        mod.on_post_tool_call(tool_name="read_file", args={}, result="ok", task_id="t", session_id="s")


class TestProviderQualifiedModel:
    """Cost-routing prefix applied at the Langfuse trace boundary only."""

    def _q(self):
        sys.modules.pop("plugins.observability.langfuse", None)
        return importlib.import_module("plugins.observability.langfuse")._provider_qualified_model

    def test_normalizes_opencode_go_alias_to_opencode_zen(self):
        assert self._q()("opencode-go", "kimi-k2.6") == "opencode-zen/kimi-k2.6"

    def test_passes_other_providers_through_unchanged(self):
        q = self._q()
        assert q("openrouter", "kimi-k2.6") == "openrouter/kimi-k2.6"
        assert q("openrouter", "moonshotai/kimi-k2.6-20260420") == "openrouter/moonshotai/kimi-k2.6-20260420"
        assert q("anthropic", "claude-opus-4-7") == "anthropic/claude-opus-4-7"

    def test_idempotent_when_already_prefixed(self):
        q = self._q()
        assert q("opencode-zen", "opencode-zen/kimi-k2.6") == "opencode-zen/kimi-k2.6"
        # Idempotent through the alias too.
        assert q("opencode-go", "opencode-zen/kimi-k2.6") == "opencode-zen/kimi-k2.6"

    def test_returns_input_unchanged_when_provider_or_model_empty(self):
        q = self._q()
        assert q("", "kimi-k2.6") == "kimi-k2.6"
        assert q("opencode-go", "") == ""


class TestUsageAndCostTotalKey:
    """Langfuse v3 reads dashboard rollup from cost_details['total'].

    Without a 'total' key, calculatedTotalCost stays 0 even when sub-buckets
    (input/output/cache_*) are populated.  Regression guard for the bug where
    the happy path (entry found) wrote per-type cost_details but never set
    total, leaving daily metrics totalCost at 0 for the entire project.
    """

    def test_happy_path_sets_total_key(self):
        from types import SimpleNamespace
        sys.modules.pop("plugins.observability.langfuse", None)
        mod = importlib.import_module("plugins.observability.langfuse")

        # OpenAI/opencode-zen shape: prompt_tokens INCLUDES cached tokens;
        # normalize_usage subtracts cached to derive the input bucket.  So
        # prompt_tokens=19793 (7505 fresh + 12288 cached).
        usage = SimpleNamespace(
            prompt_tokens=19793,
            completion_tokens=47,
            prompt_tokens_details=SimpleNamespace(cached_tokens=12288),
        )
        response = SimpleNamespace(usage=usage)
        _, cost_details = mod._usage_and_cost(
            response,
            provider="opencode-go",
            api_mode="chat_completions",
            model="kimi-k2.6",
            base_url="https://opencode.ai/zen/go/v1",
        )

        assert "total" in cost_details, "happy path must set cost_details['total'] for Langfuse rollup"
        assert cost_details["total"] > 0
        # Per-bucket sub-totals are still emitted alongside total.
        assert cost_details["input"] > 0
        assert cost_details["output"] > 0
        assert cost_details["cache_read_input_tokens"] > 0

    def test_post_api_request_dict_path_sets_total_key(self):
        """The post_api_request hook passes usage as a dict, not a response.

        Both cost paths in on_post_llm_call must set cost_details['total'] —
        patching only the response-object path leaves real Hermes traces with
        calculatedTotalCost=0 because production primarily uses the dict path.
        """
        sys.modules.pop("plugins.observability.langfuse", None)
        mod = importlib.import_module("plugins.observability.langfuse")

        captured = {}

        def stub_end(observation, *, output=None, metadata=None,
                     usage_details=None, cost_details=None):
            captured["cost_details"] = cost_details

        fake_gen = object()
        fake_state = mod.TraceState(
            trace_id="t", root_ctx=None, root_span=None,
            generations={mod._request_key(1): fake_gen},
        )
        mod._TRACE_STATE[mod._trace_key("t", "s")] = fake_state
        mod._end_observation = stub_end
        mod._get_langfuse = lambda: object()
        mod._finish_trace = lambda *a, **k: None

        mod.on_post_llm_call(
            task_id="t", session_id="s", api_call_count=1,
            provider="opencode-go", api_mode="chat_completions",
            model="kimi-k2.6", base_url="https://opencode.ai/zen/go/v1",
            usage={
                "input_tokens": 7505,
                "output_tokens": 47,
                "cache_read_tokens": 12288,
            },
            assistant_content_chars=10,
            assistant_tool_call_count=0,
        )
        cost_details = captured.get("cost_details") or {}
        assert "total" in cost_details, "dict-path must set cost_details['total']"
        assert cost_details["total"] > 0
        assert cost_details.get("input", 0) > 0
        assert cost_details.get("output", 0) > 0
        assert cost_details.get("cache_read_input_tokens", 0) > 0
