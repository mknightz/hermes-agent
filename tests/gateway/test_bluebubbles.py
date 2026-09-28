"""Tests for the BlueBubbles iMessage gateway adapter."""
import asyncio
import json
import os
import shutil
from unittest.mock import AsyncMock

import httpx
import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter


def _make_adapter(monkeypatch, **extra):
    monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
    monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "secret")
    from gateway.platforms.bluebubbles import BlueBubblesAdapter

    cfg = PlatformConfig(
        enabled=True,
        extra={
            "server_url": "http://localhost:1234",
            "password": "secret",
            **extra,
        },
    )
    return BlueBubblesAdapter(cfg)


class TestBlueBubblesConfigLoading:
    def test_apply_env_overrides_bluebubbles(self, monkeypatch):
        monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
        monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "secret")
        monkeypatch.setenv("BLUEBUBBLES_WEBHOOK_PORT", "9999")
        monkeypatch.setenv("BLUEBUBBLES_REQUIRE_MENTION", "true")
        monkeypatch.setenv("BLUEBUBBLES_MENTION_PATTERNS", r'["(?i)^amos\\b"]')
        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.BLUEBUBBLES in config.platforms
        bc = config.platforms[Platform.BLUEBUBBLES]
        assert bc.enabled is True
        assert bc.extra["server_url"] == "http://localhost:1234"
        assert bc.extra["password"] == "secret"
        assert bc.extra["webhook_port"] == 9999
        assert bc.extra["require_mention"] is True
        assert bc.extra["mention_patterns"] == ["(?i)^amos\\b"]


class TestBlueBubblesHelpers:


    def test_format_message_preserves_underscores_in_identifiers(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        text = "Use /api_v2 with FEATURE_FLAG_NAME and config_file.json"
        assert adapter.format_message(text) == text

    def test_strip_markdown_headers(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        assert adapter.format_message("## Heading\ntext") == "Heading\ntext"


    def test_init_normalizes_webhook_path(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, webhook_path="bluebubbles-webhook")
        assert adapter.webhook_path == "/bluebubbles-webhook"


    def test_server_url_normalized(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, server_url="http://localhost:1234/")
        assert adapter.server_url == "http://localhost:1234"


class _FakeBlueBubblesRequest:
    def __init__(self, payload, password="secret"):
        self.query = {"password": password}
        self.headers = {}
        self._body = json.dumps(payload).encode("utf-8")

    async def read(self):
        return self._body


class TestBlueBubblesMentionGating:
    @pytest.mark.asyncio
    async def test_group_message_without_mention_is_acknowledged_and_skipped(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch,
            require_mention=True,
            send_read_receipts=False,
        )
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        response = await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "new-message",
            "data": {
                "guid": "msg-1",
                "text": "casual family chatter",
                "handle": {"address": "+15555550100"},
                "isFromMe": False,
                "isGroup": True,
                "chats": [{"guid": "iMessage;+;group-chat"}],
            },
        }))
        await asyncio.sleep(0)

        assert response.status == 200
        assert handled == []


class TestBlueBubblesWebhookParsing:

    def test_webhook_can_fall_back_to_sender_when_chat_fields_missing(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        payload = {
            "data": {
                "guid": "MESSAGE-GUID",
                "text": "hello",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
            }
        }
        record = adapter._extract_payload_record(payload) or {}
        chat_guid = adapter._value(
            record.get("chatGuid"),
            payload.get("chatGuid"),
            record.get("chat_guid"),
            payload.get("chat_guid"),
            payload.get("guid"),
        )
        chat_identifier = adapter._value(
            record.get("chatIdentifier"),
            record.get("identifier"),
            payload.get("chatIdentifier"),
            payload.get("identifier"),
        )
        sender = (
            adapter._value(
                record.get("handle", {}).get("address")
                if isinstance(record.get("handle"), dict)
                else None,
                record.get("sender"),
                record.get("from"),
                record.get("address"),
            )
            or chat_identifier
            or chat_guid
        )
        if not (chat_guid or chat_identifier) and sender:
            chat_identifier = sender
        assert chat_identifier == "user@example.com"


    def test_extract_payload_record_accepts_list_data(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        payload = {
            "type": "new-message",
            "data": [
                {
                    "text": "hello",
                    "chatGuid": "iMessage;-;user@example.com",
                    "chatIdentifier": "user@example.com",
                }
            ],
        }
        record = adapter._extract_payload_record(payload)
        assert record == payload["data"][0]


class TestBlueBubblesGuidResolution:


    @pytest.mark.asyncio
    async def test_participant_only_match_does_not_resolve_to_group(self, monkeypatch):
        """Regression for #24157: contact appearing as a participant in a group
        chat must NOT be selected when no DM with that exact chatIdentifier exists.

        Otherwise an outbound DM reply leaks into the group thread.
        """
        adapter = _make_adapter(monkeypatch)

        async def fake_api_post(path, payload):
            return {
                "data": [
                    {
                        "guid": "iMessage;+;chat0000000000-family-group",
                        "chatIdentifier": "chat0000000000",
                        "participants": [
                            {"address": "user@example.com"},
                            {"address": "+15555550100"},
                        ],
                    }
                ]
            }

        monkeypatch.setattr(adapter, "_api_post", fake_api_post)
        result = await adapter._resolve_chat_guid("user@example.com")
        assert result is None, (
            "participant-only match must not resolve to a group GUID — DM "
            "replies would leak into the group thread"
        )


    @pytest.mark.asyncio
    async def test_unresolved_target_is_not_cached(self, monkeypatch):
        """When no exact match is found, the resolver must NOT cache anything.

        Otherwise a later attempt — after the DM has been created — would
        keep returning the stale ``None`` from cache. Also guards against a
        latent variant of #24157 where a group GUID could be cached under a
        bare address key and persist across calls.
        """
        adapter = _make_adapter(monkeypatch)

        async def fake_api_post(path, payload):
            return {
                "data": [
                    {
                        "guid": "iMessage;+;chat0000000000-family-group",
                        "chatIdentifier": "chat0000000000",
                        "participants": [{"address": "user@example.com"}],
                    }
                ]
            }

        monkeypatch.setattr(adapter, "_api_post", fake_api_post)
        await adapter._resolve_chat_guid("user@example.com")
        assert "user@example.com" not in adapter._guid_cache


class TestBlueBubblesAttachmentDownload:
    """Verify _download_attachment routes to the correct cache helper."""

    def test_download_image_uses_image_cache(self, monkeypatch):
        """Image MIME routes to cache_image_from_bytes."""
        adapter = _make_adapter(monkeypatch)
        import asyncio

        # Mock the HTTP client response
        class MockResponse:
            status_code = 200
            content = b"\x89PNG\r\n\x1a\n"

            def raise_for_status(self):
                pass

        async def mock_get(*args, **kwargs):
            return MockResponse()

        adapter.client = type("MockClient", (), {"get": mock_get})()

        cached_path = None

        async def mock_cache_image(data, ext):
            nonlocal cached_path
            cached_path = f"/tmp/test_image{ext}"
            return cached_path

        monkeypatch.setattr(
            "gateway.platforms.bluebubbles.cache_image_from_bytes_async",
            mock_cache_image,
        )

        att_meta = {"mimeType": "image/png", "transferName": "photo.png"}
        result = asyncio.get_event_loop().run_until_complete(
            adapter._download_attachment("att-guid-123", att_meta)
        )
        assert result == "/tmp/test_image.png"


class TestBlueBubblesAttachmentSend:
    @pytest.mark.asyncio
    async def test_attachment_payload_is_read_before_async_upload(self, monkeypatch, tmp_path):
        adapter = _make_adapter(monkeypatch)
        file_path = tmp_path / "payload.bin"
        payload = b"attachment-payload"
        file_path.write_bytes(payload)

        captured = {}

        async def fake_resolve_chat_guid(chat_id):
            return "iMessage;+;chat-guid"

        class MockResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"status": 200, "data": {"guid": "message-guid"}}

        class MockClient:
            async def post(self, url, *, files, data, timeout):
                captured.update(url=url, files=files, data=data, timeout=timeout)
                return MockResponse()

        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        adapter.client = MockClient()

        result = await adapter._send_attachment(
            "target", str(file_path), filename="payload.bin"
        )

        assert result.success is True
        assert captured["files"]["attachment"] == (
            "payload.bin",
            payload,
            "application/octet-stream",
        )
        assert captured["data"]["chatGuid"] == "iMessage;+;chat-guid"


class _FakeFfmpegProc:
    """Stand-in for an asyncio subprocess running ffmpeg."""

    def __init__(self, returncode=0, out_path=None, out_bytes=b"caff\x00\x01fake", hang=False):
        self.returncode = None
        self._rc = returncode
        self._out_path = out_path
        self._out_bytes = out_bytes
        self._hang = hang
        self.killed = False

    async def communicate(self):
        if self._hang:
            await asyncio.sleep(3600)
        if self._out_path is not None and self._out_bytes is not None:
            with open(self._out_path, "wb") as fh:
                fh.write(self._out_bytes)
        self.returncode = self._rc
        return b"", b"ffmpeg stderr"

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        return self.returncode


class TestBlueBubblesVoiceCafConversion:
    """Outbound voice notes must be CAF: IMCore shows a non-CAF isAudioMessage as a 0s memo."""

    @pytest.fixture
    def setup(self, monkeypatch, tmp_path):
        import gateway.platforms.bluebubbles as bb

        adapter = _make_adapter(monkeypatch)
        captured = {}

        async def fake_resolve_chat_guid(chat_id):
            return "iMessage;+;chat-guid"

        class MockResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"status": 200, "data": {"guid": "message-guid"}}

        class MockClient:
            async def post(self, url, *, files, data, timeout):
                captured.update(url=url, files=files, data=data, timeout=timeout)
                return MockResponse()

        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        adapter.client = MockClient()

        work_dir = tmp_path / "caf-work"
        monkeypatch.setattr(bb.tempfile, "mkdtemp", lambda *a, **k: (work_dir.mkdir(), str(work_dir))[1])

        calls = []
        state = {"proc": None, "exc": None, "rc": 0, "out_bytes": b"caff\x00\x01fake", "hang": False}

        async def fake_exec(*args, **kwargs):
            calls.append(args)
            if state["exc"] is not None:
                raise state["exc"]
            proc = _FakeFfmpegProc(returncode=state["rc"], out_path=args[-1],
                                   out_bytes=state["out_bytes"], hang=state["hang"])
            state["proc"] = proc
            return proc

        monkeypatch.setattr(bb.asyncio, "create_subprocess_exec", fake_exec)
        src = tmp_path / "reply.mp3"
        src.write_bytes(b"ID3-mp3-bytes")
        return dict(bb=bb, adapter=adapter, captured=captured, calls=calls, state=state,
                    src=src, work_dir=work_dir, tmp_path=tmp_path)

    @pytest.mark.asyncio
    async def test_non_caf_audio_message_is_converted_with_libopus(self, setup):
        result = await setup["adapter"].send_voice("target", str(setup["src"]))

        assert result.success is True
        assert len(setup["calls"]) == 1
        argv = setup["calls"][0]
        assert argv[0] == "ffmpeg"
        assert str(setup["src"]) in argv
        assert argv[argv.index("-c:a") + 1] == "libopus"
        assert argv[argv.index("-ar") + 1] == "48000"
        assert argv[argv.index("-ac") + 1] == "1"
        out = argv[-1]
        assert out.endswith(".caf")
        # Output goes to a private temp dir, never next to the source file.
        assert os.path.dirname(out) == str(setup["work_dir"])
        assert setup["captured"]["data"]["isAudioMessage"] == "true"

    @pytest.mark.asyncio
    async def test_converted_voice_uses_caf_mime_and_filename(self, setup):
        await setup["adapter"].send_voice("target", str(setup["src"]))

        name, payload, mime = setup["captured"]["files"]["attachment"]
        assert name == "Voice Message.caf"
        assert mime == "audio/x-caf"
        assert payload == b"caff\x00\x01fake"
        assert setup["captured"]["data"]["name"] == "Voice Message.caf"

    @pytest.mark.asyncio
    async def test_caf_input_passes_through_unconverted(self, setup):
        caf = setup["tmp_path"] / "already.caf"
        caf.write_bytes(b"caff-original")

        result = await setup["adapter"].send_voice("target", str(caf))

        assert result.success is True
        assert setup["calls"] == []
        name, payload, mime = setup["captured"]["files"]["attachment"]
        assert (name, payload, mime) == ("already.caf", b"caff-original", "audio/x-caf")

    @pytest.mark.asyncio
    async def test_non_audio_attachment_untouched(self, setup):
        doc = setup["tmp_path"] / "notes.mp3"
        doc.write_bytes(b"plain-document")

        await setup["adapter"].send_document("target", str(doc))

        assert setup["calls"] == []
        assert setup["captured"]["files"]["attachment"] == (
            "notes.mp3", b"plain-document", "application/octet-stream")
        assert "isAudioMessage" not in setup["captured"]["data"]

    @pytest.mark.asyncio
    async def test_ffmpeg_nonzero_exit_falls_back_to_original(self, setup):
        # Non-zero exit must win even if ffmpeg left a non-empty (partial) file behind.
        setup["state"]["rc"] = 1

        result = await setup["adapter"].send_voice("target", str(setup["src"]))

        assert result.success is True
        assert setup["captured"]["files"]["attachment"] == (
            "reply.mp3", b"ID3-mp3-bytes", "application/octet-stream")
        assert not setup["work_dir"].exists()

    @pytest.mark.asyncio
    async def test_ffmpeg_empty_output_falls_back_to_original(self, setup):
        setup["state"]["out_bytes"] = b""

        await setup["adapter"].send_voice("target", str(setup["src"]))

        assert setup["captured"]["files"]["attachment"][0] == "reply.mp3"
        assert not setup["work_dir"].exists()

    @pytest.mark.asyncio
    async def test_ffmpeg_missing_falls_back_and_warns_once(self, setup, caplog):
        setup["state"]["exc"] = FileNotFoundError("ffmpeg")
        adapter = setup["adapter"]

        with caplog.at_level("WARNING", logger="gateway.platforms.bluebubbles"):
            await adapter.send_voice("target", str(setup["src"]))
            first = setup["captured"]["files"]["attachment"]
            await adapter.send_voice("target", str(setup["src"]))

        assert first == ("reply.mp3", b"ID3-mp3-bytes", "application/octet-stream")
        assert setup["captured"]["files"]["attachment"] == first
        warnings = [r for r in caplog.records if "ffmpeg" in r.getMessage().lower()]
        assert len(warnings) == 1
        assert not setup["work_dir"].exists()

    @pytest.mark.asyncio
    async def test_ffmpeg_timeout_kills_process_and_falls_back(self, setup, monkeypatch):
        monkeypatch.setattr(setup["bb"], "_CAF_CONVERT_TIMEOUT_SECONDS", 0.05)
        setup["state"]["hang"] = True

        result = await setup["adapter"].send_voice("target", str(setup["src"]))

        assert result.success is True
        assert setup["state"]["proc"].killed is True
        assert setup["captured"]["files"]["attachment"][0] == "reply.mp3"
        assert not setup["work_dir"].exists()

    @pytest.mark.asyncio
    async def test_temp_caf_removed_after_successful_send(self, setup):
        await setup["adapter"].send_voice("target", str(setup["src"]))

        assert not setup["work_dir"].exists()
        assert setup["src"].exists()  # the caller's file is never deleted

    @pytest.mark.asyncio
    async def test_temp_caf_removed_when_upload_raises(self, setup):
        class BoomClient:
            async def post(self, *a, **k):
                raise httpx.ConnectError("down")

        setup["adapter"].client = BoomClient()

        result = await setup["adapter"].send_voice("target", str(setup["src"]))

        assert result.success is False
        assert not setup["work_dir"].exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
@pytest.mark.asyncio
async def test_real_ffmpeg_mp3_to_caf(tmp_path, monkeypatch):
    import subprocess

    src = tmp_path / "tone.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                    str(src)], check=True, stdin=subprocess.DEVNULL, timeout=60)
    adapter = _make_adapter(monkeypatch)

    converted = await adapter._convert_to_caf(str(src))

    assert converted is not None
    work_dir, caf_path = converted
    try:
        with open(caf_path, "rb") as fh:
            assert fh.read(4) == b"caff"
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Webhook registration
# ---------------------------------------------------------------------------


class TestBlueBubblesWebhookUrl:
    """_webhook_url property normalises local hosts to 'localhost'."""

    def test_default_host(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        # Default webhook_host is 0.0.0.0 → normalized to localhost
        assert "localhost" in adapter._webhook_url
        assert str(adapter.webhook_port) in adapter._webhook_url
        assert adapter.webhook_path in adapter._webhook_url


    def test_register_url_omits_query_when_no_password(self, monkeypatch):
        """If no password is configured, the register URL should be the bare URL."""
        monkeypatch.delenv("BLUEBUBBLES_PASSWORD", raising=False)
        from gateway.platforms.bluebubbles import BlueBubblesAdapter
        cfg = PlatformConfig(
            enabled=True,
            extra={"server_url": "http://localhost:1234", "password": ""},
        )
        adapter = BlueBubblesAdapter(cfg)
        assert adapter._webhook_register_url == adapter._webhook_url


class TestBlueBubblesWebhookRegistration:
    """Tests for _register_webhook, _unregister_webhook, _find_registered_webhooks."""

    @staticmethod
    def _mock_client(get_response=None, post_response=None, delete_ok=True):
        """Build a tiny mock httpx.AsyncClient."""

        async def mock_get(*args, **kwargs):
            class R:
                status_code = 200
                def raise_for_status(self):
                    pass
                def json(self):
                    return get_response or {"status": 200, "data": []}
            return R()

        async def mock_post(*args, **kwargs):
            class R:
                status_code = 200
                def raise_for_status(self):
                    pass
                def json(self):
                    return post_response or {"status": 200, "data": {}}
            return R()

        async def mock_delete(*args, **kwargs):
            class R:
                status_code = 200 if delete_ok else 500
                def raise_for_status(self_inner):
                    if not delete_ok:
                        raise Exception("delete failed")
            return R()

        return type(
            "MockClient", (),
            {"get": mock_get, "post": mock_post, "delete": mock_delete},
        )()

    # -- _find_registered_webhooks --

    def test_find_registered_webhooks_returns_matches(self, monkeypatch):
        import asyncio
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_url
        adapter.client = self._mock_client(
            get_response={"status": 200, "data": [
                {"id": 1, "url": url, "events": ["new-message"]},
                {"id": 2, "url": "http://other:9999/hook", "events": ["message"]},
            ]}
        )
        result = asyncio.get_event_loop().run_until_complete(
            adapter._find_registered_webhooks(url)
        )
        assert len(result) == 1
        assert result[0]["id"] == 1


    # -- _register_webhook --

    def test_register_fresh(self, monkeypatch):
        """No existing webhook → POST creates one."""
        import asyncio
        adapter = _make_adapter(monkeypatch)
        adapter.client = self._mock_client(
            get_response={"status": 200, "data": []},
            post_response={"status": 200, "data": {"id": 42}},
        )
        ok = asyncio.get_event_loop().run_until_complete(
            adapter._register_webhook()
        )
        assert ok is True


    def test_register_reuses_existing(self, monkeypatch):
        """Crash resilience — existing registration is reused, no POST needed."""
        import asyncio
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        adapter.client = self._mock_client(
            get_response={"status": 200, "data": [
                {"id": 7, "url": url, "events": ["new-message"]},
            ]},
        )

        # Track whether POST was called
        post_called = False
        orig_api_post = adapter._api_post
        async def tracking_post(path, payload):
            nonlocal post_called
            post_called = True
            return await orig_api_post(path, payload)
        adapter._api_post = tracking_post

        ok = asyncio.get_event_loop().run_until_complete(
            adapter._register_webhook()
        )
        assert ok is True
        assert not post_called, "Should reuse existing, not POST again"


    # -- _unregister_webhook --


    def test_unregister_removes_all_duplicates(self, monkeypatch):
        """Multiple orphaned registrations for same URL — all get removed."""
        import asyncio
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        deleted_ids = []

        async def mock_delete(*args, **kwargs):
            # Extract ID from URL
            url_str = args[0] if args else ""
            deleted_ids.append(url_str)
            class R:
                status_code = 200
                def raise_for_status(self):
                    pass
            return R()

        adapter.client = self._mock_client(
            get_response={"status": 200, "data": [
                {"id": 1, "url": url},
                {"id": 2, "url": url},
                {"id": 3, "url": "http://other/hook"},
            ]},
        )
        adapter.client.delete = mock_delete

        ok = asyncio.get_event_loop().run_until_complete(
            adapter._unregister_webhook()
        )
        assert ok is True
        assert len(deleted_ids) == 2


# ---------------------------------------------------------------------------
# Regression for #78183: httpx timeout exceptions stringify to "" which
# defeats _is_timeout_error, causing the plain-text fallback to re-send an
# already-delivered message (duplicate delivery).
# ---------------------------------------------------------------------------

class TestBlueBubblesTimeoutErrorNormalization:
    """When an httpx timeout has an empty string representation, the adapter
    must fall back to the exception type name so the base-layer timeout guard
    can still recognise it."""

    @pytest.mark.asyncio
    async def test_send_read_timeout_produces_matchable_error(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)

        async def fake_resolve(chat_id):
            return "iMessage;+;chat-123"
        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve)

        async def fake_api_post(path, payload):
            raise httpx.ReadTimeout("")
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)

        result = await adapter.send("chat-1", "hello world")

        assert not result.success
        assert result.error, "error must not be empty"
        assert BasePlatformAdapter._is_timeout_error(result.error), (
            f"_is_timeout_error must recognise {result.error!r}"
        )


    @pytest.mark.asyncio
    async def test_create_chat_for_handle_timeout_produces_matchable_error(
        self, monkeypatch,
    ):
        """Sibling call path — _create_chat_for_handle has the same
        error=str(exc) pattern and must also preserve the exception type."""
        adapter = _make_adapter(monkeypatch)

        async def fake_api_post(path, payload):
            raise httpx.ReadTimeout("")
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)

        result = await adapter._create_chat_for_handle("test@example.com", "hi")

        assert not result.success
        assert result.error
        assert BasePlatformAdapter._is_timeout_error(result.error)

    @pytest.mark.asyncio
    async def test_non_empty_error_string_is_unchanged(self, monkeypatch):
        """A normal exception with a message must keep its original text."""
        adapter = _make_adapter(monkeypatch)

        async def fake_resolve(chat_id):
            return "iMessage;+;chat-123"
        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve)

        async def fake_api_post(path, payload):
            raise RuntimeError("Server error '500 Internal Server Error'")
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)

        result = await adapter.send("chat-1", "hello world")

        assert not result.success
        assert "500 Internal Server Error" in (result.error or "")




class TestBlueBubblesGateBeforeDownload:
    """The require_mention gate must run BEFORE attachments are downloaded (review follow-up)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("text, downloads, handled_count", [
        ("look at this", 0, 0),          # unmentioned group attachment: never fetched
        ("hermes look at this", 1, 1),   # mentioned: fetched and dispatched
    ])
    async def test_unmentioned_group_attachment_is_not_downloaded(
            self, monkeypatch, text, downloads, handled_count):
        adapter = _make_adapter(monkeypatch, require_mention=True, send_read_receipts=False)
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        download = AsyncMock(return_value="/tmp/cached.jpg")
        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachment", download)
        response = await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "new-message",
            "data": {
                "guid": "msg-att-1",
                "text": text,
                "handle": {"address": "+15555550100"},
                "isFromMe": False,
                "isGroup": True,
                "chats": [{"guid": "iMessage;+;group-chat"}],
                "attachments": [{"guid": "att-1", "mimeType": "image/jpeg"}],
            },
        }))
        await asyncio.sleep(0)

        assert response.status == 200
        assert download.await_count == downloads
        assert len(handled) == handled_count
