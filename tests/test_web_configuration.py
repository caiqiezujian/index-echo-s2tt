import json
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.request import urlopen

import pytest

from s2tt.core.session import SessionConfig
from s2tt.transports.webpage import browser_handler, browser_settings


def test_browser_reads_public_local_defaults_without_model_paths_or_secrets(monkeypatch):
    monkeypatch.setenv("S2TT_AUTH_TOKEN", "local-test-secret")
    settings = {"backend": {"model_dir": "/private/models"}, "server": {"port": 19001}, "web": {}}
    config = SessionConfig(glossary=("Echo:回声",))
    with ThreadingHTTPServer(("127.0.0.1", 0), browser_handler(settings, config)) as server:
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with urlopen(f"http://127.0.0.1:{server.server_port}/runtime-config.json", timeout=5) as response:
                payload = response.read().decode("utf-8")
                assert response.headers["Cache-Control"] == "no-store"
            defaults = json.loads(payload)
            assert defaults["websocket_url"] == "ws://127.0.0.1:19001"
            assert defaults["glossary"] == ["Echo → 回声"]
            assert (defaults["source_language"], defaults["target_language"]) == ("en", "zh")
            assert "local-test-secret" not in payload and "/private/models" not in payload
        finally:
            server.shutdown()
            worker.join(timeout=5)


def test_browser_can_use_explicit_ssh_forward_address():
    settings = {"server": {"port": 19001}, "web": {"websocket_url": "ws://127.0.0.1:8765"}}
    assert browser_settings(settings, SessionConfig())["websocket_url"] == "ws://127.0.0.1:8765"


@pytest.mark.parametrize("address", ["http://localhost:8765", "file:///models", "ws://user:secret@localhost:8765"])
def test_invalid_browser_connection_address_is_rejected(address):
    with pytest.raises(ValueError):
        browser_settings({"web": {"websocket_url": address}}, SessionConfig())
