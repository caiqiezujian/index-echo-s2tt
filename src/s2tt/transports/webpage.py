"""Serve bundled browser assets and public defaults from the local configuration."""
from functools import partial
from http.server import SimpleHTTPRequestHandler
from importlib.resources import files
import json
from urllib.parse import urlsplit


def browser_settings(settings, config):
    address = settings.get("web", {}).get("websocket_url")
    if address is None:
        address = f"ws://127.0.0.1:{settings.get('server', {}).get('port', 8765)}"
    parsed = urlsplit(address)
    if parsed.scheme not in ("ws", "wss") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("web.websocket_url must be a ws:// or wss:// server address without credentials")
    return {"websocket_url": address, "glossary": list(config.glossary),
            "source_language": config.source_language, "target_language": config.target_language}


def browser_handler(settings, config):
    public = json.dumps(browser_settings(settings, config), ensure_ascii=False).encode("utf-8")

    class BrowserHandler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if urlsplit(self.path).path != "/runtime-config.json":
                return super().do_GET()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(public)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(public)

    return partial(BrowserHandler, directory=str(files("s2tt").joinpath("web")))
