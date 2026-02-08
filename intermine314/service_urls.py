from __future__ import annotations

from urllib.parse import urlparse, urlunparse


SERVICE_ROOT_KEYS = (
    "webServiceRoot",
    "serviceRoot",
    "serviceUrl",
    "service_url",
    "service",
    "url",
)


def normalize_service_root(root):
    text = str(root or "").strip()
    if not text:
        raise ValueError("A service root URL is required")
    if "://" not in text:
        text = "http://" + text
    parsed = urlparse(text)
    path = (parsed.path or "").rstrip("/")
    if not path.endswith("/service"):
        path = path + "/service" if path else "/service"
    normalized = parsed._replace(path=path, params="", query="", fragment="")
    return urlunparse(normalized)


def service_root_from_payload(payload, *, normalize=False):
    for key in SERVICE_ROOT_KEYS:
        value = payload.get(key) if isinstance(payload, dict) else None
        if value:
            return normalize_service_root(value) if normalize else value
    raise KeyError("Could not resolve service root from registry payload")
