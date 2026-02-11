# Tor and SOCKS5 Usage

`intermine314` now supports first-class SOCKS5 proxying through the shared HTTP
transport layer.

## Install SOCKS support

```bash
pip install "intermine314[proxy]"
```

## Quick start (Tor local SOCKS port)

Use `socks5h://` so DNS resolution also happens through Tor.

```python
from intermine314.webservice import Service

service = Service(
    "https://bar.utoronto.ca/thalemine/service",
    proxy_url="socks5h://127.0.0.1:9050",
    request_timeout=15,
)

print(service.version)
```

Convenience API:

```python
from intermine314.webservice import Service

service = Service.tor(
    "https://bar.utoronto.ca/thalemine/service",
    token="YOUR_TOKEN",
)

print(service.version)
```

## HTTPS safety policy in Tor mode

When Tor routing is enabled (`tor=True` or a Tor SOCKS proxy URL), `Service`
and `Registry` require `https://` endpoints by default.

To explicitly allow plaintext HTTP over Tor (not recommended), pass:

```python
service = Service(
    "http://example.org/service",
    proxy_url="socks5h://127.0.0.1:9050",
    allow_http_over_tor=True,
)
```

## Environment-based proxy

You can configure proxying globally with:

```bash
export INTERMINE314_PROXY_URL="socks5h://127.0.0.1:9050"
```

Then create `Service`/`Registry` objects normally.

## Timeout behavior

- `InterMineURLOpener.open()` now uses normalized connect/read timeouts.
- Default request timeout is applied consistently across session requests.
- `Service.model` fetches model XML through the opener transport path.

## Transport notes

- Session construction is centralized in `intermine314.service.transport.build_session`.
- Proxy settings are applied for both `http` and `https`.
- `trust_env` is disabled when an explicit proxy is configured.
- Retry policy handles transient HTTP failures (`429`, `500`, `502`, `503`, `504`).
