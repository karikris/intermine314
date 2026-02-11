from __future__ import annotations

import os

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PROXY_URL_ENV_VAR = "INTERMINE314_PROXY_URL"


def resolve_proxy_url(proxy_url: str | None = None) -> str | None:
    if proxy_url is not None:
        value = str(proxy_url).strip()
        return value or None
    env_value = os.getenv(PROXY_URL_ENV_VAR, "").strip()
    return env_value or None


def build_session(*, proxy_url: str | None, user_agent: str | None = None) -> requests.Session:
    proxy_url = resolve_proxy_url(proxy_url)
    session = requests.Session()
    if proxy_url:
        session.proxies = {"http": proxy_url, "https": proxy_url}
        session.trust_env = False

    retries = Retry(
        total=5,
        backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST", "PUT", "DELETE"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=32, pool_maxsize=32)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    if user_agent:
        session.headers.update({"User-Agent": user_agent})
    return session
