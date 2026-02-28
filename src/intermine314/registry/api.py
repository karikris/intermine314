from __future__ import annotations

from intermine314.config.constants import DEFAULT_REQUEST_TIMEOUT_SECONDS, DEFAULT_REGISTRY_INSTANCES_URL
from intermine314.service import Registry, Service

"""Functions for making use of registry data."""

NO_SUCH_MINE = "No such mine available"

__all__ = [
    "NO_SUCH_MINE",
    "RegistryAPIError",
    "RegistryLookupError",
    "RegistryQueryError",
    "get_version",
    "get_info",
    "get_data",
    "get_mines",
]


class RegistryAPIError(RuntimeError):
    """Base exception for modern registry helper APIs."""


class RegistryLookupError(RegistryAPIError):
    """Raised when a registry lookup cannot resolve a mine/root."""


class RegistryQueryError(RegistryAPIError):
    """Raised when querying mine metadata fails."""


def _safe_registry_info(
    mine,
    *,
    registry_url=DEFAULT_REGISTRY_INSTANCES_URL,
    request_timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    proxy_url=None,
    session=None,
    verify_tls=True,
    tor=False,
    allow_http_over_tor=False,
):
    registry = Registry(
        registry_url=registry_url,
        request_timeout=request_timeout,
        proxy_url=proxy_url,
        session=session,
        verify_tls=verify_tls,
        tor=tor,
        allow_http_over_tor=allow_http_over_tor,
    )
    info = registry.info(mine)
    return registry, info


def get_version(
    mine,
    *,
    registry_url=DEFAULT_REGISTRY_INSTANCES_URL,
    request_timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    proxy_url=None,
    session=None,
    verify_tls=True,
    tor=False,
    allow_http_over_tor=False,
):
    try:
        _, info = _safe_registry_info(
            mine,
            registry_url=registry_url,
            request_timeout=request_timeout,
            proxy_url=proxy_url,
            session=session,
            verify_tls=verify_tls,
            tor=tor,
            allow_http_over_tor=allow_http_over_tor,
        )
    except Exception as exc:
        raise RegistryLookupError(f"Failed to resolve registry version info for mine {mine!r}") from exc
    if not isinstance(info, dict):
        raise RegistryLookupError(f"Registry returned invalid info payload for mine {mine!r}")
    return {
        "api_version": info.get("api_version"),
        "release_version": info.get("release_version"),
        "intermine_version": info.get("intermine_version"),
    }


def get_info(
    mine,
    *,
    registry_url=DEFAULT_REGISTRY_INSTANCES_URL,
    request_timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    proxy_url=None,
    session=None,
    verify_tls=True,
    tor=False,
    allow_http_over_tor=False,
):
    try:
        _, info = _safe_registry_info(
            mine,
            registry_url=registry_url,
            request_timeout=request_timeout,
            proxy_url=proxy_url,
            session=session,
            verify_tls=verify_tls,
            tor=tor,
            allow_http_over_tor=allow_http_over_tor,
        )
    except Exception as exc:
        raise RegistryLookupError(f"Failed to resolve registry info for mine {mine!r}") from exc
    if not isinstance(info, dict):
        raise RegistryLookupError(f"Registry returned invalid info payload for mine {mine!r}")
    return dict(info)


def get_data(
    mine,
    *,
    registry_url=DEFAULT_REGISTRY_INSTANCES_URL,
    request_timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    proxy_url=None,
    session=None,
    verify_tls=True,
    tor=False,
    allow_http_over_tor=False,
):
    try:
        registry, info = _safe_registry_info(
            mine,
            registry_url=registry_url,
            request_timeout=request_timeout,
            proxy_url=proxy_url,
            session=session,
            verify_tls=verify_tls,
            tor=tor,
            allow_http_over_tor=allow_http_over_tor,
        )
    except Exception as exc:
        raise RegistryLookupError(f"Failed to resolve mine registry entry for {mine!r}") from exc
    if not isinstance(info, dict):
        raise RegistryLookupError(f"Registry returned invalid info payload for mine {mine!r}")

    service_root = info.get("url") or registry.service_root(mine)
    if not service_root:
        raise RegistryLookupError(f"No service root available for mine {mine!r}")
    try:
        service = Service(
            service_root,
            request_timeout=request_timeout,
            proxy_url=proxy_url,
            session=session,
            verify_tls=verify_tls,
            tor=tor,
            allow_http_over_tor=allow_http_over_tor,
        )
    except Exception as exc:
        raise RegistryQueryError(f"Failed to create service client for mine {mine!r}") from exc

    dataset_names = []
    query_shapes = (
        ("DataSet", ("DataSet.name", "DataSet.url"), ("DataSet.name", "name")),
        ("Dataset", ("Dataset.name", "Dataset.url"), ("Dataset.name", "name")),
    )
    for class_name, views, keys in query_shapes:
        try:
            query = service.new_query(class_name)
            query.add_view(*views)
            for row in query.rows(row="dict", start=0, size=500):
                value = None
                for key in keys:
                    if key in row and row[key]:
                        value = row[key]
                        break
                if value:
                    dataset_names.append(str(value))
            break
        except Exception:
            continue
    return sorted(set(dataset_names))


def get_mines(
    organism=None,
    *,
    registry_url=DEFAULT_REGISTRY_INSTANCES_URL,
    request_timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    proxy_url=None,
    session=None,
    verify_tls=True,
    tor=False,
    allow_http_over_tor=False,
):
    try:
        mines = Service.get_all_mines(
            organism=organism,
            registry_url=registry_url,
            request_timeout=request_timeout,
            proxy_url=proxy_url,
            session=session,
            verify_tls=verify_tls,
            tor=tor,
            allow_http_over_tor=allow_http_over_tor,
        )
    except Exception as exc:
        raise RegistryQueryError("Failed to resolve mine list from registry") from exc
    return sorted(set(m.get("name") for m in mines if isinstance(m, dict) and m.get("name")))
