import logging
import warnings

from intermine314.config.constants import DEFAULT_REQUEST_TIMEOUT_SECONDS, DEFAULT_REGISTRY_INSTANCES_URL
from intermine314.service import Registry, Service
from intermine314.service.transport import is_tor_proxy_url, resolve_proxy_url
from intermine314.util.logging import log_structured_event

"""
Functions for making use of registry data
================================================

"""

NO_SUCH_MINE = "No such mine available"
_REGISTRY_API_LOG = logging.getLogger("intermine314.registry.api")
_LEGACY_API_DEPRECATION_EMITTED: set[str] = set()
_TOR_ENV_IMPLICIT_TRANSPORT_WARNED = False
_LEGACY_API_CALLS_BY_NAME: dict[str, int] = {}
_LEGACY_API_SUPPRESSED_BY_NAME: dict[str, int] = {}
_LEGACY_API_REPLACEMENTS = {
    "getVersion": "get_version",
    "getInfo": "get_info",
    "getData": "get_data",
    "getMines": "get_mines",
}
LEGACY_REGISTRY_API_DEPRECATION_STARTED_IN = "0.1.5"
LEGACY_REGISTRY_API_REMOVAL_NOT_BEFORE = "0.3.0"

__all__ = [
    "NO_SUCH_MINE",
    "RegistryAPIError",
    "RegistryLookupError",
    "RegistryQueryError",
    "get_version",
    "get_info",
    "get_data",
    "get_mines",
    "getVersion",
    "getInfo",
    "getData",
    "getMines",
    "legacy_registry_api_metrics",
    "legacy_registry_api_deprecation_status",
    "LEGACY_REGISTRY_API_DEPRECATION_STARTED_IN",
    "LEGACY_REGISTRY_API_REMOVAL_NOT_BEFORE",
]


class RegistryAPIError(RuntimeError):
    """Base exception for modern registry helper APIs."""


class RegistryLookupError(RegistryAPIError):
    """Raised when a registry lookup cannot resolve a mine/root."""


class RegistryQueryError(RegistryAPIError):
    """Raised when querying mine metadata fails."""


def _transport_mode(proxy_url, tor):
    if bool(tor):
        return "tor"
    if proxy_url:
        return "proxy"
    return "direct"


def _explicit_transport_supplied(*, proxy_url, session, tor):
    return proxy_url is not None or session is not None or bool(tor)


def _warn_legacy_api_deprecated_once(api_name):
    if api_name in _LEGACY_API_DEPRECATION_EMITTED:
        return
    _LEGACY_API_DEPRECATION_EMITTED.add(api_name)
    replacement = _LEGACY_API_REPLACEMENTS.get(api_name)
    replacement_note = (
        f" Prefer intermine314.registry.api.{replacement}() for structured return values."
        if replacement
        else ""
    )
    warnings.warn(
        f"intermine314.registry.api.{api_name} is deprecated; use Registry/Service APIs with explicit transport kwargs."
        f" Planned removal is not before {LEGACY_REGISTRY_API_REMOVAL_NOT_BEFORE}."
        f"{replacement_note}",
        DeprecationWarning,
        stacklevel=3,
    )


def _warn_tor_env_implicit_transport_once(*, api_name, explicit_transport):
    global _TOR_ENV_IMPLICIT_TRANSPORT_WARNED
    if explicit_transport or _TOR_ENV_IMPLICIT_TRANSPORT_WARNED:
        return
    if not is_tor_proxy_url(resolve_proxy_url(None)):
        return
    _TOR_ENV_IMPLICIT_TRANSPORT_WARNED = True
    warnings.warn(
        f"Legacy registry helper {api_name} was called without explicit transport while a Tor proxy environment is active. "
        "Pass proxy_url/session/tor explicitly to avoid transport ambiguity.",
        RuntimeWarning,
        stacklevel=3,
    )


def _log_legacy_api_usage(api_name, *, registry_url, proxy_url, tor, explicit_transport):
    if not _REGISTRY_API_LOG.isEnabledFor(logging.INFO):
        return
    env_proxy = resolve_proxy_url(None)
    log_structured_event(
        _REGISTRY_API_LOG,
        logging.INFO,
        "legacy_registry_api_usage",
        api=api_name,
        registry_url=registry_url,
        transport_mode=_transport_mode(proxy_url, tor),
        tor_enabled=bool(tor),
        proxy_configured=bool(proxy_url),
        explicit_transport=bool(explicit_transport),
        tor_env_proxy_active=bool(is_tor_proxy_url(env_proxy)),
    )


def _record_legacy_api_call(api_name):
    _LEGACY_API_CALLS_BY_NAME[api_name] = int(_LEGACY_API_CALLS_BY_NAME.get(api_name, 0)) + 1


def _log_legacy_exception_suppressed(
    api_name,
    *,
    exception,
    registry_url,
    proxy_url,
    tor,
    explicit_transport,
):
    if not _REGISTRY_API_LOG.isEnabledFor(logging.WARNING):
        return
    log_structured_event(
        _REGISTRY_API_LOG,
        logging.WARNING,
        "legacy_registry_api_exception_suppressed",
        api=api_name,
        registry_url=registry_url,
        transport_mode=_transport_mode(proxy_url, tor),
        tor_enabled=bool(tor),
        proxy_configured=bool(proxy_url),
        explicit_transport=bool(explicit_transport),
        exception_type=type(exception).__name__,
    )


def _record_legacy_exception_suppressed(
    api_name,
    *,
    exception,
    registry_url,
    proxy_url,
    tor,
    explicit_transport,
):
    _LEGACY_API_SUPPRESSED_BY_NAME[api_name] = int(_LEGACY_API_SUPPRESSED_BY_NAME.get(api_name, 0)) + 1
    _log_legacy_exception_suppressed(
        api_name,
        exception=exception,
        registry_url=registry_url,
        proxy_url=proxy_url,
        tor=tor,
        explicit_transport=explicit_transport,
    )


def _prepare_legacy_api_call(api_name, *, registry_url, proxy_url, session, tor):
    explicit_transport = _explicit_transport_supplied(proxy_url=proxy_url, session=session, tor=tor)
    _record_legacy_api_call(api_name)
    _warn_legacy_api_deprecated_once(api_name)
    _warn_tor_env_implicit_transport_once(api_name=api_name, explicit_transport=explicit_transport)
    _log_legacy_api_usage(
        api_name,
        registry_url=registry_url,
        proxy_url=proxy_url,
        tor=tor,
        explicit_transport=explicit_transport,
    )
    return explicit_transport


def _legacy_info_lines(info):
    lines = [
        f"Description: {info.get('description') or ''}",
        f"URL: {info.get('url') or ''}",
        f"API Version: {info.get('api_version') or ''}",
        f"Release Version: {info.get('release_version') or ''}",
        f"InterMine Version: {info.get('intermine_version') or ''}",
        "Organisms: ",
    ]
    lines.extend(str(organism) for organism in info.get("organisms", []))
    lines.append("Neighbours: ")
    lines.extend(str(neighbour) for neighbour in info.get("neighbours", []))
    return lines


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


def legacy_registry_api_metrics():
    return {
        "legacy_api_calls_total": int(sum(_LEGACY_API_CALLS_BY_NAME.values())),
        "legacy_api_calls_by_name": dict(_LEGACY_API_CALLS_BY_NAME),
        "legacy_api_suppressed_total": int(sum(_LEGACY_API_SUPPRESSED_BY_NAME.values())),
        "legacy_api_suppressed_by_name": dict(_LEGACY_API_SUPPRESSED_BY_NAME),
    }


def legacy_registry_api_deprecation_status():
    return {
        "started_in": LEGACY_REGISTRY_API_DEPRECATION_STARTED_IN,
        "removal_not_before": LEGACY_REGISTRY_API_REMOVAL_NOT_BEFORE,
        "legacy_wrappers": sorted(_LEGACY_API_REPLACEMENTS),
        "replacement_api": dict(_LEGACY_API_REPLACEMENTS),
    }


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


def getVersion(
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
    """
    A function to return the API version, release version and
    InterMine version numbers
    ================================================
    example:

        >>> from intermine314 import registry
        >>> registry.getVersion('flymine')
        >>> {'API Version:': '30', 'Release Version:': '48 2019 October',
        'InterMine Version:': '4.1.0'}

    """
    explicit_transport = _prepare_legacy_api_call(
        "getVersion",
        registry_url=registry_url,
        proxy_url=proxy_url,
        session=session,
        tor=tor,
    )
    try:
        version_info = get_version(
            mine,
            registry_url=registry_url,
            request_timeout=request_timeout,
            proxy_url=proxy_url,
            session=session,
            verify_tls=verify_tls,
            tor=tor,
            allow_http_over_tor=allow_http_over_tor,
        )
        return {
            "API Version:": version_info.get("api_version"),
            "Release Version:": version_info.get("release_version"),
            "InterMine Version:": version_info.get("intermine_version"),
        }
    except Exception as exc:
        _record_legacy_exception_suppressed(
            "getVersion",
            exception=exc,
            registry_url=registry_url,
            proxy_url=proxy_url,
            tor=tor,
            explicit_transport=explicit_transport,
        )
        return NO_SUCH_MINE


def getInfo(
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
    """
    A function to get information about a mine
    ================================================
    example:

        >>> from intermine314 import registry
        >>> registry.getInfo('flymine')
        Description:  An integrated database for Drosophila genomics
        URL: https://www.flymine.org/flymine
        API Version: 25
        Release Version: 45.1 2017 August
        InterMine Version: 1.8.5
        Organisms:
        D. melanogaster
        Neighbours:
        MODs

    """
    explicit_transport = _prepare_legacy_api_call(
        "getInfo",
        registry_url=registry_url,
        proxy_url=proxy_url,
        session=session,
        tor=tor,
    )
    try:
        info = get_info(
            mine,
            registry_url=registry_url,
            request_timeout=request_timeout,
            proxy_url=proxy_url,
            session=session,
            verify_tls=verify_tls,
            tor=tor,
            allow_http_over_tor=allow_http_over_tor,
        )
        for line in _legacy_info_lines(info):
            print(line)
        return None
    except Exception as exc:
        _record_legacy_exception_suppressed(
            "getInfo",
            exception=exc,
            registry_url=registry_url,
            proxy_url=proxy_url,
            tor=tor,
            explicit_transport=explicit_transport,
        )
        return NO_SUCH_MINE


def getData(
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
    """
    A function to get datasets corresponding to a mine
    ================================================
    example:

        >>> from intermine314 import registry
        >>> registry.getData('flymine')
        Name: Affymetrix array: Drosophila1
        Name: Affymetrix array: Drosophila2
        Name: Affymetrix array: GeneChip Drosophila Genome 2.0 Array
        Name: Affymetrix array: GeneChip Drosophila Genome Array
        Name: Anoph-Expr data set
        Name: BDGP cDNA clone data set.....


    """
    explicit_transport = _prepare_legacy_api_call(
        "getData",
        registry_url=registry_url,
        proxy_url=proxy_url,
        session=session,
        tor=tor,
    )
    try:
        dataset_names = get_data(
            mine,
            registry_url=registry_url,
            request_timeout=request_timeout,
            proxy_url=proxy_url,
            session=session,
            verify_tls=verify_tls,
            tor=tor,
            allow_http_over_tor=allow_http_over_tor,
        )
        for name in dataset_names:
            print("Name: " + name)
        return None
    except Exception as exc:
        _record_legacy_exception_suppressed(
            "getData",
            exception=exc,
            registry_url=registry_url,
            proxy_url=proxy_url,
            tor=tor,
            explicit_transport=explicit_transport,
        )
        return NO_SUCH_MINE


def getMines(
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
    """
    A function to get mines containing the organism
    ================================================
    example:

        >>> from intermine314 import registry
        >>> registry.getMines('D. melanogaster')
        FlyMine
        FlyMine Beta
        XenMine

    """
    explicit_transport = _prepare_legacy_api_call(
        "getMines",
        registry_url=registry_url,
        proxy_url=proxy_url,
        session=session,
        tor=tor,
    )
    try:
        names = get_mines(
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
        _record_legacy_exception_suppressed(
            "getMines",
            exception=exc,
            registry_url=registry_url,
            proxy_url=proxy_url,
            tor=tor,
            explicit_transport=explicit_transport,
        )
        return NO_SUCH_MINE

    if not names:
        return NO_SUCH_MINE
    for name in names:
        print(name)
    return None
