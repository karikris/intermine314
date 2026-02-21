import logging

from intermine314.config.constants import DEFAULT_REQUEST_TIMEOUT_SECONDS, DEFAULT_REGISTRY_INSTANCES_URL
from intermine314.service import Registry, Service
from intermine314.util.logging import log_structured_event

"""
Functions for making use of registry data
================================================

"""

NO_SUCH_MINE = "No such mine available"
_REGISTRY_API_LOG = logging.getLogger("intermine314.registry.api")


def _transport_mode(proxy_url, tor):
    if bool(tor):
        return "tor"
    if proxy_url:
        return "proxy"
    return "direct"


def _log_legacy_api_usage(api_name, *, registry_url, proxy_url, tor):
    if not _REGISTRY_API_LOG.isEnabledFor(logging.INFO):
        return
    log_structured_event(
        _REGISTRY_API_LOG,
        logging.INFO,
        "legacy_registry_api_usage",
        api=api_name,
        registry_url=registry_url,
        transport_mode=_transport_mode(proxy_url, tor),
        tor_enabled=bool(tor),
        proxy_configured=bool(proxy_url),
    )


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
    try:
        _log_legacy_api_usage(
            "getVersion",
            registry_url=registry_url,
            proxy_url=proxy_url,
            tor=tor,
        )
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
        return {
            "API Version:": info.get("api_version"),
            "Release Version:": info.get("release_version"),
            "InterMine Version:": info.get("intermine_version"),
        }
    except Exception:
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
    try:
        _log_legacy_api_usage(
            "getInfo",
            registry_url=registry_url,
            proxy_url=proxy_url,
            tor=tor,
        )
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
        print("Description: " + (info.get("description") or ""))
        print("URL: " + (info.get("url") or ""))
        print("API Version: " + (info.get("api_version") or ""))
        print("Release Version: " + (info.get("release_version") or ""))
        print("InterMine Version: " + (info.get("intermine_version") or ""))
        print("Organisms: ")
        for organism in info.get("organisms", []):
            print(organism)
        print("Neighbours: ")
        for neighbour in info.get("neighbours", []):
            print(neighbour)
        return None
    except Exception:
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
    try:
        _log_legacy_api_usage(
            "getData",
            registry_url=registry_url,
            proxy_url=proxy_url,
            tor=tor,
        )
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
        service_root = info.get("url") or registry.service_root(mine)
        if not service_root:
            return NO_SUCH_MINE

        service = Service(
            service_root,
            request_timeout=request_timeout,
            proxy_url=proxy_url,
            session=session,
            verify_tls=verify_tls,
            tor=tor,
            allow_http_over_tor=allow_http_over_tor,
        )
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

        for name in sorted(set(dataset_names)):
            print("Name: " + name)
        return None
    except Exception:
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
    try:
        _log_legacy_api_usage(
            "getMines",
            registry_url=registry_url,
            proxy_url=proxy_url,
            tor=tor,
        )
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
    except Exception:
        return NO_SUCH_MINE

    names = sorted(set(m.get("name") for m in mines if isinstance(m, dict) and m.get("name")))
    if not names:
        return NO_SUCH_MINE
    for name in names:
        print(name)
    return None
