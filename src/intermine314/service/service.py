from __future__ import annotations

from collections import OrderedDict
from collections.abc import MutableMapping as DictMixin
from contextlib import closing
import json
import logging
from urllib.parse import urlparse

from intermine314.config.runtime_defaults import get_runtime_defaults
from intermine314.model import Attribute, Column, Model, Reference
from intermine314.registry.mines import resolve_mine_user_agent
from intermine314.service.errors import ServiceError, WebserviceError
from intermine314.service.resource_utils import (
    close_resource_quietly as _close_resource_quietly,
    resolve_verify_tls as _resolve_verify_tls,
)
from intermine314.service.session import InterMineURLOpener, ResultIterator
from intermine314.service.transport import (
    enforce_tor_dns_safe_proxy_url,
    is_tor_proxy_url,
    resolve_proxy_url,
)
from intermine314.service.urls import normalize_service_root, service_root_from_payload
from intermine314.util.logging import log_structured_event

_QUERY_CLASS = None
_REGISTRY_TRANSPORT_LOG = logging.getLogger("intermine314.registry.transport")


def _runtime_defaults():
    return get_runtime_defaults()


def _runtime_default_registry_instances_url() -> str:
    return str(_runtime_defaults().service_defaults.default_registry_instances_url)


def _runtime_default_registry_service_cache_size() -> int:
    return int(_runtime_defaults().registry_defaults.default_registry_service_cache_size)


def _runtime_default_request_timeout_seconds() -> int:
    return int(_runtime_defaults().service_defaults.default_request_timeout_seconds)


def _transport_mode(proxy_url, tor):
    if bool(tor):
        return "tor"
    if proxy_url:
        return "proxy"
    return "direct"


def _verify_tls_mode(verify_tls):
    if isinstance(verify_tls, bool):
        return "enabled" if verify_tls else "disabled"
    return "custom_ca"


def _resolve_service_user_agent(root, user_agent):
    if user_agent is not None:
        text = str(user_agent).strip()
        return text or None
    return resolve_mine_user_agent(root)


def _log_registry_transport_event(event, **fields):
    if not _REGISTRY_TRANSPORT_LOG.isEnabledFor(logging.DEBUG):
        return
    log_structured_event(_REGISTRY_TRANSPORT_LOG, logging.DEBUG, event, **fields)


def _validate_positive_int(value, name):
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_https_when_tor(url, *, tor_enabled, allow_http_over_tor, context):
    if not tor_enabled or allow_http_over_tor:
        return
    scheme = (urlparse(str(url)).scheme or "").lower()
    if scheme != "https":
        raise ValueError(
            f"{context} must use https:// when Tor routing is enabled. "
            f"Got: {url!r}. Set allow_http_over_tor=True to opt in explicitly."
        )


def _query_class():
    global _QUERY_CLASS
    if _QUERY_CLASS is None:
        from intermine314.query import Query

        _QUERY_CLASS = Query
    return _QUERY_CLASS


class Registry(DictMixin):
    """Registry client that discovers mine service roots and caches Service clients."""

    MINES_PATH = "/mines.json"
    INSTANCES_PATH = "/service/instances"
    _DEFAULT_REGISTRY_URL = None
    _MAX_CACHED_SERVICES = None

    def __init__(
        self,
        registry_url=None,
        request_timeout=None,
        proxy_url=None,
        session=None,
        verify_tls=True,
        tor=False,
        strict_tor_proxy_scheme=True,
        allow_insecure_tor_proxy_scheme=False,
        allow_http_over_tor=False,
        max_cached_services=None,
        user_agent=None,
    ):
        if registry_url is None:
            registry_url = _runtime_default_registry_instances_url()
        if request_timeout is None:
            request_timeout = _runtime_default_request_timeout_seconds()
        self.registry_url = registry_url.rstrip("/")
        self.request_timeout = request_timeout
        resolved_proxy_url = resolve_proxy_url(proxy_url)
        self.tor = bool(tor) or is_tor_proxy_url(resolved_proxy_url)
        self.strict_tor_proxy_scheme = bool(strict_tor_proxy_scheme)
        self.allow_insecure_tor_proxy_scheme = bool(allow_insecure_tor_proxy_scheme)
        self.proxy_url = enforce_tor_dns_safe_proxy_url(
            resolved_proxy_url,
            tor_mode=self.tor,
            context="Registry proxy_url",
            strict_tor_proxy_scheme=self.strict_tor_proxy_scheme,
            allow_insecure_tor_proxy_scheme=self.allow_insecure_tor_proxy_scheme,
        )
        self.verify_tls = _resolve_verify_tls(verify_tls)
        self.user_agent = _resolve_service_user_agent(self.registry_url, user_agent)
        self.allow_http_over_tor = bool(allow_http_over_tor)
        _log_registry_transport_event(
            "registry_transport_init",
            transport_mode=_transport_mode(self.proxy_url, self.tor),
            tor_enabled=bool(self.tor),
            proxy_configured=bool(self.proxy_url),
            verify_tls_mode=_verify_tls_mode(self.verify_tls),
            verify_tls_custom_ca=not isinstance(self.verify_tls, bool),
        )
        _require_https_when_tor(
            self.registry_url,
            tor_enabled=self.tor,
            allow_http_over_tor=self.allow_http_over_tor,
            context="Registry URL",
        )
        self._session = session
        self._opener = InterMineURLOpener(
            request_timeout=self.request_timeout,
            proxy_url=self.proxy_url,
            session=session,
            verify_tls=self.verify_tls,
            tor_mode=self.tor,
            strict_tor_proxy_scheme=self.strict_tor_proxy_scheme,
            allow_insecure_tor_proxy_scheme=self.allow_insecure_tor_proxy_scheme,
            user_agent=self.user_agent,
        )
        self._session = self._opener._session
        self._owns_session = bool(getattr(self._opener, "_owns_session", False))
        self._closed = False
        with closing(self._opener.open(self._list_url())) as registry_resp:
            data = registry_resp.read()
        mine_data = json.loads(ensure_str(data))
        mines = self._extract_mines(mine_data)
        self.__mine_dict = dict(((mine["name"], mine) for mine in mines))
        self.__synonyms = dict(((name.lower(), name) for name in list(self.__mine_dict.keys())))
        default_cache_size = (
            self._MAX_CACHED_SERVICES
            if self._MAX_CACHED_SERVICES is not None
            else _runtime_default_registry_service_cache_size()
        )
        raw_max_cached_services = default_cache_size if max_cached_services is None else max_cached_services
        _validate_positive_int(raw_max_cached_services, "max_cached_services")
        self._max_cached_services = int(raw_max_cached_services)
        self.__mine_cache = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0
        self._cache_clears = 0
        self._cache_closed_services = 0
        self._log_cache_event("registry_cache_initialized", mine_count=len(self.__mine_dict))

    def _adopt_session_ownership(self):
        if getattr(self, "_opener", None) is not None:
            adopt = getattr(self._opener, "adopt_session_ownership", None)
            if callable(adopt):
                adopt()
            self._session = self._opener._session
            self._owns_session = bool(getattr(self._opener, "_owns_session", False))

    def clear_cache(self, *, close_services=True):
        cache = getattr(self, "_Registry__mine_cache", None)
        if not isinstance(cache, dict):
            return 0

        cached_services = list(cache.values()) if close_services else []
        cleared_count = len(cache)
        cache.clear()

        closed = 0
        for service in cached_services:
            close_fn = getattr(service, "close", None)
            if not callable(close_fn):
                continue
            try:
                close_fn()
                closed += 1
            except Exception:
                continue

        self._cache_clears += 1
        self._cache_closed_services += closed
        self._log_cache_event(
            "registry_service_cache_cleared",
            cleared_count=cleared_count,
            closed_cached_services=closed,
            close_services=bool(close_services),
        )
        return cleared_count

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self.clear_cache(close_services=True)
        mine_dict = getattr(self, "_Registry__mine_dict", None)
        if isinstance(mine_dict, dict):
            mine_dict.clear()
        synonyms = getattr(self, "_Registry__synonyms", None)
        if isinstance(synonyms, dict):
            synonyms.clear()
        opener = getattr(self, "_opener", None)
        if opener is not None:
            _close_resource_quietly(opener)
        elif bool(getattr(self, "_owns_session", False)):
            _close_resource_quietly(getattr(self, "_session", None))
        self._session = None
        self._owns_session = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        _ = (exc_type, exc, tb)
        self.close()
        return False

    def __del__(self):  # pragma: no cover - non-deterministic GC timing
        try:
            self.close()
        except Exception:
            return

    def service_cache_metrics(self):
        cache_size = len(self.__mine_cache)
        max_size = int(self._max_cached_services)
        hits = int(self._cache_hits)
        misses = int(self._cache_misses)
        evictions = int(self._cache_evictions)
        clears = int(getattr(self, "_cache_clears", 0))
        closed_services = int(getattr(self, "_cache_closed_services", 0))
        return {
            "cache_size": cache_size,
            "max_cache_size": max_size,
            "cache_hits": hits,
            "cache_misses": misses,
            "cache_evictions": evictions,
            "cache_clears": clears,
            "cache_closed_services": closed_services,
            "registry_service_cache_size": cache_size,
            "registry_service_cache_max_size": max_size,
            "registry_service_cache_hits": hits,
            "registry_service_cache_misses": misses,
            "registry_service_cache_evictions": evictions,
            "registry_service_cache_clears": clears,
            "registry_service_cache_closed_services": closed_services,
        }

    def _log_cache_event(self, event, **fields):
        if not _REGISTRY_TRANSPORT_LOG.isEnabledFor(logging.DEBUG):
            return
        metrics = self.service_cache_metrics()
        metrics.update(fields)
        log_structured_event(_REGISTRY_TRANSPORT_LOG, logging.DEBUG, event, **metrics)

    def _list_url(self):
        if self.registry_url.endswith(self.MINES_PATH.rstrip("/")):
            return self.registry_url
        if self.registry_url.endswith(self.INSTANCES_PATH.rstrip("/")):
            return self.registry_url
        if self.registry_url.endswith("/registry"):
            return self.registry_url + self.MINES_PATH
        return self.registry_url + self.INSTANCES_PATH

    def _extract_mines(self, data):
        if "instances" in data:
            return data["instances"]
        if "mines" in data:
            return data["mines"]
        raise ServiceError("Registry response missing expected 'instances' or 'mines' data")

    def _service_root(self, mine):
        try:
            return service_root_from_payload(mine)
        except KeyError:
            raise KeyError("Missing service URL for mine")

    def __contains__(self, name):
        return name.lower() in self.__synonyms

    def __getitem__(self, name):
        lc = name.lower()
        if lc not in self.__synonyms:
            raise KeyError("Unknown mine: " + name)

        if lc in self.__mine_cache:
            self.__mine_cache.move_to_end(lc)
            self._cache_hits += 1
            self._log_cache_event("registry_service_cache_hit", mine=lc)
            return self.__mine_cache[lc]

        if len(self.__mine_cache) >= int(self._max_cached_services):
            evicted_mine, evicted_service = self.__mine_cache.popitem(last=False)
            closed_evicted = callable(getattr(evicted_service, "close", None))
            _close_resource_quietly(evicted_service)
            self._cache_evictions += 1
            if closed_evicted:
                self._cache_closed_services += 1
            self._log_cache_event("registry_service_cache_evict", mine=lc, evicted_mine=evicted_mine)

        mine = self.__mine_dict[self.__synonyms[lc]]
        self.__mine_cache[lc] = Service(
            self._service_root(mine),
            request_timeout=self.request_timeout,
            proxy_url=self.proxy_url,
            session=self._session,
            verify_tls=self.verify_tls,
            tor=self.tor,
            strict_tor_proxy_scheme=self.strict_tor_proxy_scheme,
            allow_insecure_tor_proxy_scheme=self.allow_insecure_tor_proxy_scheme,
            allow_http_over_tor=self.allow_http_over_tor,
            user_agent=self.user_agent,
        )
        self._cache_misses += 1
        self._log_cache_event("registry_service_cache_miss", mine=lc)
        return self.__mine_cache[lc]

    def __setitem__(self, name, item):
        raise NotImplementedError("You cannot add items to a registry")

    def __delitem__(self, name):
        raise NotImplementedError("You cannot remove items from a registry")

    def __len__(self):
        return len(self.__mine_dict)

    def __iter__(self):
        return iter(self.__mine_dict)

    def keys(self):
        return list(self.__mine_dict.keys())

    def info(self, name):
        """Return the registry info dictionary for a mine."""
        lc = name.lower()
        if lc in self.__synonyms:
            return self.__mine_dict[self.__synonyms[lc]]
        raise KeyError("Unknown mine: " + name)

    def service_root(self, name):
        """Return the service root URL for a mine."""
        return self._service_root(self.info(name))

    def all_mines(self, organism=None):
        """Return registry info dictionaries, optionally filtered by organism."""
        mines = list(self.__mine_dict.values())
        if organism is None:
            return mines
        target = organism.strip()
        filtered = []
        for mine in mines:
            organisms = mine.get("organisms") or []
            for entry in organisms:
                if entry.strip() == target:
                    filtered.append(mine)
                    break
        return filtered


def ensure_str(stringlike):
    if isinstance(stringlike, bytes):
        return stringlike.decode("utf-8")
    if isinstance(stringlike, str):
        return stringlike
    return str(stringlike)


class Service:
    """InterMine webservice client with query execution and transport lifecycle."""

    QUERY_PATH = "/query/results"
    MODEL_PATH = "/model"
    VERSION_PATH = "/version/ws"
    RELEASE_PATH = "/version/release"
    SCHEME = "http://"
    SERVICE_RESOLUTION_PATH = "/check/"

    def __init__(
        self,
        root,
        username=None,
        password=None,
        token=None,
        prefetch_depth=1,
        prefetch_id_only=False,
        request_timeout=None,
        proxy_url=None,
        session=None,
        verify_tls=True,
        tor=False,
        strict_tor_proxy_scheme=True,
        allow_insecure_tor_proxy_scheme=False,
        allow_http_over_tor=False,
        user_agent=None,
    ):
        root = normalize_service_root(root)
        if request_timeout is None:
            request_timeout = _runtime_default_request_timeout_seconds()

        self.root = root
        self.prefetch_depth = prefetch_depth
        self.prefetch_id_only = prefetch_id_only
        self.request_timeout = request_timeout
        resolved_proxy_url = resolve_proxy_url(proxy_url)
        self.tor = bool(tor) or is_tor_proxy_url(resolved_proxy_url)
        self.strict_tor_proxy_scheme = bool(strict_tor_proxy_scheme)
        self.allow_insecure_tor_proxy_scheme = bool(allow_insecure_tor_proxy_scheme)
        self.proxy_url = enforce_tor_dns_safe_proxy_url(
            resolved_proxy_url,
            tor_mode=self.tor,
            context="Service proxy_url",
            strict_tor_proxy_scheme=self.strict_tor_proxy_scheme,
            allow_insecure_tor_proxy_scheme=self.allow_insecure_tor_proxy_scheme,
        )
        self.verify_tls = _resolve_verify_tls(verify_tls)
        self.user_agent = _resolve_service_user_agent(root, user_agent)
        self.allow_http_over_tor = bool(allow_http_over_tor)
        _log_registry_transport_event(
            "service_transport_init",
            transport_mode=_transport_mode(self.proxy_url, self.tor),
            tor_enabled=bool(self.tor),
            proxy_configured=bool(self.proxy_url),
            verify_tls_mode=_verify_tls_mode(self.verify_tls),
            verify_tls_custom_ca=not isinstance(self.verify_tls, bool),
        )
        _require_https_when_tor(
            root,
            tor_enabled=self.tor,
            allow_http_over_tor=self.allow_http_over_tor,
            context="Service root URL",
        )

        self._model = None
        self._version = None
        self._release = None
        self._closed = False
        self._owns_session = False

        opener_session = session
        transfer_session_ownership = False
        if token:
            if token == "random":
                pre_auth_opener = InterMineURLOpener(
                    request_timeout=self.request_timeout,
                    proxy_url=self.proxy_url,
                    session=opener_session,
                    verify_tls=self.verify_tls,
                    tor_mode=self.tor,
                    strict_tor_proxy_scheme=self.strict_tor_proxy_scheme,
                    allow_insecure_tor_proxy_scheme=self.allow_insecure_tor_proxy_scheme,
                    user_agent=self.user_agent,
                )
                token = self.get_anonymous_token(url=root, opener=pre_auth_opener)
                opener_session = pre_auth_opener._session
                transfer_session_ownership = bool(getattr(pre_auth_opener, "_owns_session", False))
                if transfer_session_ownership and opener_session is not None:
                    release = getattr(pre_auth_opener, "_set_session", None)
                    if callable(release):
                        release(opener_session, owns_session=False)
            self.opener = InterMineURLOpener(
                token=token,
                request_timeout=self.request_timeout,
                proxy_url=self.proxy_url,
                session=opener_session,
                verify_tls=self.verify_tls,
                tor_mode=self.tor,
                strict_tor_proxy_scheme=self.strict_tor_proxy_scheme,
                allow_insecure_tor_proxy_scheme=self.allow_insecure_tor_proxy_scheme,
                user_agent=self.user_agent,
            )
            if transfer_session_ownership:
                adopt = getattr(self.opener, "adopt_session_ownership", None)
                if callable(adopt):
                    adopt()
        elif username:
            if token:
                raise ValueError("Both username and token credentials supplied")
            if not password:
                raise ValueError("Username given, but no password supplied")

            self.opener = InterMineURLOpener(
                (username, password),
                request_timeout=self.request_timeout,
                proxy_url=self.proxy_url,
                session=opener_session,
                verify_tls=self.verify_tls,
                tor_mode=self.tor,
                strict_tor_proxy_scheme=self.strict_tor_proxy_scheme,
                allow_insecure_tor_proxy_scheme=self.allow_insecure_tor_proxy_scheme,
                user_agent=self.user_agent,
            )
        else:
            self.opener = InterMineURLOpener(
                request_timeout=self.request_timeout,
                proxy_url=self.proxy_url,
                session=opener_session,
                verify_tls=self.verify_tls,
                tor_mode=self.tor,
                strict_tor_proxy_scheme=self.strict_tor_proxy_scheme,
                allow_insecure_tor_proxy_scheme=self.allow_insecure_tor_proxy_scheme,
                user_agent=self.user_agent,
            )

        self._owns_session = bool(getattr(self.opener, "_owns_session", False))

        try:
            self.version
        except WebserviceError as e:
            raise ServiceError(f"Could not validate service - is the root url ({root}) correct? {e}")

        if token and self.version < 6:
            raise ServiceError("This service does not support API access token authentication")

    def get_anonymous_token(self, url, opener=None):
        url += "/session"
        if opener is None:
            opener = self.opener if hasattr(self, "opener") else InterMineURLOpener(
                request_timeout=self.request_timeout,
                proxy_url=self.proxy_url,
                verify_tls=self.verify_tls,
                tor_mode=getattr(self, "tor", None),
                strict_tor_proxy_scheme=getattr(self, "strict_tor_proxy_scheme", True),
                allow_insecure_tor_proxy_scheme=getattr(self, "allow_insecure_tor_proxy_scheme", False),
                user_agent=getattr(self, "user_agent", None),
            )

        with closing(opener.open(url, method="GET", timeout=opener._timeout)) as token_resp:
            payload = ensure_str(token_resp.read())
        token = json.loads(payload)["token"]
        return token

    def _adopt_session_ownership(self):
        opener = getattr(self, "opener", None)
        adopt = getattr(opener, "adopt_session_ownership", None)
        if callable(adopt):
            adopt()
        self._owns_session = bool(getattr(opener, "_owns_session", False))

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        opener = getattr(self, "opener", None)
        if opener is not None:
            _close_resource_quietly(opener)
        self._owns_session = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        _ = (exc_type, exc, tb)
        self.close()
        return False

    def __del__(self):  # pragma: no cover - non-deterministic GC timing
        try:
            self.close()
        except Exception:
            return

    @property
    def version(self):
        """Return the webservice version as an integer."""
        try:
            if self._version is None:
                try:
                    url = self.root + self.VERSION_PATH
                    with closing(self.opener.open(url)) as version_resp:
                        self._version = int(version_resp.read())
                except ValueError as e:
                    raise ServiceError("Could not parse a valid webservice version: " + str(e))
        except AttributeError as e:
            raise Exception(e)
        return self._version

    def resolve_service_path(self, variant):
        """Resolve the path to optional services."""
        url = self.root + self.SERVICE_RESOLUTION_PATH + variant
        with closing(self.opener.open(url)) as variant_resp:
            return variant_resp.read()

    @property
    def release(self):
        """Return the datawarehouse release label."""
        if self._release is None:
            with closing(self.opener.open(self.root + self.RELEASE_PATH)) as resp:
                self._release = ensure_str(resp.read()).strip()
        return self._release

    def select(self, *columns):
        """Construct a new Query and optionally select output columns."""
        query = _query_class()(self.model, self)
        if len(columns) == 1:
            view = columns[0]
            if isinstance(view, Attribute):
                return query.select("%s.%s" % (view.declared_in.name, view))

            if isinstance(view, Reference):
                return query.select("%s.%s.*" % (view.declared_in.name, view))
            elif not isinstance(view, Column) and not str(view).endswith("*"):
                path = self.model.make_path(view)
                if not path.is_attribute():
                    return query.select(str(view) + ".*")

        return query.select(*columns)

    def flush(self):
        """Flush cached service metadata."""
        self._model = None
        self._version = None
        self._release = None

    @property
    def model(self):
        """Return the service data model, loading it lazily."""
        if self._model is None:
            model_xml = self.opener.read(self.root + self.MODEL_PATH)
            self._model = Model(model_xml, self)
        return self._model

    def get_results(self, path, params, rowformat, view, cld=None):
        """Return a result iterator for a query request."""
        return ResultIterator(self, path, params, rowformat, view, cld)
