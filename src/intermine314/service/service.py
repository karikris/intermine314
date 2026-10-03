from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict
from collections.abc import MutableMapping as DictMixin
from contextlib import closing
from types import SimpleNamespace
from urllib.parse import urlencode, urlparse
from xml.dom import minidom
from xml.etree import ElementTree as _ET

from intermine314.compatibility import class_name, resolve_compatibility
from intermine314.config.runtime_defaults import get_runtime_defaults
from intermine314.decorators import requires_version
from intermine314.service.errors import ServiceError, WebserviceError
from intermine314.service.resource_utils import (
    close_resource_quietly as _close_resource_quietly,
)
from intermine314.service.resource_utils import (
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
_DEFAULT_USER_AGENT = "intermine314/benchmark-runtime"


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
    _ = root
    if user_agent is not None:
        text = str(user_agent).strip()
        return text or None
    return _DEFAULT_USER_AGENT


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


def _normalize_query_columns(model, columns, root):
    """Resolve descriptor roots without losing mixed selections.

    Single Fields retain their declaring class, as in the 1.13.0 factory.
    Mixed inherited Fields use the most specific compatible selected root.
    String-only calls keep their existing optional-model validation policy.
    """
    def flatten(values):
        for value in values:
            if isinstance(value, (list, tuple, set)):
                yield from flatten(value)
            elif isinstance(value, str):
                yield from (token for token in re.split(r"(?:,?\s+|,)", value.strip()) if token)
            else:
                yield value

    values = list(flatten(columns))
    if all(isinstance(value, str) for value in values):
        return class_name(root), values, {}, False

    from intermine314.model import Class, Column, Field, ModelError, Path, Reference

    descriptors = any(isinstance(value, (Class, Field, Column, Path)) for value in values)
    has_model = callable(getattr(model, "make_path", None))
    selected_root = class_name(root)
    candidates = []
    if descriptors and not has_model:
        raise ModelError("Descriptor selections require a valid service model")
    if descriptors:
        for value in values:
            if isinstance(value, Field):
                candidates.append(value.declared_in.name)
            elif isinstance(value, Class):
                candidates.append(value.name)
            elif isinstance(value, (Column, Path)):
                candidates.append(value._path.root.name if isinstance(value, Column) else value.root.name)
            elif isinstance(value, str) and value.split('.', 1)[0] in model.classes:
                candidates.append(value.split('.', 1)[0])
        classes = [model.get_class(candidate) for candidate in dict.fromkeys(candidates)]
        possible_roots = [model.get_class(selected_root)] if root is not None else classes
        compatible = next((candidate for candidate in possible_roots
                           if all(candidate.isa(parent) for parent in classes)), None)
        if compatible is None:
            names = ([selected_root] if root is not None else []) + candidates
            raise ModelError(f"Incompatible selection roots: {', '.join(names)}")
        selected_root = compatible.name

    paths, subclasses = [], {}
    for value in values:
        if isinstance(value, Field):
            text = selected_root + '.' + value.name
            if isinstance(value, Reference):
                text += '.*'
        elif isinstance(value, Class):
            text = selected_root + '.*'
        else:
            text = str(value)
            if descriptors and text.split('.', 1)[0] in candidates:
                text = selected_root + text[len(text.split('.', 1)[0]):]
            if isinstance(value, Column):
                for path, subclass in value._subclasses.items():
                    path = selected_root + path[len(value._path.root.name):]
                    if path in subclasses and subclasses[path] != subclass:
                        raise ModelError(f"Conflicting subclass selections for {path}")
                    subclasses[path] = subclass
                if not value._path.is_attribute():
                    text += '.*'
            elif isinstance(value, Path) and not value.is_attribute():
                text += '.*'
        paths.append(text)
    return selected_root, paths, subclasses, descriptors


class Registry(DictMixin):
    """Registry client that discovers mine service roots and caches Service clients."""

    MINES_PATH = "/mines.json"
    INSTANCES_PATH = "/service/instances"
    _DEFAULT_REGISTRY_URL = None
    _MAX_CACHED_SERVICES = None
    _DEFAULT_COMPATIBILITY = "native"

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
        *,
        compatibility=None,
    ):
        self.compatibility = resolve_compatibility(compatibility, default=self._DEFAULT_COMPATIBILITY)
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
        self.__mine_dict = dict((mine["name"], mine) for mine in mines)
        self.__synonyms = dict((name.lower(), name) for name in list(self.__mine_dict.keys()))
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
        self.__mine_cache[lc] = self._new_service(
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
            compatibility=self.compatibility,
        )
        self._cache_misses += 1
        self._log_cache_event("registry_service_cache_miss", mine=lc)
        return self.__mine_cache[lc]

    def _new_service(self, root, **kwargs):
        if self.compatibility == "legacy":
            from intermine314.webservice import Service as service_class
        else:
            service_class = Service
        return service_class(root, **kwargs)

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
    TEMPLATEQUERY_PATH = "/template/results"
    TEMPLATES_PATH = "/templates"
    ALL_TEMPLATES_PATH = "/alltemplates"
    QUERY_LIST_UPLOAD_PATH = "/query/tolist"
    QUERY_LIST_APPEND_PATH = "/query/append/tolist"
    LIST_MANAGER_METHODS = frozenset([
        "get_list", "get_all_lists", "get_all_list_names", "create_list",
        "get_list_count", "delete_lists", "l",
    ])
    SEARCH_PATH = "/search"
    WIDGETS_PATH = "/widgets"
    IDS_PATH = "/ids"
    MODEL_PATH = "/model"
    VERSION_PATH = "/version/ws"
    RELEASE_PATH = "/version/release"
    USERS_PATH = "/users"
    LIST_PATH = "/lists"
    LIST_CREATION_PATH = "/lists"
    LIST_RENAME_PATH = "/lists/rename"
    LIST_APPENDING_PATH = "/lists/append"
    LIST_TAG_PATH = "/list/tags"
    LIST_ENRICHMENT_PATH = "/list/enrichment"
    SERVICE_RESOLUTION_PATH = "/check/"
    _DEFAULT_COMPATIBILITY = "native"

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
        *,
        compatibility=None,
    ):
        self.compatibility = resolve_compatibility(compatibility, default=self._DEFAULT_COMPATIBILITY)
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
        self._model_xml = None
        self._model_name = None
        self._query_model = None
        self._version = None
        self._release = None
        self._widgets = None
        self._templates = None
        self._templates_raw = None
        self._all_templates = None
        self._all_templates_raw = None
        self._all_templates_names = None
        self._list_manager = None
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
                try:
                    token = self._request_anonymous_token(url=root, opener=pre_auth_opener)
                except BaseException:
                    pre_auth_opener.close()
                    raise
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
            try:
                self.version
            except WebserviceError as e:
                raise ServiceError(f"Could not validate service - is the root url ({root}) correct? {e}")
            if token and self.version < 6:
                raise ServiceError("This service does not support API access token authentication")
        except BaseException:
            self.close()
            raise

    def get_anonymous_token(self, url):
        """Generate a 24-hour anonymous session token using this client's opener."""
        return self._request_anonymous_token(url)

    def list_manager(self):
        """Return an independent manager using this service's opener."""
        from intermine314.lists.listmanager import ListManager

        return ListManager(self)

    def __getattr__(self, name):
        if name in self.LIST_MANAGER_METHODS:
            return getattr(self._get_list_manager(), name)
        raise AttributeError("Could not find " + name)

    def _get_list_manager(self):
        """Allocate the internal manager only when a list delegate needs it."""
        if self._list_manager is None:
            self._list_manager = self.list_manager()
        return self._list_manager

    @requires_version(9)
    def register(self, username, password):
        """Register an account and return a configured, authenticated Service.

        Owned sessions are independent between the two clients. An externally
        supplied session remains borrowed by both clients.
        """
        payload = urlencode({'name': username, 'password': password})
        with closing(self.opener.clone()) as registrar:
            registrar.token = None
            registrar.using_authentication = False
            data = self._get_json(self.USERS_PATH, payload=payload, opener=registrar)
        return type(self)(
            self.root, token=data['user']['temporaryToken'],
            prefetch_depth=self.prefetch_depth, prefetch_id_only=self.prefetch_id_only,
            request_timeout=self.request_timeout, proxy_url=self.proxy_url,
            session=None if self.opener._owns_session else self.opener._session,
            verify_tls=self.verify_tls, tor=self.tor,
            strict_tor_proxy_scheme=self.strict_tor_proxy_scheme,
            allow_insecure_tor_proxy_scheme=self.allow_insecure_tor_proxy_scheme,
            allow_http_over_tor=self.allow_http_over_tor, user_agent=self.user_agent,
            compatibility=self.compatibility,
        )

    @requires_version(16)
    def get_deregistration_token(self, validity=300):
        """Return proof for account deletion, valid for 1 through 86400 seconds."""
        if validity < 1 or validity > 86400:
            raise ValueError("Validity must be between 1 and 86400 seconds")
        data = self._get_json('/user/deregistration', payload=urlencode({'validity': str(validity)}))
        return data['token']

    @requires_version(16)
    def deregister(self, deregistration_token):
        """Delete this account using a UUID dictionary or token string; return XML bytes."""
        if isinstance(deregistration_token, dict):
            deregistration_token = deregistration_token['uuid']
        uri = self.root + '/user?' + urlencode({
            'deregistrationToken': deregistration_token, 'format': 'xml',
        })
        self.flush()
        return self.opener.delete(uri)

    def resolve_ids(self, data_type, identifiers, extra='',
                    case_sensitive=False, wildcards=False):
        """Submit identifiers to API version 10+ and return a resolution Job."""
        if self.version < 10:
            raise ServiceError('This feature requires API version 10+')
        if not data_type:
            raise ServiceError('No data-type supplied')
        if not identifiers:
            raise ServiceError('No identifiers supplied')

        data = json.dumps({
            'type': data_type, 'identifiers': list(identifiers), 'extra': extra,
            'caseSensitive': case_sensitive, 'wildCards': wildcards,
        })
        response = json.loads(self.opener.post_content(
            self.root + self.IDS_PATH, data, InterMineURLOpener.JSON,
        ))
        if response['error'] is not None:
            raise ServiceError(response['error'])

        from intermine314.idresolution import Job

        return Job(self, response['uid'])

    def _invalidate_caches(self):
        for name in ('_model', '_model_xml', '_model_name', '_query_model', '_version', '_release', '_widgets',
                     '_templates', '_templates_raw', '_all_templates', '_all_templates_raw', '_all_templates_names'):
            setattr(self, name, None)

    def flush(self):
        """Clean internal temporary lists and invalidate caches, retaining the opener.

        Independent caller-owned managers are unaffected. Failed cleanup keeps
        the internal manager and metadata caches available for a later retry.
        """
        if self._list_manager is not None:
            self._list_manager.delete_temporary_lists()
        self._list_manager = None
        self._invalidate_caches()

    def _request_anonymous_token(self, url, opener=None):
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
        """Return the optional service path as bytes through the managed opener."""
        url = self.root + self.SERVICE_RESOLUTION_PATH + variant
        with closing(self.opener.open(url)) as response:
            return response.read()

    @property
    def release(self):
        """Return and cache the decoded, stripped data warehouse release."""
        if self._release is None:
            with closing(self.opener.open(self.root + self.RELEASE_PATH)) as response:
                self._release = ensure_str(response.read()).strip()
        return self._release

    def _get_json(self, path, payload=None, *, opener=None):
        """Read service JSON, retaining parse errors and the service error contract."""
        with closing((self.opener if opener is None else opener).open(
            self.root + path, payload, headers={'Accept': 'application/json'},
        )) as response:
            data = json.loads(ensure_str(response.read()))
        if data['error'] is not None:
            raise ServiceError(data['error'])
        return data

    def _get_xml(self, path):
        """Read a DOM document through the service's configured transport."""
        with closing(self.opener.open(
            self.root + path, headers={'Accept': 'application/xml'},
        )) as response:
            return minidom.parse(response)

    def _read_template_snapshot(self, path, *, by_user=False):
        """Read raw template XML, rejecting duplicate names within each owner."""
        snapshot = {}
        dom = self._get_xml(path)
        try:
            for element in dom.getElementsByTagName('template'):
                name = element.getAttribute('name')
                templates = (snapshot.setdefault(element.getAttribute('userName'), {})
                             if by_user else snapshot)
                if name in templates:
                    raise ServiceError('Two templates with same name: ' + name)
                templates[name] = element.toxml()
        finally:
            dom.unlink()
        return snapshot

    @property
    def templates(self):
        """Return global names mapped to XML, replaced by Templates on access."""
        if self._templates is None:
            if self._templates_raw is None:
                self._templates_raw = self._read_template_snapshot(self.TEMPLATES_PATH)
            self._templates = dict(self._templates_raw)
        return self._templates

    def _get_all_templates_raw(self):
        """Share one immutable-by-convention XML snapshot across discovery views."""
        if self._all_templates_raw is None:
            self._all_templates_raw = self._read_template_snapshot(self.ALL_TEMPLATES_PATH, by_user=True)
        return self._all_templates_raw

    @property
    def all_templates(self):
        """Return owner dictionaries of XML or lazily parsed bound Templates."""
        if self._all_templates is None:
            self._all_templates = {user: dict(templates)
                                   for user, templates in self._get_all_templates_raw().items()}
        return self._all_templates

    @property
    def all_templates_names(self):
        """Return cached names per owner from the shared discovery snapshot."""
        if self._all_templates_names is None:
            self._all_templates_names = {user: list(templates)
                                         for user, templates in self._get_all_templates_raw().items()}
        return self._all_templates_names

    def get_template(self, name):
        """Return and cache a parsed global Template bound to this service."""
        try:
            template = self.templates[name]
        except KeyError:
            raise ServiceError("There is no template called '" + name + "' at this service") from None
        from intermine314.query import Template

        if not isinstance(template, Template):
            template = Template.from_xml(template, self.model, self, compatibility=self.compatibility)
            self.templates[name] = template
        return template

    def get_template_by_user(self, name, username):
        """Return and cache a Template under its actual owner and name."""
        try:
            templates = self.all_templates[username]
        except KeyError:
            raise ServiceError("There is no user called '" + username + "'") from None
        try:
            template = templates[name]
        except KeyError:
            raise ServiceError("There is no template called '" + name
                               + "' at this service belonging to '" + username + "'") from None
        from intermine314.query import Template

        if not isinstance(template, Template):
            template = Template.from_xml(template, self.model, self, compatibility=self.compatibility)
            template.user_name = username
            templates[name] = template
        return template

    def search(self, term, **facets):
        """Search indexed objects and return their results and facet information."""
        params = [('q', term)]
        params.extend(('facet_' + name, value) for name, value in facets.items())
        data = self._get_json(self.SEARCH_PATH, payload=urlencode(params, doseq=True))
        return data['results'], data['facets']

    @property
    def widgets(self):
        """Return cached widget metadata keyed by each widget's name."""
        if self._widgets is None:
            widgets = self._get_json(self.WIDGETS_PATH)['widgets']
            self._widgets = {widget['name']: widget for widget in widgets}
        return self._widgets

    def select(self, *columns, **kwargs):
        """Construct a bound query from columns/descriptors or ``xml=...``.

        ``root`` optionally supplies the root for relative columns or saved XML.
        XML and explicit columns cannot be combined.
        """
        unsupported = kwargs.keys() - {'xml', 'root'}
        if unsupported:
            raise TypeError(f"Unsupported query factory arguments: {', '.join(sorted(unsupported))}")
        root = kwargs.get('root')
        if 'xml' in kwargs:
            if columns:
                raise TypeError("xml and columns cannot be combined")
            return self.load_query(kwargs['xml'], root=root)
        model = self._resolve_query_model()
        root, paths, subclasses, descriptors = _normalize_query_columns(model, columns, root)
        query = _query_class()(
            model=model, service=self, root=root,
            validate=getattr(self, "compatibility", self._DEFAULT_COMPATIBILITY) == "legacy",
            compatibility=getattr(self, "compatibility", self._DEFAULT_COMPATIBILITY),
        )
        for path, subclass in subclasses.items():
            query.add_constraint(path=path, subclass=subclass)
        if subclasses:
            # Descriptor refinements must be valid before selecting fields,
            # including in the native profile's permissive string mode.
            query.verify_constraint_paths()
        if len(paths) == 1 and not descriptors:
            token = paths[0]
            if token and not token.endswith('*'):
                if query._has_model():
                    from intermine314.model import ModelError

                    try:
                        path = query._model_path(query.prefix_path(token))
                    except ModelError:
                        if query.compatibility == 'legacy':
                            raise
                        if root is None and '.' not in token:
                            paths[0] += '.*'
                    else:
                        if not path.is_attribute():
                            paths[0] += '.*'
                elif '.' not in token:
                    paths[0] += '.*'
        query.select(*paths)
        if descriptors:
            query.verify_views()
        return query

    new_query = select
    query = select

    def load_query(self, xml, root=None):
        """Load a saved query bound to this service and its managed transport."""
        return _query_class().from_xml(
            xml, model=self._resolve_query_model(), service=self, root=root,
            compatibility=getattr(self, "compatibility", self._DEFAULT_COMPATIBILITY),
        )

    def _read_model_xml(self):
        if getattr(self, "_model_xml", None) is None:
            with closing(self.opener.open(self.root + self.MODEL_PATH, method="GET")) as response:
                self._model_xml = response.read()
        return self._model_xml

    def _resolve_model_name(self):
        cached = getattr(self, "_model_name", None)
        if cached is not None:
            return str(cached)
        try:
            node = _ET.fromstring(self._read_model_xml())
            name = str(node.attrib.get("name", "")).strip()
        except Exception:
            name = ""
        self._model_name = name
        return name

    def _resolve_query_model(self):
        cached = getattr(self, "_query_model", None)
        if cached is not None:
            return cached
        if getattr(self, "compatibility", self._DEFAULT_COMPATIBILITY) == "legacy":
            return self.model
        # Read/name/parse share one payload, even if strict model parsing fails.
        name = self._resolve_model_name()
        if getattr(self, "_model_xml", None) is not None:
            try:
                return self.model
            except Exception:
                pass
        # Native string queries also support unavailable or name-only models.
        if name:
            self._query_model = SimpleNamespace(name=name)
        return getattr(self, "_query_model", None)

    @property
    def model(self):
        """Lazily parse the model using this service's managed transport."""
        if getattr(self, "_model", None) is None:
            from intermine314.model import Model

            # Close the owned response before parsing, including parse failures.
            payload = self._read_model_xml()
            self._model = Model(payload, service=self)
            self._model_name = self._model.name
            self._query_model = self._model
        return self._model

    def get_results(self, path, params, rowformat, view, cld=None, *, decimal_paths=()):
        """Return a result iterator for a query request."""
        if rowformat == "jsonobjects" or rowformat.startswith("object"):
            from intermine314.model import Class, Model, ModelError

            if not isinstance(cld, Class):
                model = self.model
                if not isinstance(model, Model):
                    raise ModelError("Object results require a valid service model")
                name = cld if cld is not None else (str(view[0]).split(".", 1)[0] if view else None)
                if name is None:
                    raise ModelError("Object results require a root class or selected view")
                cld = model.get_class(name)
        return ResultIterator(self, path, params, rowformat, view, cld, decimal_paths=decimal_paths)

    def execute(self, spec):
        """Build an execution adapter for a query specification."""
        from intermine314.query.executor import QueryExecutor

        return QueryExecutor(self, spec)
