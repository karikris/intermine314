import warnings
import xml.etree.ElementTree as etree

from intermine314.config.constants import DEFAULT_REQUEST_TIMEOUT_SECONDS
from intermine314.service.service import Registry
from intermine314.service.transport import build_session, resolve_proxy_url
from intermine314.service.urls import service_root_from_payload

"""
Legacy helpers for saved-query account operations.
===============================================

This module is deprecated. Prefer ``intermine314.service.Service`` APIs directly,
or instantiate ``QueryManager`` explicitly instead of using module-level state.
"""


REGISTRY_INSTANCES_URL = Registry.DEFAULT_REGISTRY_URL.rstrip("/")
REQUEST_TIMEOUT_SECONDS = DEFAULT_REQUEST_TIMEOUT_SECONDS

USER_QUERIES_PATH = "/user/queries"
SERVICE_VERSION_PATH = "/version"

NO_SAVED_QUERIES = "No saved queries"
NO_SUCH_QUERY_AVAILABLE = "No such query available"
INCORRECT_FORMAT = "Incorrect format"

_LEGACY_DEPRECATION_MESSAGE = (
    "intermine314.query_manager module-level functions are deprecated. "
    "Use QueryManager instances or Service APIs instead."
)
_LEGACY_WARNING_EMITTED = False


def _exception_message(exc, suffix):
    return f"An exception of type {type(exc).__name__} occurred.{suffix}"


class QueryManager:
    """Stateful saved-query manager with explicit, instance-local credentials/session."""

    def __init__(
        self,
        *,
        registry_instances_url=REGISTRY_INSTANCES_URL,
        request_timeout=REQUEST_TIMEOUT_SECONDS,
        proxy_url=None,
        session=None,
    ):
        self.registry_instances_url = str(registry_instances_url).rstrip("/")
        self.request_timeout = request_timeout
        self._proxy_url = resolve_proxy_url(proxy_url)
        self._session = session
        self._mine = ""
        self._token = ""

    @property
    def mine(self):
        return self._mine

    @property
    def token(self):
        return self._token

    @property
    def http_session(self):
        return self._session

    def get_saved_credentials(self):
        return self._mine, self._token

    def clear_credentials(self):
        self._mine = ""
        self._token = ""

    def configure_http_session(self, proxy_url=None, session=None):
        """
        Override HTTP transport.

        Use ``proxy_url=\"socks5h://127.0.0.1:9050\"`` for Tor routing.
        """
        if session is not None:
            self._session = session
            return
        self._proxy_url = resolve_proxy_url(proxy_url)
        self._session = build_session(proxy_url=self._proxy_url, user_agent=None)

    def _ensure_session(self):
        if self._session is None:
            self._session = build_session(proxy_url=self._proxy_url, user_agent=None)
        return self._session

    def _request(self, method, url, **kwargs):
        kwargs.setdefault("timeout", self.request_timeout)
        response = self._ensure_session().request(method, url, **kwargs)
        response.raise_for_status()
        return response

    def _request_json(self, method, url, **kwargs):
        return self._request(method, url, **kwargs).json()

    def _registry_instance_payload(self, mine_name):
        src = f"{self.registry_instances_url}/{mine_name}"
        payload = self._request_json("GET", src)
        instance = payload.get("instance")
        if not isinstance(instance, dict):
            raise KeyError("Registry response missing 'instance' block")
        return instance

    @staticmethod
    def _service_root_from_instance(instance):
        return service_root_from_payload(instance, normalize=True)

    def _resolve_service_root(self, mine_name):
        return self._service_root_from_instance(self._registry_instance_payload(mine_name))

    @staticmethod
    def _user_queries_url(service_root):
        return f"{service_root}{USER_QUERIES_PATH}"

    def _get_user_queries(self, service_root, api_token):
        return self._request_json(
            "GET",
            self._user_queries_url(service_root),
            params={"token": api_token},
        )

    @staticmethod
    def _query_names(payload):
        queries = payload.get("queries", {})
        if isinstance(queries, dict):
            return list(queries.keys())
        return []

    def _service_version(self, service_root, api_token):
        payload = self._request_json(
            "GET",
            f"{service_root}{SERVICE_VERSION_PATH}",
            params={"token": api_token},
        )
        try:
            return int(payload)
        except Exception:
            return int(str(payload).strip())

    def save_mine_and_token(self, mine_name, api_token):
        self._mine = mine_name
        self._token = api_token

        if self._mine == "mock":
            return None

        try:
            service_root = self._resolve_service_root(self._mine)
        except Exception as exc:
            return _exception_message(exc, " Check mine")

        try:
            _ = self._query_names(self._get_user_queries(service_root, self._token))
        except Exception as exc:
            return _exception_message(exc, " Check token")
        return None

    def get_all_query_names(self):
        if self._mine == "mock":
            names = ["query1"]
        else:
            service_root = self._resolve_service_root(self._mine)
            names = self._query_names(self._get_user_queries(service_root, self._token))

        if not names:
            return NO_SAVED_QUERIES
        return ", ".join(names)

    def get_query(self, name):
        if self._mine == "mock":
            ans = "c1, c2" if name == "query1" else "<saved-queries></saved-queries>"
        else:
            service_root = self._resolve_service_root(self._mine)
            ans = self._request(
                "GET",
                self._user_queries_url(service_root),
                params={"filter": name, "format": "xml", "token": self._token},
            ).text
        if ans == "<saved-queries></saved-queries>":
            return NO_SUCH_QUERY_AVAILABLE
        return ans

    def delete_query(self, name):
        if self._mine == "mock":
            names = ["query1", "query2"]
            if name not in names:
                return NO_SUCH_QUERY_AVAILABLE
            return name + " is deleted"

        service_root = self._resolve_service_root(self._mine)
        names = self._query_names(self._get_user_queries(service_root, self._token))
        if name not in names:
            return NO_SUCH_QUERY_AVAILABLE

        self._request(
            "DELETE",
            f"{self._user_queries_url(service_root)}/{name}",
            params={"token": self._token},
        )
        return name + " is deleted"

    def post_query(self, value, input_func=input):
        param_name = "xml"
        root = etree.fromstring(value)
        query_name = root.attrib["name"]

        if self._mine == "mock":
            names = ["query1", "query2"]
            service_root = None
        else:
            service_root = self._resolve_service_root(self._mine)
            version = self._service_version(service_root, self._token)
            if version >= 27:
                param_name = "query"
            names = self._query_names(self._get_user_queries(service_root, self._token))

        count = 0
        for existing_name in names:
            if existing_name == query_name:
                count += 1
                print("The query name exists")
                resp = input_func("Do you want to replace the old query? [y/n]")
                if resp == "y":
                    count = 0

        if count == 0:
            if self._mine == "mock":
                names = ["query1", "query2", "query3"]
            else:
                self._request(
                    "PUT",
                    self._user_queries_url(service_root),
                    params={param_name: value, "token": self._token},
                )
                names = self._query_names(self._get_user_queries(service_root, self._token))

            if query_name not in names:
                print("Note: name should contain no special symbol and should be defined first")
                return INCORRECT_FORMAT
            return query_name + " is posted"

        print("Use a query name other than " + query_name)
        return INCORRECT_FORMAT


def _warn_legacy_usage():
    global _LEGACY_WARNING_EMITTED
    if _LEGACY_WARNING_EMITTED:
        return
    warnings.warn(_LEGACY_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=3)
    _LEGACY_WARNING_EMITTED = True


_DEFAULT_MANAGER = QueryManager()


def reset_state():
    """Reset legacy module-level state (credentials + transport)."""
    global _DEFAULT_MANAGER
    _DEFAULT_MANAGER = QueryManager()


def get_saved_credentials():
    return _DEFAULT_MANAGER.get_saved_credentials()


def configure_http_session(proxy_url=None, session=None):
    _warn_legacy_usage()
    _DEFAULT_MANAGER.configure_http_session(proxy_url=proxy_url, session=session)


def save_mine_and_token(m, t):
    _warn_legacy_usage()
    return _DEFAULT_MANAGER.save_mine_and_token(m, t)


def get_all_query_names():
    _warn_legacy_usage()
    return _DEFAULT_MANAGER.get_all_query_names()


def get_query(name):
    _warn_legacy_usage()
    return _DEFAULT_MANAGER.get_query(name)


def delete_query(name):
    _warn_legacy_usage()
    return _DEFAULT_MANAGER.delete_query(name)


def post_query(value):
    _warn_legacy_usage()
    return _DEFAULT_MANAGER.post_query(value)


def __getattr__(name):
    if name == "mine":
        return _DEFAULT_MANAGER.mine
    if name == "token":
        return _DEFAULT_MANAGER.token
    if name == "HTTP_SESSION":
        return _DEFAULT_MANAGER.http_session
    raise AttributeError(name)
