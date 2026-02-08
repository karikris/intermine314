import requests

import xml.etree.ElementTree as etree

from intermine314.service_urls import service_root_from_payload
from intermine314.webservice import Registry

"""
Functions for better usage of saved queries.
================================================
Prompts the user to enter the API token and mine corresponding to the account.

example:

    >>>from intermine314 import query_manager as qm
"""


REGISTRY_INSTANCES_URL = Registry.DEFAULT_REGISTRY_URL.rstrip("/")
REQUEST_TIMEOUT_SECONDS = 60
HTTP_SESSION = requests.Session()

USER_QUERIES_PATH = "/user/queries"
SERVICE_VERSION_PATH = "/version"

NO_SAVED_QUERIES = "No saved queries"
NO_SUCH_QUERY_AVAILABLE = "No such query available"
INCORRECT_FORMAT = "Incorrect format"

mine = ""
token = ""


def _exception_message(exc, suffix):
    return f"An exception of type {type(exc).__name__} occurred.{suffix}"


def _request(method, url, **kwargs):
    kwargs.setdefault("timeout", REQUEST_TIMEOUT_SECONDS)
    response = HTTP_SESSION.request(method, url, **kwargs)
    response.raise_for_status()
    return response


def _request_json(method, url, **kwargs):
    return _request(method, url, **kwargs).json()


def _registry_instance_payload(mine_name):
    src = f"{REGISTRY_INSTANCES_URL}/{mine_name}"
    payload = _request_json("GET", src)
    instance = payload.get("instance")
    if not isinstance(instance, dict):
        raise KeyError("Registry response missing 'instance' block")
    return instance


def _service_root_from_instance(instance):
    return service_root_from_payload(instance, normalize=True)


def _resolve_service_root(mine_name):
    return _service_root_from_instance(_registry_instance_payload(mine_name))


def _user_queries_url(service_root):
    return f"{service_root}{USER_QUERIES_PATH}"


def _get_user_queries(service_root, api_token):
    return _request_json(
        "GET",
        _user_queries_url(service_root),
        params={"token": api_token},
    )


def _query_names(payload):
    queries = payload.get("queries", {})
    if isinstance(queries, dict):
        return list(queries.keys())
    return []


def _service_version(service_root, api_token):
    payload = _request_json(
        "GET",
        f"{service_root}{SERVICE_VERSION_PATH}",
        params={"token": api_token},
    )
    try:
        return int(payload)
    except Exception:
        return int(str(payload).strip())


def save_mine_and_token(m, t):
    """
    A function to access an account from a particular mine.
    ================================================
    example:

        >>>from intermine314 import query_manager as qm
        >>>qm.save_mine_and_token("flymine","<enter token>")
        <now you can access account linked to the token>

    """
    global mine
    global token
    mine = m
    token = t

    # test shortcut
    if mine == "mock":
        return None

    try:
        service_root = _resolve_service_root(mine)
    except Exception as exc:
        return _exception_message(exc, " Check mine")

    try:
        _ = _query_names(_get_user_queries(service_root, token))
    except Exception as exc:
        return _exception_message(exc, " Check token")
    return None


def get_all_query_names():
    """
    A function to list all the queries that are saved in a user account.
    ================================================
    example:

        >>>from intermine314 import query_manager as qm
        >>>qm.save_mine_and_token("flymine","<enter token>")
        >>>qm.get_all_query_names()
        <returns the names of all the saved queries in user account>

    """
    if mine == "mock":
        names = ["query1"]
    else:
        service_root = _resolve_service_root(mine)
        names = _query_names(_get_user_queries(service_root, token))

    if not names:
        return NO_SAVED_QUERIES
    return ", ".join(names)


def get_query(name):
    """
    A function that returns the columns that a given query constitutes.
    ================================================
    example:

        >>>from intermine314 import query_manager as qm
        >>>qm.save_mine_and_token("flymine","<enter token>")
        >>>qm.get_query('queryName')
        <returns information about the query whose name is 'queryName'>

    """
    if mine == "mock":
        ans = "c1, c2" if name == "query1" else "<saved-queries></saved-queries>"
    else:
        service_root = _resolve_service_root(mine)
        ans = _request(
            "GET",
            _user_queries_url(service_root),
            params={"filter": name, "format": "xml", "token": token},
        ).text
    if ans == "<saved-queries></saved-queries>":
        return NO_SUCH_QUERY_AVAILABLE
    return ans


def delete_query(name):
    """
    A function that deletes a given query.
    ================================================
    example:

        >>>from intermine314 import query_manager as qm
        >>>qm.save_mine_and_token("flymine","<enter token>")
        >>>qm.delete_query('queryName')
        <deletes the query whose name is 'queryName' from user's account>

    """
    if mine == "mock":
        names = ["query1", "query2"]
        if name not in names:
            return NO_SUCH_QUERY_AVAILABLE
        return name + " is deleted"

    service_root = _resolve_service_root(mine)
    names = _query_names(_get_user_queries(service_root, token))
    if name not in names:
        return NO_SUCH_QUERY_AVAILABLE

    _request(
        "DELETE",
        f"{_user_queries_url(service_root)}/{name}",
        params={"token": token},
    )
    return name + " is deleted"


def post_query(value):
    """
    A function to post a query (string containing xml or json) to a user account.
    ================================================
    example:
        >>>from intermine314 import query_manager as qm
        >>>qm.save_mine_and_token("flymine","<enter token>")
        >>>qm.post_query('<query name="" model="genomic" view="Gene.length\
            Gene.symbol" longDescription="" sortOrder="Gene.length asc">\
            </query>')
    Note that the name should be defined first.
    """
    param_name = "xml"
    root = etree.fromstring(value)
    query_name = root.attrib["name"]

    if mine == "mock":
        names = ["query1", "query2"]
        service_root = None
    else:
        service_root = _resolve_service_root(mine)
        version = _service_version(service_root, token)
        if version >= 27:
            param_name = "query"
        names = _query_names(_get_user_queries(service_root, token))

    count = 0
    for existing_name in names:
        if existing_name == query_name:
            count += 1
            print("The query name exists")
            resp = input("Do you want to replace the old query? [y/n]")
            if resp == "y":
                count = 0

    if count == 0:
        if mine == "mock":
            names = ["query1", "query2", "query3"]
        else:
            _request(
                "PUT",
                _user_queries_url(service_root),
                params={param_name: value, "token": token},
            )
            names = _query_names(_get_user_queries(service_root, token))

        if query_name not in names:
            print("Note: name should contain no special symbol and should be defined first")
            return INCORRECT_FORMAT
        return query_name + " is posted"

    print("Use a query name other than " + query_name)
