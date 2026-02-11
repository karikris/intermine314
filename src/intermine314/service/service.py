from xml.dom import minidom
from contextlib import closing
import json
from uuid import uuid4

from urllib.parse import urlencode
from collections.abc import MutableMapping as DictMixin

# Local intermine314 imports
from intermine314.model import Model, Attribute, Reference, Collection, Column
from intermine314.lists.listmanager import ListManager
from intermine314.errors import ServiceError, WebserviceError
from intermine314.service.session import InterMineURLOpener, ResultIterator
from intermine314 import idresolution
from intermine314.decorators import requires_version
from intermine314.constants import DEFAULT_LIST_CHUNK_SIZE, DEFAULT_REQUEST_TIMEOUT_SECONDS
from intermine314.service.transport import build_session, resolve_proxy_url
from intermine314.service_urls import normalize_service_root, service_root_from_payload

"""
Webservice Interaction Routines for InterMine Webservices
=========================================================

Classes for dealing with communication with an InterMine
RESTful webservice.

"""

__author__ = "Alex Kalderimis"
__organization__ = "InterMine"
__license__ = "LGPL"
__contact__ = "toffe.kari@gmail.com"

_QUERY_CLASS = None
_TEMPLATE_CLASS = None


def _query_classes():
    global _QUERY_CLASS, _TEMPLATE_CLASS
    if _QUERY_CLASS is None or _TEMPLATE_CLASS is None:
        from intermine314.query import Query, Template

        _QUERY_CLASS = Query
        _TEMPLATE_CLASS = Template
    return _QUERY_CLASS, _TEMPLATE_CLASS


class Registry(DictMixin):
    """
    A Class representing an InterMine registry.
    ===========================================

    Registries are web-services that mines can automatically register
    themselves with, and thus enable service discovery by clients.

    SYNOPSIS
    --------

    example::

        from intermine314.webservice import Registry

        # Connect to the default registry service
        # at www.intermine.org/registry
        registry = Registry()

        # Find all the available mines:
        for name, mine in registry.items():
            print(name, mine.version)

        # Dict-like interface for accessing mines.
        flymine = registry["flymine"]

        # The mine object is a Service
        for gene in flymine.select("Gene.*").results():
            process(gene)

    This class is meant to aid with interoperation between
    mines by allowing them to discover one-another, and
    allow users to always have correct connection information.
    """

    MINES_PATH = "/mines.json"
    INSTANCES_PATH = "/service/instances"
    DEFAULT_REGISTRY_URL = "https://registry.intermine.org/service/instances"

    def __init__(
        self,
        registry_url=DEFAULT_REGISTRY_URL,
        request_timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        proxy_url=None,
        session=None,
        verify_tls=True,
    ):
        self.registry_url = registry_url.rstrip("/")
        self.request_timeout = request_timeout
        self.proxy_url = resolve_proxy_url(proxy_url)
        self.verify_tls = bool(verify_tls)
        self._session = session
        opener = InterMineURLOpener(
            request_timeout=self.request_timeout,
            proxy_url=self.proxy_url,
            session=session,
            verify_tls=self.verify_tls,
        )
        self._session = opener._session
        data = opener.open(self._list_url()).read()
        mine_data = json.loads(ensure_str(data))
        mines = self._extract_mines(mine_data)
        self.__mine_dict = dict(((mine["name"], mine) for mine in mines))
        self.__synonyms = dict(((name.lower(), name) for name in list(self.__mine_dict.keys())))
        self.__mine_cache = {}

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
        if lc in self.__synonyms:
            if lc not in self.__mine_cache:
                mine = self.__mine_dict[self.__synonyms[lc]]
                self.__mine_cache[lc] = Service(
                    self._service_root(mine),
                    request_timeout=self.request_timeout,
                    proxy_url=self.proxy_url,
                    session=self._session,
                    verify_tls=self.verify_tls,
                )
            return self.__mine_cache[lc]
        else:
            raise KeyError("Unknown mine: " + name)

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
        """
        Return the registry info dictionary for a mine.
        """
        lc = name.lower()
        if lc in self.__synonyms:
            return self.__mine_dict[self.__synonyms[lc]]
        raise KeyError("Unknown mine: " + name)

    def service_root(self, name):
        """
        Return the service root URL for a mine.
        """
        return self._service_root(self.info(name))

    def all_mines(self, organism=None):
        """
        Return a list of registry info dictionaries, optionally filtered by organism.
        """
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
    """
    A class representing connections to different InterMine WebServices
    ===================================================================

    The intermine314.webservice.Service class is the main interface for the user.
    It will provide access to queries and templates, as well as doing the
    background task of fetching the data model, and actually requesting
    the query results.

    SYNOPSIS
    --------

    example::

      from intermine314.webservice import Service
      service = Service("https://www.flymine.org/query/service")

      template = service.get_template("Gene_Pathways")
      for row in template.results(A={"value":"zen"}):
        do_something_with(row)
        ...

      query = service.new_query()
      query.add_view("Gene.symbol", "Gene.pathway.name")
      query.add_constraint("Gene", "LOOKUP", "zen")
      for row in query.results():
        do_something_with(row)
        ...

      new_list = service.create_list("some/file/with.ids", "Gene")
      list_on_server = service.get_list("On server")
      in_both = new_list & list_on_server
      in_both.name = "Intersection of these lists"
      for row in in_both:
        do_something_with(row)
        ...

    OVERVIEW
    --------
    The two methods the user will be most concerned with are:
      - L{Service.new_query}: constructs a new query to query a service with
      - L{Service.get_template}: gets a template from the service
      - L{ListManager.create_list}: creates a new list on the service

    For list management information, see L{ListManager}.

    TERMINOLOGY
    -----------
    X{Query} is the term for an arbitrarily complex structured request for
    data from the webservice. The user is responsible for specifying the
    structure that determines what records are returned, and what information
    about each record is provided.

    X{Template} is the term for a predefined "Query", ie: one that has been
    written and saved on the webservice you will access. The definition
    of the query is already done, but the user may want to specify the
    values of the constraints that exist on the template. Templates are accessed
    by name, and while you can easily introspect templates, it is assumed
    you know what they do when you use them

    X{List} is a saved result set containing a set of objects previously identified
    in the database. Lists can be created and managed using this client library.

    @see: L{intermine314.query}
    """

    QUERY_PATH = "/query/results"
    LIST_ENRICHMENT_PATH = "/list/enrichment"
    WIDGETS_PATH = "/widgets"
    SEARCH_PATH = "/search"
    QUERY_LIST_UPLOAD_PATH = "/query/tolist"
    QUERY_LIST_APPEND_PATH = "/query/append/tolist"
    MODEL_PATH = "/model"
    TEMPLATES_PATH = "/templates"
    TEMPLATEQUERY_PATH = "/template/results"
    ALL_TEMPLATES_PATH = "/alltemplates"
    LIST_PATH = "/lists"
    LIST_CREATION_PATH = "/lists"
    LIST_RENAME_PATH = "/lists/rename"
    LIST_APPENDING_PATH = "/lists/append"
    LIST_TAG_PATH = "/list/tags"
    SAVEDQUERY_PATH = "/savedqueries/xml"
    VERSION_PATH = "/version/ws"
    RELEASE_PATH = "/version/release"
    SCHEME = "http://"
    SERVICE_RESOLUTION_PATH = "/check/"
    IDS_PATH = "/ids"
    USERS_PATH = "/users"
    REGISTRY_URL = Registry.DEFAULT_REGISTRY_URL

    def __init__(
        self,
        root,
        username=None,
        password=None,
        token=None,
        prefetch_depth=1,
        prefetch_id_only=False,
        request_timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        proxy_url=None,
        session=None,
        verify_tls=True,
    ):
        """
        Constructor
        ===========

        Construct a connection to a webservice::

            url = "https://www.flymine.org/query/service"

            # An unauthenticated connection - access to all public data
            service = Service(url)

            # An authenticated connection - access to private and public data
            service = Service(url, token="ABC123456")


        @param root: the root url of the webservice (required)
        @param username: your login name (optional)
        @param password: your password (required if a username is given)
        @param token: your API access token(optional - used in preference to username and password)

        @raise ServiceError: if the version cannot be fetched and parsed
        @raise ValueError:   if a username is supplied, but no password

        There are two alternative authentication systems supported by InterMine
        webservices. The first is username and password authentication, which
        is supported by all webservices. Newer webservices (version 6+)
        also support API access token authentication, which is the recommended
        system to use. Token access is more secure as you will never have
        to transmit your username or password, and the token can be easily changed
        or disabled without changing your webapp login details.

        """
        root = normalize_service_root(root)

        self.root = root
        self.prefetch_depth = prefetch_depth
        self.prefetch_id_only = prefetch_id_only
        self.request_timeout = request_timeout
        self.proxy_url = resolve_proxy_url(proxy_url)
        self.verify_tls = bool(verify_tls)
        # Initialize empty cached data.
        self._templates = None
        self._all_templates = None
        self._all_templates_names = None
        self._model = None
        self._version = None
        self._release = None
        self._widgets = None
        self._list_manager = ListManager(self)
        self.__missing_method_name = None
        opener_session = session
        if token:
            if token == "random":
                pre_auth_opener = InterMineURLOpener(
                    request_timeout=self.request_timeout,
                    proxy_url=self.proxy_url,
                    session=opener_session,
                    verify_tls=self.verify_tls,
                )
                token = self.get_anonymous_token(url=root, opener=pre_auth_opener)
                opener_session = pre_auth_opener._session
            self.opener = InterMineURLOpener(
                token=token,
                request_timeout=self.request_timeout,
                proxy_url=self.proxy_url,
                session=opener_session,
                verify_tls=self.verify_tls,
            )
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
            )
        else:
            self.opener = InterMineURLOpener(
                request_timeout=self.request_timeout,
                proxy_url=self.proxy_url,
                session=opener_session,
                verify_tls=self.verify_tls,
            )

        try:
            self.version
        except WebserviceError as e:
            raise ServiceError("Could not validate service - is the root url (%s) correct? %s" % (root, e))

        if token and self.version < 6:
            raise ServiceError("This service does not support API access token authentication")

        # Set up sugary aliases
        self.query = self.new_query

    # Delegated list methods

    LIST_MANAGER_METHODS = frozenset(
        ["get_list", "get_all_lists", "get_all_list_names", "create_list", "get_list_count", "delete_lists", "l"]
    )

    def get_anonymous_token(self, url, opener=None):
        """
        Generates an anonymous session token valid for 24 hours
        =======================================================
        """
        url += "/session"
        if opener is None:
            opener = self.opener if hasattr(self, "opener") else InterMineURLOpener(
                request_timeout=self.request_timeout,
                proxy_url=self.proxy_url,
                verify_tls=self.verify_tls,
            )

        session = opener._session or build_session(proxy_url=self.proxy_url, user_agent=None)
        token_resp = session.get(
            url=url,
            timeout=opener._timeout,
            verify=opener._verify_tls,
        )
        token_resp.raise_for_status()
        token = token_resp.json()["token"]
        return token

    def get_mine_info(self, mine_name, registry_url=None):
        """
        Fetch registry info for a given mine.
        """
        registry = Registry(
            registry_url=registry_url or self.REGISTRY_URL,
            request_timeout=self.request_timeout,
            proxy_url=self.proxy_url,
            verify_tls=self.verify_tls,
        )
        return registry.info(mine_name)

    @classmethod
    def get_all_mines(
        cls,
        organism=None,
        registry_url=None,
        request_timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        proxy_url=None,
        verify_tls=True,
    ):
        """
        Fetch all registry mines, optionally filtered by organism.
        """
        registry = Registry(
            registry_url=registry_url or cls.REGISTRY_URL,
            request_timeout=request_timeout,
            proxy_url=proxy_url,
            verify_tls=verify_tls,
        )
        return registry.all_mines(organism=organism)

    def list_manager(self):
        """
        Get a new ListManager to use with this service.
        ===============================================

        This method is primarily useful as a context manager
        when creating temporary lists, since on context exit all
        temporary lists will be cleaned up::

            with service.list_manager() as manager:
                temp_a = manager.create_list(file_a, "Gene")
                temp_b = manager.create_list(file_b, "Gene")
                for gene in (temp_a & temp_b):
                    print(gene.primaryIdentifier, "is in both")

        @rtype: ListManager
        """
        return ListManager(self)

    def list_templates(self, include=None, exclude=None, limit=None):
        """
        Return template names, with optional substring include/exclude filters.
        """
        names = sorted(self.templates.keys())
        include_terms = [str(term).lower() for term in (include or []) if str(term).strip()]
        exclude_terms = [str(term).lower() for term in (exclude or []) if str(term).strip()]

        if include_terms:
            names = [name for name in names if any(term in name.lower() for term in include_terms)]
        if exclude_terms:
            names = [name for name in names if not any(term in name.lower() for term in exclude_terms)]
        if limit is not None:
            names = names[: max(0, int(limit))]
        return names

    def create_batched_lists(
        self,
        identifiers,
        list_type,
        chunk_size=DEFAULT_LIST_CHUNK_SIZE,
        name_prefix="intermine314_batch",
        description=None,
        tags=None,
    ):
        """
        Create multiple server-side lists from identifiers in fixed-size chunks.
        """
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
            raise TypeError("chunk_size must be an integer")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if not list_type:
            raise ValueError("list_type is required")

        tags = list(tags or [])
        run_id = uuid4().hex[:8]
        batch = []
        created = []
        batch_index = 1

        def flush(current):
            nonlocal batch_index
            if not current:
                return
            name = f"{name_prefix}_{run_id}_{batch_index:05d}"
            batch_index += 1
            created.append(
                self.create_list(
                    current,
                    list_type=list_type,
                    name=name,
                    description=description,
                    tags=tags,
                )
            )

        for identifier in identifiers:
            if identifier is None:
                continue
            value = str(identifier).strip()
            if not value:
                continue
            batch.append(value)
            if len(batch) >= chunk_size:
                flush(batch)
                batch = []

        flush(batch)
        return created

    def __getattribute__(self, name):
        return object.__getattribute__(self, name)

    def __getattr__(self, name):
        if name in self.LIST_MANAGER_METHODS:
            method = getattr(self._list_manager, name)
            return method
        raise AttributeError("Could not find " + name)

    def __del__(self):  # On going out of scope, try and clean up.
        try:
            self._list_manager.delete_temporary_lists()
        except ReferenceError:
            pass

    @property
    def version(self):
        """
        Returns the webservice version
        ==============================

        The version specifies what capabilities a
        specific webservice provides. The most current
        version is 3

        may raise ServiceError: if the version cannot be fetched

        @rtype: int
        """
        try:
            if self._version is None:
                try:
                    url = self.root + self.VERSION_PATH
                    self._version = int(self.opener.open(url).read())
                except ValueError as e:
                    raise ServiceError("Could not parse a valid webservice version: " + str(e))
        except AttributeError as e:
            raise Exception(e)
        return self._version

    def resolve_service_path(self, variant):
        """Resolve the path to optional services"""
        url = self.root + self.SERVICE_RESOLUTION_PATH + variant
        return self.opener.open(url).read()

    @property
    def release(self):
        """
        Returns the datawarehouse release
        =================================

        Service.release S{->} string

        The release is an arbitrary string used to distinguish
        releases of the datawarehouse. This usually coincides
        with updates to the data contained within. While a string,
        releases usually sort in ascending order of recentness
        (eg: "release-26", "release-27", "release-28"). They can also
        have less machine readable meanings (eg: "beta")

        @rtype: string
        """
        if self._release is None:
            with closing(self.opener.open(self.root + self.RELEASE_PATH)) as resp:
                self._release = ensure_str(resp.read()).strip()
        return self._release

    def load_query(self, xml, root=None):
        """
        Construct a new Query object for the given webservice
        =====================================================

        This is the standard method for instantiating new Query
        objects. Queries require access to the data model, as well
        as the service itself, so it is easiest to access them through
        this factory method.

        @return: L{intermine314.query.Query}
        """
        query_class, _ = _query_classes()
        return query_class.from_xml(xml, self.model, root=root)

    def select(self, *columns, **kwargs):
        """
        Construct a new Query object with the given columns selected.
        =============================================================

        As new_query, except that instead of a root class, a list of
        output column expressions are passed instead.
        """
        if "xml" in kwargs:
            return self.load_query(kwargs["xml"])
        query_class, _ = _query_classes()
        query = query_class(self.model, self)
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

    new_query = select

    def get_template(self, name):
        """
        Returns a template of the given name
        ====================================

        Tries to retrieve a template of the given name
        from the webservice. If you are trying to fetch
        a private template (ie. one you made yourself
        and is not available to others) then you may need to authenticate

        @see: L{intermine314.webservice.Service.__init__}

        @param name: the template's name
        @type name: string

        @raise ServiceError: if the template does not exist
        @raise QueryParseError: if the template cannot be parsed

        @return: L{intermine314.query.Template}
        """
        try:
            t = self.templates[name]
        except KeyError:
            raise ServiceError("There is no template called '" + name + "' at this service")
        _, template_class = _query_classes()
        if not isinstance(t, template_class):
            t = template_class.from_xml(t, self.model, self)
            self.templates[name] = t
        return t

    def get_template_by_user(self, name, username):
        """
        Returns a template of the given name belonging to username
        ==========================================================

        Tries to retrieve a template of the given name belonging
        to the username from the webservice. You need to authenticate
        as admin

        @see: L{intermine314.webservice.Service.__init__}

        @param name: the template's name
        @type name: string

        @param username: the username
        @type name: string

        @raise ServiceError: if the template or user does not exist
        @raise QueryParseError: if the template cannot be parsed

        @return: L{intermine314.query.Template}
        """
        try:
            templates = self.all_templates[username]
        except KeyError:
            raise ServiceError("There is no user called '" + username + "'")
        try:
            t = templates[name]
        except KeyError:
            raise ServiceError(
                "There is no template called '" + name + "' at this service belonging to '" + username + "'"
            )
        _, template_class = _query_classes()
        if not isinstance(t, template_class):
            t = template_class.from_xml(t, self.model, self)
            t.user_name = username
            self.all_templates[name] = t
        return t

    def _get_json(self, path, payload=None):
        headers = {"Accept": "application/json"}
        with closing(self.opener.open(self.root + path, payload, headers=headers)) as resp:
            data = json.loads(ensure_str(resp.read()))
            if data["error"] is not None:
                raise ServiceError(data["error"])
            return data

    def _get_xml(self, path):
        headers = {"Accept": "application/xml"}
        with closing(self.opener.open(self.root + path, headers=headers)) as sock:
            return minidom.parse(sock)

    def search(self, term, **facets):
        """
        Perform an unstructured search by term
        =======================================

        This seach method performs a search of all objects
        indexed by the service endpoint, returning results
        and facets for those results.

        @param term The search term
        @param facets The facets to search by (eg: Organism = 'H. sapiens')

        @return (list, dict) The results, and a dictionary of facetting informtation.
        """
        if isinstance(term, bytes):
            term = ensure_str(term)
        params = [("q", term)]
        for facet, value in list(facets.items()):
            if isinstance(value, bytes):
                value = ensure_str(value)
            params.append(("facet_{0}".format(facet), value))
        payload = urlencode(params, doseq=True)
        resp = self._get_json(self.SEARCH_PATH, payload=payload)
        return (resp["results"], resp["facets"])

    @property
    def widgets(self):
        """
        The dictionary of widgets from the webservice
        ==============================================

        The set of widgets available to a service does not
        change between releases, so they are cached.
        If you are running a long running process, you may
        wish to periodically dump the cache by calling
        L{Service.flush}, or simply get a new Service object.

        @return dict
        """
        if self._widgets is None:
            ws = self._get_json(self.WIDGETS_PATH)["widgets"]
            self._widgets = dict(([w["name"], w] for w in ws))
        return self._widgets

    def resolve_ids(self, data_type, identifiers, extra="", case_sensitive=False, wildcards=False):
        """
        Submit an Identifier Resolution Job
        ===================================

        Request that a set of identifiers be resolved to objects in
        the data store.

        @param data_type: The type of these identifiers (eg. 'Gene')
        @type data_type: String

        @param identifiers: The ids to resolve (eg. ['eve', 'zen', 'pparg'])
        @type identifiers: iterable of string

        @param extra: A disambiguating value (eg. "Drosophila melanogaster")
        @type extra: String

        @param case_sensitive: Whether to treat IDs case sensitively.
        @type case_sensitive: Boolean

        @param wildcards: Whether or not to interpret wildcards (eg: "eve*")
        @type wildcards: Boolean

        @return: {idresolution.Job} The job.
        """
        if self.version < 10:
            raise ServiceError("This feature requires API version 10+")
        if not data_type:
            raise ServiceError("No data-type supplied")
        if not identifiers:
            raise ServiceError("No identifiers supplied")

        data = json.dumps(
            {
                "type": data_type,
                "identifiers": list(identifiers),
                "extra": extra,
                "caseSensitive": case_sensitive,
                "wildCards": wildcards,
            }
        )
        text = self.opener.post_content(self.root + self.IDS_PATH, data, InterMineURLOpener.JSON)
        ret = json.loads(text)
        if ret["error"] is not None:
            raise ServiceError(ret["error"])
        if ret["uid"] is None:
            raise Exception("No uid found in " + ret)

        return idresolution.Job(self, ret["uid"])

    def flush(self):
        """
        Flushes any cached data.
        """
        self._list_manager.delete_temporary_lists()
        self._list_manager = ListManager(self)
        self._templates = None
        self._all_templates = None
        self._all_templates_names = None
        self._model = None
        self._version = None
        self._release = None
        self._widgets = None

    @property
    def templates(self):
        """
        The dictionary of templates from the webservice
        ===============================================

        Service.templates S{->} dict(intermine314.query.Template|string)

        For efficiency's sake, Templates are not parsed until
        they are required, and until then they are stored as XML
        strings. It is recommended that in most cases you would want
        to use L{Service.get_template}.

        You can use this property however to test for template existence though::

         if name in service.templates:
            template = service.get_template(name)

        @rtype: dict

        """
        if self._templates is None:
            templates = {}
            dom = self._get_xml(self.TEMPLATES_PATH)
            for e in dom.getElementsByTagName("template"):
                name = e.getAttribute("name")
                if name in templates:
                    raise ServiceError("Two templates with same name: " + name)
                else:
                    templates[name] = e.toxml()
            self._templates = templates
        return self._templates

    @property
    def all_templates(self):
        """
        The dictionary of templates by users from the webservice
        ========================================================

        Service.all_templates S{->} dict(string|string)

        You need to be authenticated as admin.

        For efficiency's sake, Templates are not parsed until
        they are required, and until then they are stored as XML
        strings. It is recommended that in most cases you would want
        to use L{Service.get_template}.

        You can use this property however to test for template existence::

         if name in service.templates:
           template = service.get_template(name)

        @rtype: dict

        """
        if self._all_templates is None:
            all_templates = {}
            dom = self._get_xml(self.ALL_TEMPLATES_PATH)
            for e in dom.getElementsByTagName("template"):
                user = e.getAttribute("userName")
                name = e.getAttribute("name")
                if user in all_templates:
                    templates = all_templates[user]
                    templates[name] = e.toxml()
                else:
                    templates = {}
                    templates[name] = e.toxml()
                    all_templates[user] = templates
            self._all_templates = all_templates
        return self._all_templates

    @property
    def all_templates_names(self):
        """
        The dictionary of templates names by users from the webservice
        =============================================================

        Service.all_templates_names S{->} dict(string|array)

        You need to be authenticated as admin.

        Example::
          allTemplatesNames = service.all_templates_names
          for user in allTemplatesNames:
            userTemplatesNames = allTemplatesNames[user]
            for templateName in userTemplatesNames:
              template = service.get_template_by_user(templateName, user)

        @rtype: dict

        """
        if self._all_templates_names is None:
            all_templates_names = {}
            dom = self._get_xml(self.ALL_TEMPLATES_PATH)
            for e in dom.getElementsByTagName("template"):
                user = e.getAttribute("userName")
                name = e.getAttribute("name")
                if user in all_templates_names:
                    all_templates_names[user].append(name)
                else:
                    templates_names = []
                    templates_names.append(name)
                    all_templates_names[user] = templates_names
            self._all_templates_names = all_templates_names
        return self._all_templates_names

    @property
    def model(self):
        """
        The data model for the webservice you are querying
        ==================================================

        Service.model S{->} L{intermine314.model.Model}

        This is used when constructing queries to provide them
        with information on the structure of the data model
        they are accessing. You are very unlikely to want to
        access this object directly.

        raises ModelParseError: if the model cannot be read

        @rtype: L{intermine314.model.Model}

        """
        if self._model is None:
            model_xml = self.opener.read(self.root + self.MODEL_PATH)
            self._model = Model(model_xml, self)
        return self._model

    def get_results(self, path, params, rowformat, view, cld=None):
        """
        Return an Iterator over the rows of the results
        ===============================================

        This method is called internally by the query objects
        when they are called to get results. You will not
        normally need to call it directly

        @param path: The resource path (eg: "/query/results")
        @type path: string
        @param params: The query parameters for this request as a dictionary
        @type params: dict
        @param rowformat: One of "rr", "object", "count", "dict", "list", "tsv", "csv", "jsonrows", "jsonobjects"
        @type rowformat: string
        @param view: The output columns
        @type view: list

        @raise WebserviceError: for failed requests

        @return: L{intermine314.webservice.ResultIterator}
        """
        return ResultIterator(self, path, params, rowformat, view, cld)

    @requires_version(9)
    def register(self, username, password):
        """
        Register a new user with this service.
        =======================================

        @return {Service} an authenticated service.
        """
        if isinstance(username, bytes):
            username = ensure_str(username)
        if isinstance(password, bytes):
            password = ensure_str(password)
        payload = urlencode({"name": username, "password": password})
        registrar = Service(self.root)
        resp = registrar._get_json(self.USERS_PATH, payload=payload)
        token = resp["user"]["temporaryToken"]
        return Service(self.root, token=token)

    @requires_version(16)
    def get_deregistration_token(self, validity=300):
        if validity < 1 or validity > 24 * 60 * 60:
            raise ValueError("Validity not a reasonable value: 1ms - 2hrs")
        params = urlencode({"validity": str(validity)})
        resp = self._get_json("/user/deregistration", payload=params)
        return resp["token"]

    @requires_version(16)
    def deregister(self, deregistration_token):
        """
        Remove a User from the service
        ==============================

        @param deregistration_token A token to prove you really want to do this

        @return string All the user's data.
        """
        if "uuid" in deregistration_token:
            deregistration_token = deregistration_token["uuid"]

        path = self.root + "/user"
        params = {"deregistrationToken": deregistration_token, "format": "xml"}
        uri = path + "?" + urlencode(params)
        self.flush()
        userdata = self.opener.delete(uri)
        return userdata
