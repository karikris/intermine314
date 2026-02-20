import codecs
import json
import logging
import weakref
from contextlib import closing, contextmanager
from urllib.parse import urlencode

from intermine314.config.constants import DEFAULT_LIST_ENTRIES_BATCH_SIZE
from intermine314.service.errors import WebserviceError
from intermine314.lists.list import List


class ListManager:
    """
    A Class for Managing List Content and Operations
    ================================================

    This class provides methods to manage list contents and operations.

    This class may be called itself, but all the useful methods it has
    are also available on the Service object, which delegates to this class,
    while other methods are more coneniently accessed through the list objects
    themselves.

    NB: The methods for creating lists can conflict in threaded applications,
    if two threads are each allocated the same unused list name. You are
    strongly advised to use locks to synchronise any list creation requests
    (create_list, or intersect, union, subtract, diff) unless you are choosing
    your own names each time and are confident that these will not conflict.
    """

    LOG = logging.getLogger(__name__)
    DEFAULT_LIST_NAME = "my_list"
    DEFAULT_DESCRIPTION = "List created with Python client library"
    DEFAULT_UPLOAD_CHUNK_SIZE = int(DEFAULT_LIST_ENTRIES_BATCH_SIZE)

    INTERSECTION_PATH = "/lists/intersect/json"
    UNION_PATH = "/lists/union/json"
    DIFFERENCE_PATH = "/lists/diff/json"
    SUBTRACTION_PATH = "/lists/subtract/json"

    def __init__(self, service):
        self.service = weakref.proxy(service)
        self.lists = None
        self._lists_cache_valid = False
        self._refresh_lists_calls = 0
        self._bulk_mutation_depth = 0
        self._bulk_refresh_pending = False
        self._temp_lists = set()

    def refresh_lists(self):
        """
        Update the list information with the latest details from the server
        """

        self.lists = {}
        url = self.service.root + self.service.LIST_PATH
        data = self.service.opener.read(url)
        list_info = json.loads(data)
        if not list_info.get("wasSuccessful"):
            raise ListServiceError(list_info.get("error"))
        raw_lists = list_info.get("lists") or []
        payload_bytes = len(data) if isinstance(data, (bytes, bytearray)) else len(str(data))
        self._refresh_lists_calls += 1
        self._lists_cache_valid = True
        self._bulk_refresh_pending = False
        self.LOG.debug(
            "refreshed lists catalog: call=%d payload_bytes=%d list_count=%d",
            self._refresh_lists_calls,
            payload_bytes,
            len(raw_lists),
        )
        for l in raw_lists:
            l = ListManager.safe_dict(l)
            self.lists[l["name"]] = List(service=self.service, manager=self, **l)

    def _ensure_lists_for_iteration(self):
        if self.lists is None or not self._lists_cache_valid:
            self.refresh_lists()

    def _mark_cache_invalid(self):
        self._lists_cache_valid = False
        if self._bulk_mutation_depth > 0:
            self._bulk_refresh_pending = True

    @contextmanager
    def bulk_mutation(self):
        self._bulk_mutation_depth += 1
        try:
            yield self
        finally:
            self._bulk_mutation_depth -= 1
            if self._bulk_mutation_depth == 0 and self._bulk_refresh_pending:
                self.refresh_lists()

    @staticmethod
    def safe_dict(d):
        """Recursively clone json structure with UTF-8 dictionary keys"""

        if isinstance(d, dict):
            return dict(d)
        return d

    def get_list(self, name):
        """Return a list from the service by name, if it exists"""

        if self.lists is None:
            self.refresh_lists()
        if not self._lists_cache_valid and name not in self.lists:
            self.refresh_lists()
        return self.lists.get(name)

    def l(self, name):
        """Alias for get_list"""

        return self.get_list(name)

    def get_all_lists(self):
        """Get all the lists on a webservice"""

        self._ensure_lists_for_iteration()
        return self.lists.values()

    def get_all_list_names(self):
        """Get all the names of the lists in a particular webservice"""

        self._ensure_lists_for_iteration()
        return self.lists.keys()

    def get_list_count(self):
        """
        Return the number of lists accessible at the given webservice.
        This number will vary depending on who you are authenticated as.
        """

        return len(self.get_all_list_names())

    def get_unused_list_name(self):
        """
        Get an unused list name
        =======================

        This method returns a new name that does not conflict
        with any currently existing list name.

        The list name is only guaranteed to be unused at the time
        of allocation.
        """

        self._ensure_lists_for_iteration()
        list_names = self.get_all_list_names()
        self.LOG.debug("allocating temporary list name from existing_count=%d", len(list_names))
        counter = 1
        name = f"{self.DEFAULT_LIST_NAME}_{counter}"
        while name in list_names:
            counter += 1
            name = f"{self.DEFAULT_LIST_NAME}_{counter}"
        self._temp_lists.add(name)
        return name

    def _fetch_bytes(self, url, payload=None):
        with closing(self.service.opener.open(url, payload)) as response:
            return response.read()

    def _iter_payload_chunks(self, values, *, quote_values, chunk_size=None):
        if chunk_size is None:
            chunk_size = self.DEFAULT_UPLOAD_CHUNK_SIZE
        chunk_size = max(1, int(chunk_size))
        current = []
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            current.append(f'"{text}"' if quote_values else text)
            if len(current) >= chunk_size:
                payload = "\n".join(current).encode("utf-8")
                yield payload, len(current)
                current = []
        if current:
            payload = "\n".join(current).encode("utf-8")
            yield payload, len(current)

    def _create_list_stub(self, *, name, list_type, description=None, tags=None, size=0):
        return List(
            service=self.service,
            manager=self,
            name=name,
            title=name,
            description=description,
            type=list_type or "Object",
            size=size,
            tags=list(tags or []),
        )

    def _coerce_size(self, response_data):
        for key in ("size", "listSize", "newSize", "count"):
            if key in response_data:
                try:
                    return int(response_data[key])
                except Exception:
                    return None
        return None

    def _upsert_cached_list(
        self,
        *,
        name,
        expected_type=None,
        expected_description=None,
        expected_tags=None,
        response_data=None,
    ):
        if self.lists is None:
            self.lists = {}
        im_list = self.lists.get(name)
        if im_list is None and expected_type:
            size = self._coerce_size(response_data or {}) or 0
            im_list = self._create_list_stub(
                name=name,
                list_type=expected_type,
                description=expected_description,
                tags=expected_tags,
                size=size,
            )
            self.lists[name] = im_list
        if im_list is None:
            return None
        parsed_size = self._coerce_size(response_data or {})
        if parsed_size is not None:
            im_list._size = parsed_size
        if expected_description is not None and getattr(im_list, "_description", None) is None:
            im_list._description = expected_description
        if expected_tags:
            im_list._tags = frozenset(expected_tags)
        return im_list

    def _submit_create_payload(self, *, name, list_type, description, tags, add, payload):
        uri = self.service.root + self.service.LIST_CREATION_PATH
        query_form = {
            "name": name,
            "type": list_type,
            "description": description,
            "tags": ";".join(tags),
        }
        if add:
            query_form["add"] = [x.lower() for x in add if x]
        uri += "?" + urlencode(query_form, doseq=True)
        return self.service.opener.post_plain_text(uri, payload)

    def _submit_append_payload(self, *, name, payload):
        uri = self.service.root + self.service.LIST_APPENDING_PATH
        uri += "?" + urlencode({"name": name})
        return self.service.opener.post_plain_text(uri, payload)

    def _submit_chunked_identifier_payloads(
        self,
        *,
        payload_iter,
        name,
        list_type,
        description,
        tags,
        add,
        append_only=False,
    ):
        chunks_sent = 0
        ids_total = 0
        current = None
        for payload, chunk_count in payload_iter:
            chunks_sent += 1
            ids_total += int(chunk_count)
            if append_only:
                body = self._submit_append_payload(name=name, payload=payload)
            elif chunks_sent == 1:
                body = self._submit_create_payload(
                    name=name,
                    list_type=list_type,
                    description=description,
                    tags=tags,
                    add=add,
                    payload=payload,
                )
            else:
                body = self._submit_append_payload(name=name, payload=payload)

            current = self.parse_list_upload_response(
                body,
                expected_name=name,
                expected_type=list_type,
                expected_description=description,
                expected_tags=tags,
            )

        if chunks_sent == 0:
            raise ValueError("Lists must have one or more elements")

        self.LOG.debug(
            "submitted list identifier payloads: list_name=%s ids_total=%d chunks_sent=%d chunk_size=%d append_only=%s",
            name,
            ids_total,
            chunks_sent,
            self.DEFAULT_UPLOAD_CHUNK_SIZE,
            bool(append_only),
        )
        return current

    def append_to_list_content(
        self,
        *,
        list_name,
        content,
        list_type="",
        description=None,
        tags=None,
    ):
        expected_tags = list(tags or [])
        try:  # Queryable append
            q = self._get_listable_query(content)
        except AttributeError:
            q = None
        if q is not None:
            uri = q.get_list_append_uri()
            params = q.to_query_params()
            params["listName"] = list_name
            params["path"] = None
            form = urlencode(params)
            data = self._fetch_bytes(uri, form)
            return self.parse_list_upload_response(
                data,
                expected_name=list_name,
                expected_type=list_type,
                expected_description=description,
                expected_tags=expected_tags,
            )

        try:
            stream = iter(content)
            if hasattr(content, "read"):  # File-like and iterable.
                return self._submit_chunked_identifier_payloads(
                    payload_iter=self._iter_payload_chunks(stream, quote_values=False),
                    name=list_name,
                    list_type=list_type,
                    description=description,
                    tags=expected_tags,
                    add=[],
                    append_only=True,
                )
        except (AttributeError, TypeError):
            pass

        try:
            ids = content.read()
        except AttributeError:
            try:
                with closing(codecs.open(content, "r", "UTF-8")) as stream:
                    return self._submit_chunked_identifier_payloads(
                        payload_iter=self._iter_payload_chunks(stream, quote_values=False),
                        name=list_name,
                        list_type=list_type,
                        description=description,
                        tags=expected_tags,
                        add=[],
                        append_only=True,
                    )
            except (TypeError, IOError):
                try:
                    ids = content.strip()
                except AttributeError:
                    try:
                        idents = iter(content)
                    except TypeError:
                        raise TypeError("Cannot append list content from " + repr(content))
                    return self._submit_chunked_identifier_payloads(
                        payload_iter=self._iter_payload_chunks(idents, quote_values=True),
                        name=list_name,
                        list_type=list_type,
                        description=description,
                        tags=expected_tags,
                        add=[],
                        append_only=True,
                    )

        if ids is None or not str(ids).strip():
            raise ValueError("Lists must have one or more elements")
        body = self._submit_append_payload(name=list_name, payload=ids)
        return self.parse_list_upload_response(
            body,
            expected_name=list_name,
            expected_type=list_type,
            expected_description=description,
            expected_tags=expected_tags,
        )

    def _get_listable_query(self, queryable):
        q = queryable.to_query()
        if not q.views:
            q.add_view(q.root.name + ".id")
        else:
            # Check to see if the class of the selected items is
            # unambiguous

            up_to_attrs = set(v[0 : v.rindex(".")] for v in q.views)
            if len(up_to_attrs) == 1:
                q.select(up_to_attrs.pop() + ".id")
        return q

    def _create_list_from_queryable(
        self,
        queryable,
        name,
        description,
        tags,
    ):
        q = self._get_listable_query(queryable)
        expected_type = getattr(getattr(q, "root", None), "name", "") or ""
        uri = q.get_list_upload_uri()
        params = q.to_query_params()
        params["listName"] = name
        params["description"] = description
        params["tags"] = ";".join(tags)
        form = urlencode(params)
        data = self._fetch_bytes(uri, form)
        return self.parse_list_upload_response(
            data,
            expected_name=name,
            expected_type=expected_type,
            expected_description=description,
            expected_tags=tags,
        )

    def create_list(
        self,
        content,
        list_type="",
        name=None,
        description=None,
        tags=None,
        add=None,
        organism=None,
    ):
        """
        Create a new list in the webservice
        ===================================

        If no name is given, the list will be considered to be a temporary
        list, and will be automatically deleted when the program ends.
        To prevent this happening, give the list a name, either on creation,
        or by renaming it.

        This method is not thread safe for anonymous lists - it will need
        synchronisation with locks if you intend to create lists with multiple
        threads in parallel.

        @param content: The source of the identifiers for this list.
        This can be:
                  * A string with white-space separated terms.
                  * The name of a file that contains the terms.
                  * A file-handle like thing (something with a 'read' method)
                  * An iterable of identifiers
                  * A query with a single column.
                  * Another list.
        @param list_type: The type of objects to include in the list.
                          This parameter is not required if the content
                          parameter implicitly includes the type
                          (as queries and lists do).
        @param name: The name for the new list.
                     If none is provided one will be generated, and the
                     list will be deleted when the list manager exits context.
        @param description: A description for the list
                            (free text, default = None)
        @param tags: A set of strings to use as tags (default = [])
        @param add: The issues groups that can be treated as matches.
                    This should be a collection of strings naming issue groups
                    that would otherwise be ignored, but in this case will be
                    added to the list. The available groups are:
                      * DUPLICATE - More than one match was found.
                      * WILDCARD - A wildcard match was made.
                      * TYPE_CONVERTED - A match was found, but in another type
                                         (eg. found a protein
                                         and we could convert it to a gene).
                      * OTHER - other issue types
                      * :all - All issues should be considered acceptable.
                    This only makes sense with text uploads
                    - it is not required (or used) when
                    the content is a list or a query.
        @param organism: organism name

        @rtype: intermine314.lists.List
        """

        if description is None:
            description = self.DEFAULT_DESCRIPTION
        if tags is None:
            tags = []
        if add is None:
            add = []

        if name is None:
            name = self.get_unused_list_name()

        item_content = content
        ids = None

        if organism:
            query = self.service.new_query(list_type)
            if isinstance(organism, list):
                query.add_constraint("{0}.organism.name".format(list_type), "ONE OF", organism)
            else:
                query.add_constraint("organism", "LOOKUP", organism)
            if isinstance(item_content, list):
                query.add_constraint("symbol", "ONE OF", item_content)
            item_content = query

        try:  # Queryable
            return self._create_list_from_queryable(item_content, name, description, tags)
        except AttributeError:
            pass

        try:
            stream = iter(item_content)
            if hasattr(item_content, "read"):  # File-like and iterable.
                return self._submit_chunked_identifier_payloads(
                    payload_iter=self._iter_payload_chunks(stream, quote_values=False),
                    name=name,
                    list_type=list_type,
                    description=description,
                    tags=tags,
                    add=add,
                )
        except (AttributeError, TypeError):
            pass

        try:
            ids = item_content.read()  # File-like fallback
        except AttributeError:
            try:
                with closing(codecs.open(item_content, "r", "UTF-8")) as stream:  # File path
                    return self._submit_chunked_identifier_payloads(
                        payload_iter=self._iter_payload_chunks(stream, quote_values=False),
                        name=name,
                        list_type=list_type,
                        description=description,
                        tags=tags,
                        add=add,
                    )
            except (TypeError, IOError):
                try:
                    ids = item_content.strip()  # Stringy thing
                except AttributeError:
                    try:  # Iterable of identifiers
                        idents = iter(item_content)
                    except TypeError:
                        raise TypeError("Cannot create list from " + repr(item_content))
                    return self._submit_chunked_identifier_payloads(
                        payload_iter=self._iter_payload_chunks(idents, quote_values=True),
                        name=name,
                        list_type=list_type,
                        description=description,
                        tags=tags,
                        add=add,
                    )

        if ids is None or not str(ids).strip():
            raise ValueError("Lists must have one or more elements")

        body = self._submit_create_payload(
            name=name,
            list_type=list_type,
            description=description,
            tags=tags,
            add=add,
            payload=ids,
        )
        return self.parse_list_upload_response(
            body,
            expected_name=name,
            expected_type=list_type,
            expected_description=description,
            expected_tags=tags,
        )

    def parse_list_upload_response(
        self,
        response,
        *,
        expected_name=None,
        expected_type=None,
        expected_description=None,
        expected_tags=None,
    ):
        """
        Intepret the response from the webserver to a list request,
        and return the List it describes
        """

        response_data = self._body_to_json(response)
        list_name = response_data.get("listName") or expected_name
        if not list_name:
            self._mark_cache_invalid()
            if self._bulk_mutation_depth == 0:
                self.refresh_lists()
            raise ListServiceError("listName missing from list upload response")
        unmatched = response_data.get("unmatchedIdentifiers") or []
        self.LOG.debug(
            "parsed list upload response: list_name=%s unmatched_count=%d response_keys=%s",
            list_name,
            len(unmatched),
            sorted(response_data.keys()),
        )
        im_list = self._upsert_cached_list(
            name=list_name,
            expected_type=expected_type,
            expected_description=expected_description,
            expected_tags=expected_tags,
            response_data=response_data,
        )
        if im_list is None:
            self._mark_cache_invalid()
            if self._bulk_mutation_depth == 0:
                self.refresh_lists()
                im_list = self.get_list(list_name)
        if im_list is None:
            raise ListServiceError("Could not resolve list from response: " + str(list_name))
        im_list._add_failed_matches(unmatched)
        return im_list

    def delete_lists(self, lists):
        """Delete the given lists from the webserver"""

        known_names = None
        if self.lists is not None and self._lists_cache_valid:
            known_names = set(self.lists.keys())
        for l in lists:
            if isinstance(l, List):
                name = l.name
            else:
                name = str(l)
            if known_names is not None and name not in known_names:
                self.LOG.debug("%s does not exist in cache - skipping delete", name)
                continue
            self.LOG.debug("deleting %s", name)
            uri = self.service.root + self.service.LIST_PATH
            query_form = {"name": name}
            uri += "?" + urlencode(query_form)
            response = self.service.opener.delete(uri)
            response_data = self._body_to_json(response)
            if not response_data.get("wasSuccessful"):
                self._mark_cache_invalid()
                raise ListServiceError(response_data.get("error"))
            if self.lists is not None:
                self.lists.pop(name, None)
            self._temp_lists.discard(name)
            if known_names is not None:
                known_names.discard(name)

    def remove_tags(self, to_remove_from, tags):
        """
        Add the tags to the given list
        ==============================

        Returns the current tags of this list.
        """

        uri = self.service.root + self.service.LIST_TAG_PATH
        form = {"name": to_remove_from.name, "tags": ";".join(tags)}
        uri += "?" + urlencode(form)
        body = self.service.opener.delete(uri)
        return self._body_to_json(body)["tags"]

    def add_tags(self, to_tag, tags):
        """
        Add the tags to the given list
        ==============================

        Returns the current tags of this list.
        """

        uri = self.service.root + self.service.LIST_TAG_PATH
        form = {"name": to_tag.name, "tags": ";".join(tags)}
        body = self._fetch_bytes(uri, urlencode(form))
        return self._body_to_json(body)["tags"]

    def get_tags(self, im_list):
        """
        Get the up-to-date set of tags for a given list
        ===============================================

        Returns the current tags of this list.
        """

        uri = self.service.root + self.service.LIST_TAG_PATH
        form = {"name": im_list.name}
        uri += "?" + urlencode(form)
        body = self._fetch_bytes(uri)
        return self._body_to_json(body)["tags"]

    def _body_to_json(self, body):
        if isinstance(body, (bytes, bytearray)):
            text = body.decode("utf8")
        elif isinstance(body, str):
            text = body
        else:
            text = str(body)
        try:
            data = json.loads(text)
        except ValueError:
            preview = text[:512] + ("...<truncated>" if len(text) > 512 else "")
            raise ListServiceError("Error parsing response: " + preview)
        if not data.get("wasSuccessful"):
            raise ListServiceError(data.get("error"))
        return data

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        traceback,
    ):
        self.LOG.debug("Exiting context - deleting temporary_lists_count=%d", len(self._temp_lists))
        self.delete_temporary_lists()

    def delete_temporary_lists(self):
        """
        Delete all the lists considered temporary (those created without names)
        """

        if self._temp_lists:
            self.delete_lists(self._temp_lists)
            self._temp_lists = set()

    def intersect(
        self,
        lists,
        name=None,
        description=None,
        tags=None,
    ):
        """
        Calculate the intersection of a given set of lists, and return the
        list representing the result
        """

        return self._do_operation(
            self.INTERSECTION_PATH,
            "Intersection",
            lists,
            name,
            description,
            list(tags or []),
        )

    def union(
        self,
        lists,
        name=None,
        description=None,
        tags=None,
    ):
        """
        Calculate the union of a given set of lists,
        and return the list representing the result
        """

        return self._do_operation(
            self.UNION_PATH,
            "Union",
            lists,
            name,
            description,
            list(tags or []),
        )

    def xor(
        self,
        lists,
        name=None,
        description=None,
        tags=None,
    ):
        """
        Calculate the symmetric difference of a given set of lists,
        and return the list representing the result
        """

        return self._do_operation(
            self.DIFFERENCE_PATH,
            "Difference",
            lists,
            name,
            description,
            list(tags or []),
        )

    def subtract(
        self,
        lefts,
        rights,
        name=None,
        description=None,
        tags=None,
    ):
        """
        Calculate the subtraction of rights from lefts,
        and return the list representing the result
        """

        left_names = self.make_list_names(lefts)
        right_names = self.make_list_names(rights)
        expected_type = self._infer_list_type(lefts, rights)
        tags = list(tags or [])
        if description is None:
            description = "Subtraction of " + " and ".join(right_names) + " from " + " and ".join(left_names)
        if name is None:
            name = self.get_unused_list_name()
        uri = self.service.root + self.SUBTRACTION_PATH
        uri += "?" + urlencode(
            {
                "name": name,
                "description": description,
                "references": ";".join(left_names),
                "subtract": ";".join(right_names),
                "tags": ";".join(tags),
            }
        )
        data = self._fetch_bytes(uri)
        return self.parse_list_upload_response(
            data,
            expected_name=name,
            expected_type=expected_type,
            expected_description=description,
            expected_tags=tags,
        )

    def _do_operation(
        self,
        path,
        operation,
        lists,
        name,
        description,
        tags,
    ):
        list_names = self.make_list_names(lists)
        expected_type = self._infer_list_type(lists)
        if description is None:
            description = operation + " of " + " and ".join(list_names)
        if name is None:
            name = self.get_unused_list_name()
        uri = self.service.root + path
        uri += "?" + urlencode(
            {
                "name": name,
                "lists": ";".join(list_names),
                "description": description,
                "tags": ";".join(tags),
            }
        )
        data = self._fetch_bytes(uri)
        return self.parse_list_upload_response(
            data,
            expected_name=name,
            expected_type=expected_type,
            expected_description=description,
            expected_tags=tags,
        )

    @staticmethod
    def _infer_list_type(*groups):
        for group in groups:
            for item in group:
                list_type = getattr(item, "list_type", None)
                if list_type:
                    return str(list_type)
        return ""

    def make_list_names(self, lists):
        """Turn a list of things into a list of list names"""

        list_names = []
        for l in lists:
            try:
                t = l.list_type
                list_names.append(l.name)
            except AttributeError:
                try:
                    m = l.model
                    list_names.append(self.create_list(l).name)
                except AttributeError:
                    list_names.append(str(l))

        return list_names


class ListServiceError(WebserviceError):
    """Errors thrown when something goes wrong with list requests"""

    pass
