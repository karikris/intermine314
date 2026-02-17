from __future__ import annotations

from contextlib import closing
from io import BytesIO
from xml.etree import ElementTree as ET

from intermine314.service.errors import ServiceError

_XML_ACCEPT_HEADERS = {"Accept": "application/xml"}
_XML_TAG_TEMPLATE = "template"
_XML_ATTR_NAME = "name"
_XML_ATTR_USERNAME = "userName"


def _decode_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    return str(value)


def _template_query_classes():
    # Preserve compatibility for callers/tests that monkeypatch
    # intermine314.service.service._query_classes.
    from intermine314.service import service as service_module

    return service_module._query_classes()


class TemplateCatalogMixin:
    def _user_template_cache(self, username):
        user_templates = self.all_templates.get(username)
        if isinstance(user_templates, dict):
            return user_templates
        repaired = {}
        self.all_templates[username] = repaired
        return repaired

    @staticmethod
    def _template_tag(tag):
        return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else str(tag)

    def _read_xml_bytes(self, path):
        with closing(self.opener.open(self.root + path, headers=_XML_ACCEPT_HEADERS)) as xml_resp:
            payload = xml_resp.read()
        if isinstance(payload, bytes):
            return payload
        if isinstance(payload, str):
            return payload.encode("utf-8")
        return bytes(payload)

    def _iter_template_nodes(self, payload):
        try:
            parser = ET.iterparse(BytesIO(payload), events=("end",))
            for _event, node in parser:
                if self._template_tag(node.tag) != _XML_TAG_TEMPLATE:
                    continue
                name = node.get(_XML_ATTR_NAME, "")
                if not name:
                    node.clear()
                    continue
                user = node.get(_XML_ATTR_USERNAME, "")
                xml_bytes = ET.tostring(node, encoding="utf-8")
                node.clear()
                yield user, name, xml_bytes
        except ET.ParseError as exc:
            raise ServiceError("Could not parse template catalog XML: %s" % (exc,))

    def _template_source_text(self, source):
        if isinstance(source, memoryview):
            source = source.tobytes()
        if isinstance(source, bytearray):
            source = bytes(source)
        if isinstance(source, bytes):
            return _decode_text(source)
        return source

    def _load_all_templates_catalog(self):
        all_templates = {}
        all_template_names = {}
        payload = self._read_xml_bytes(self.ALL_TEMPLATES_PATH)
        for user, name, xml_bytes in self._iter_template_nodes(payload):
            user_templates = all_templates.setdefault(user, {})
            user_templates[name] = xml_bytes
            all_template_names.setdefault(user, []).append(name)
        self._all_templates = all_templates
        self._all_templates_names = all_template_names

    def _all_template_names_from_cache(self):
        names = {}
        for user, templates in (self._all_templates or {}).items():
            if not isinstance(templates, dict):
                continue
            names[user] = list(templates.keys())
        return names

    def list_templates(self, include=None, exclude=None, limit=None):
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

    def get_template(self, name):
        try:
            template_source = self.templates[name]
        except KeyError:
            raise ServiceError("There is no template called '" + name + "' at this service")
        _, template_class = _template_query_classes()
        if not isinstance(template_source, template_class):
            template_source = template_class.from_xml(self._template_source_text(template_source), self.model, self)
            self.templates[name] = template_source
        return template_source

    def get_template_by_user(self, name, username):
        if username not in self.all_templates:
            raise ServiceError("There is no user called '" + username + "'")
        templates = self._user_template_cache(username)
        try:
            template_source = templates[name]
        except KeyError:
            raise ServiceError(
                "There is no template called '" + name + "' at this service belonging to '" + username + "'"
            )
        _, template_class = _template_query_classes()
        if not isinstance(template_source, template_class):
            template_source = template_class.from_xml(self._template_source_text(template_source), self.model, self)
            template_source.user_name = username
            templates[name] = template_source
        return template_source

    @property
    def templates(self):
        if self._templates is None:
            templates = {}
            payload = self._read_xml_bytes(self.TEMPLATES_PATH)
            for _user, name, xml_bytes in self._iter_template_nodes(payload):
                if name in templates:
                    raise ServiceError("Two templates with same name: " + name)
                templates[name] = xml_bytes
            self._templates = templates
        return self._templates

    @property
    def all_templates(self):
        if self._all_templates is None:
            self._load_all_templates_catalog()
        return self._all_templates

    @property
    def all_templates_names(self):
        if self._all_templates_names is None:
            if self._all_templates is not None:
                self._all_templates_names = self._all_template_names_from_cache()
            else:
                self._load_all_templates_catalog()
        return self._all_templates_names
