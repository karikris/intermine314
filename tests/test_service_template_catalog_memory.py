from __future__ import annotations

from intermine314.service.service import Service


class _FakeListManager:
    def delete_temporary_lists(self):
        return None


class _BytesResponse:
    def __init__(self, payload):
        self._payload = payload
        self.closed = False

    def read(self):
        return self._payload

    def close(self):
        self.closed = True


class _CatalogOpener:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []
        self.responses = []

    def open(self, url, *_args, **_kwargs):
        self.calls.append(url)
        resp = _BytesResponse(self._payload)
        self.responses.append(resp)
        return resp


class _NoNetworkOpener:
    def __init__(self):
        self.calls = []

    def open(self, url, *_args, **_kwargs):
        self.calls.append(url)
        raise AssertionError("network must not be called")


def _make_service(opener):
    service = Service.__new__(Service)
    service.root = "https://example.org/service"
    service.opener = opener
    service._templates = None
    service._all_templates = None
    service._all_templates_names = None
    service._model = None
    service._version = None
    service._release = None
    service._widgets = None
    service._list_manager = _FakeListManager()
    return service


def _build_large_all_templates_xml(template_count=400, body_chars=256):
    nodes = []
    for idx in range(template_count):
        user = f"user_{idx % 7}"
        name = f"Template_{idx:04d}"
        body = "X" * body_chars
        nodes.append(
            f'<template name="{name}" userName="{user}">'
            f'<query name="{name}" model="genomic" view="Gene.symbol">{body}</query>'
            "</template>"
        )
    xml = "<templates>" + "".join(nodes) + "</templates>"
    return xml.encode("utf-8")


def test_all_templates_catalog_cached_as_compact_bytes():
    payload = _build_large_all_templates_xml(template_count=300, body_chars=320)
    opener = _CatalogOpener(payload)
    service = _make_service(opener)

    all_templates = service.all_templates

    entry_count = sum(len(user_templates) for user_templates in all_templates.values())
    stored_bytes = sum(
        len(xml_blob)
        for user_templates in all_templates.values()
        for xml_blob in user_templates.values()
        if isinstance(xml_blob, (bytes, bytearray))
    )

    assert entry_count == 300
    assert stored_bytes > 0
    assert stored_bytes <= int(len(payload) * 1.25)
    assert all(
        isinstance(xml_blob, (bytes, bytearray))
        for user_templates in all_templates.values()
        for xml_blob in user_templates.values()
    )

    assert opener.calls == ["https://example.org/service/alltemplates"]
    assert len(opener.responses) == 1
    assert opener.responses[0].closed is True


def test_all_templates_names_derived_from_cache_without_network():
    opener = _NoNetworkOpener()
    service = _make_service(opener)
    service._all_templates = {
        "alice": {
            "A": b"<template name='A' userName='alice'/>",
            "B": b"<template name='B' userName='alice'/>",
        },
        "bob": {
            "X": b"<template name='X' userName='bob'/>",
        },
    }

    names = service.all_templates_names

    assert names == {"alice": ["A", "B"], "bob": ["X"]}
    assert opener.calls == []


def test_all_templates_names_reuses_cached_catalog_no_extra_fetch():
    payload = _build_large_all_templates_xml(template_count=25, body_chars=32)
    opener = _CatalogOpener(payload)
    service = _make_service(opener)

    _ = service.all_templates
    first_call_count = len(opener.calls)
    names = service.all_templates_names

    assert first_call_count == 1
    assert len(opener.calls) == 1
    assert sum(len(v) for v in names.values()) == 25

