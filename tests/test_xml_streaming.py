from io import BytesIO
from xml.parsers.expat import ExpatError

import pytest

from intermine314.model import Model, ModelError, ModelParseError
from intermine314.service.xmlstream import selected_elements, xml_events
from tests.fixtures.compatibility import fixture_bytes


class ShortReads(BytesIO):
    def read(self, size=-1):
        assert 0 < size <= 65536
        return super().read(min(size, 1))


def test_model_sized_reads_handle_split_unicode_and_keep_borrowed_open():
    source = ShortReads('<model name="Müller" package="p"><class name="人"><attribute name="名" type="String"/></class></model>'.encode())
    model = Model(source)
    assert model.name == 'Müller' and model.get_class('人').get_field('名').type_name == 'String'
    assert not source.closed


def test_model_parse_errors_retain_full_source_and_expat_cause():
    text = '<model name="m" package="p">' + '<class name="A"/>' + ' ' * 70000 + '<broken>'
    source = BytesIO(text.encode())
    with pytest.raises(ModelParseError) as exc:
        Model(source)
    assert exc.value.source == text
    assert isinstance(exc.value.cause, ExpatError)
    assert not source.closed


def test_ancestry_memoization_preserves_diamond_order_duplicates_and_mutability():
    model = Model('<model name="m" package="p"><class name="A"/><class name="B" extends="A"/><class name="C" extends="A"/><class name="D" extends="B C"/></model>')
    d = model.get_class('D')
    assert [parent.name for parent in d.parent_classes] == ['B', 'C', 'A', 'A']
    assert len(model._ancestry_cache) == 4
    assert d.field_dict['id'] is model.get_class('A').field_dict['id']
    model.get_class('A').parents.append('D')
    with pytest.raises(ModelError, match='Inheritance cycle'):
        model.to_ancestry(d)
    with pytest.raises(ModelError, match='Inheritance cycle'):
        model.vivify()


def test_model_and_template_paths_do_not_construct_complete_doms(monkeypatch):
    import xml.dom.minidom

    def forbidden(*args, **kwargs):
        pytest.fail('Metadata discovery must not build a full DOM')
    monkeypatch.setattr(xml.dom.minidom, 'parse', forbidden)
    monkeypatch.setattr(xml.dom.minidom, 'parseString', forbidden)
    assert Model(fixture_bytes('model.xml')).name == 'testmodel'
    elements = list(selected_elements(xml_events(BytesIO(fixture_bytes('templates.xml'))), 'template'))
    assert len(elements) == 12


def test_selected_elements_preserve_nested_order_comments_and_namespace_prefixes():
    from xml.etree.ElementTree import tostring

    xml = b'<root><template name="outer" xmlns:p="urn:p"><!--comment--><?test data?><p:query/><template name="inner"/></template></root>'
    elements = list(selected_elements(xml_events(BytesIO(xml)), 'template'))
    assert [item.get('name') for item in elements] == ['outer', 'inner']
    text = tostring(elements[0], encoding='unicode')
    assert '<!--comment-->' in text and '<?test data?>' in text
    assert 'xmlns:p="urn:p"' in text and '<p:query' in text


def test_invalid_reader_payload_still_raises_public_model_parse_error():
    class InvalidReader:
        calls = 0
        def read(self, size):
            self.calls += 1
            return object() if self.calls == 1 else b''
    with pytest.raises(ModelParseError) as exc:
        Model(InvalidReader())
    assert isinstance(exc.value.cause, TypeError)


def test_complete_document_uses_shared_bounded_events():
    from intermine314.service.xmlstream import xml_document

    stream = ShortReads(b'<?xml version="1.0"?><root><!--c--><?target data?><value><![CDATA[A & B]]></value></root>')
    root = xml_document(stream)
    assert root.tag == 'root' and root.find('value').text == 'A & B'
    assert len(root) == 3 and not stream.closed
    with pytest.raises(ExpatError):
        xml_document(ShortReads(b'<root>'))


def test_query_document_parser_keeps_borrowed_stream_and_direct_value_text(monkeypatch):
    import xml.dom.minidom

    from intermine314.query import Query
    from tests.support.lifecycle import fresh_model

    def forbidden(*args, **kwargs):
        pytest.fail('Query XML loading must use the shared event parser')

    monkeypatch.setattr(xml.dom.minidom, 'parse', forbidden)
    monkeypatch.setattr(xml.dom.minidom, 'parseString', forbidden)
    stream = ShortReads(b'<query view="Employee.name"><constraint path="Employee.name" op="ONE OF"><value><![CDATA[A & B]]><!-- c -->tail</value><value>direct<ignored>nested</ignored>end</value></constraint></query>')
    query = Query.from_xml(stream, fresh_model())
    assert query.get_constraint('A').values == ['A & Btail', 'directend']
    assert not stream.closed
