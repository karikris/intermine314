def query_to_xml(query) -> str:
    return query.to_xml()


def template_to_xml(template) -> str:
    return template.to_xml()


def load_query_xml(service, xml: str):
    return service.load_query(xml)
