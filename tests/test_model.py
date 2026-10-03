"""Standalone descriptors and paths, using the pinned upstream test model."""

import io
from functools import reduce

import pytest

from intermine314.model import (
    Attribute,
    Class,
    CodelessNode,
    Collection,
    Column,
    ComposedClass,
    ConstraintNode,
    ConstraintTree,
    Field,
    Model,
    ModelError,
    ModelParseError,
    Path,
    PathParseError,
    Reference,
)
from tests.fixtures.compatibility import fixture_bytes


@pytest.fixture
def model():
    return Model(fixture_bytes("model.xml"))


def xml_model(classes):
    return f'<model name="test" package="org.test">{classes}</model>'


def test_original_model_class_and_error_contract(model):
    assert len(model.classes) == 19
    for name in ("Employee", "Company", "Department"):
        assert model.get_class(name).name == name
    ceo = model.get_class("Employee.department.company.CEO")
    assert ceo.name == "CEO"
    assert ceo.isa("Employee") and ceo.isa(model.get_class("Employee"))
    assert ceo.isa("Thing") and ceo.isa(ceo)
    assert not ceo.isa("Broke")
    assert [c.name for c in model.to_classes(["Employee", "Company"])] == [
        "Employee",
        "Company",
    ]
    for name, message in (
        ("Foo", "'Foo' is not a class in this model"),
        ("Employee.name", "'Employee.name' is not a class"),
    ):
        with pytest.raises(ModelError) as caught:
            model.get_class(name)
        assert caught.value.message == message
    with pytest.raises(ModelError) as caught:
        ceo.get_field("foo")
    assert caught.value.message == "There is no field called foo in CEO"


def test_inherited_fields_reverse_links_and_sorted_introspection(model):
    ceo = model.get_class("CEO")
    for name in ("name", "age", "seniority", "address", "department"):
        assert isinstance(ceo.get_field(name), Field)
        assert name in ceo and ceo.get_field(name) in ceo
    assert ceo.get_field("name") is model.get_class("Employable").get_field("name")
    assert {f.name for f in ceo} == set(ceo.field_dict)
    assert [f.name for f in ceo.fields] == sorted(ceo.field_dict)
    dep = model.get_class("Department")
    assert all(isinstance(f, Attribute) for f in dep.attributes)
    assert all(type(f) is Reference for f in dep.references)
    assert all(isinstance(f, Collection) for f in dep.collections)
    employees = dep.get_field("employees")
    department = model.get_class("Employee").get_field("department")
    assert employees.type_class is model.get_class("Employee")
    assert employees.reverse_reference is department
    assert department.reverse_reference is employees
    assert employees.declared_in is dep
    assert (
        repr(department)
        == "department is a Department, which links back to this as employees"
    )
    assert (
        repr(employees)
        == "employees is a group of Employee objects, which link back to this as department"
    )
    assert (
        repr(model.get_class("Company").get_field("secretarys"))
        == "secretarys is a group of Secretary objects"
    )
    assert repr(dep.get_field("name")) == "name is a String"
    assert str(dep.get_field("name")) == "name"
    assert [
        f.fieldtype
        for f in (dep.get_field("name"), dep.get_field("company"), employees)
    ] == ["attribute", "reference", "collection"]
    assert (
        repr(dep)
        == "<intermine314.model.Class org.intermine.model.testmodel.Department>"
    )


def test_java_types_and_synthetic_id(model):
    types = model.get_class("Types")
    expected = {
        "name": "String",
        "booleanType": "boolean",
        "floatType": "float",
        "doubleType": "double",
        "shortType": "short",
        "intType": "int",
        "longType": "long",
        "booleanObjType": "Boolean",
        "floatObjType": "Float",
        "doubleObjType": "Double",
        "shortObjType": "Short",
        "intObjType": "Integer",
        "longObjType": "Long",
        "bigDecimalObjType": "BigDecimal",
        "dateObjType": "Date",
        "stringObjType": "String",
        "id": "Integer",
    }
    assert {f.name: f.type_name for f in types.fields} == expected
    assert all(f.type_class is None for f in types.attributes)
    assert model.get_class("Employee").has_id
    simple = model.get_class("SimpleObject")
    assert simple.parents == ["Object"] and simple.parent_classes == []
    assert not simple.has_id and "id" not in simple
    with pytest.raises(Exception, match="Fields should never be directly instantiated"):
        _ = Field("x", "int", types).fieldtype


def test_composed_class_combines_fields_and_all_ancestry(model):
    composed = model.get_class("Employee,Broke")
    assert isinstance(composed, ComposedClass) and isinstance(composed, Class)
    assert composed.name == "Employee_Broke" and composed.has_id
    assert composed.parents == ["Employable", "HasAddress"]
    assert {c.name for c in composed.parent_classes} == {
        "Employee",
        "Broke",
        "Employable",
        "HasAddress",
        "Thing",
    }
    assert composed.get_field("age") is model.get_class("Employee").get_field("age")
    assert composed.get_field("debt") is model.get_class("Broke").get_field("debt")
    assert composed.isa("Thing") and composed.isa("Broke")
    assert not model.get_class("SimpleObject,Broke").has_id


def test_path_descriptor_operations_and_errors(model):
    path = model.make_path("Employee.department.company.name")
    assert isinstance(path, Path)
    assert str(path) == "Employee.department.company.name"
    assert repr(path) == "<intermine314.model.Path: Employee.department.company.name>"
    assert path.root is model.get_class("Employee")
    assert path.end is model.get_class("Company").get_field("name")
    assert path.is_attribute() and not path.is_class() and not path.is_reference()
    assert path.end_class is None and path.get_class() is None
    prefix = path.prefix()
    assert prefix.is_reference() and prefix.end_class is model.get_class("Company")
    assert prefix.append("name") == path
    assert model.make_path("Employee").append("department", "company", "name") == path
    root = model.make_path(model.get_class("Employee"))
    assert root.is_class() and root.end_class is root.root
    assert model.make_path("Department.employees").is_reference()
    with pytest.raises(PathParseError, match="does not have a prefix"):
        root.prefix()
    assert model.validate_path(str(path)) is True
    for bad in ("NoSuchClass", "Employee.noSuchField", "Employee.name.invalid"):
        with pytest.raises(ModelError):
            model.make_path(bad)
        with pytest.raises(ModelError):
            model.validate_path(bad)


def test_subclass_paths_and_independent_defaults(model):
    overrides = {"Department.employees": "Manager"}
    path = model.make_path("Department.employees.title", overrides)
    assert path.end is model.get_class("Manager").get_field("title")
    assert path.prefix().end_class is model.get_class("Manager")
    assert model.make_path("Employee.title", {"Employee": "Manager"}).is_attribute()
    a = model.make_path("Department.employees")
    b = model.make_path("Department.employees", overrides)
    assert a == b and a == str(b)
    assert hash(a) == hash(b) == hash(str(b))
    assert len({a, b}) == 1
    a.subclasses["Employee"] = "Manager"
    assert model.make_path("Employee").subclasses == {}
    first, second = model.column("Employee"), model.column("Employee")
    first._subclasses["Employee"] = "Manager"
    assert second._subclasses == {}


@pytest.mark.parametrize(
    "source_kind",
    [
        "text",
        "bytes",
        "path",
        "path_text",
        "file_url",
        "borrowed_text",
        "borrowed_bytes",
    ],
)
def test_model_sources_and_borrowed_ownership(tmp_path, source_kind):
    payload = fixture_bytes("model.xml")
    path = tmp_path / "model.xml"
    path.write_bytes(payload)
    sources = {
        "text": payload.decode(),
        "bytes": payload,
        "path": path,
        "path_text": str(path),
        "file_url": path.as_uri(),
        "borrowed_text": io.StringIO(payload.decode()),
        "borrowed_bytes": io.BytesIO(payload),
    }
    source = sources[source_kind]
    assert len(Model(source).classes) == 19
    if hasattr(source, "read"):
        assert not source.closed


@pytest.mark.parametrize("payload", [fixture_bytes("model.xml"), b"broken XML"])
def test_model_closes_owned_url_response_on_success_and_failure(monkeypatch, payload):
    response = io.BytesIO(payload)
    calls = []

    def urlopen(source):
        calls.append(source)
        return response

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    if payload == b"broken XML":
        with pytest.raises(ModelParseError):
            Model("https://offline.example/model")
    else:
        assert Model("https://offline.example/model").name == "testmodel"
    assert calls == ["https://offline.example/model"] and response.closed


def test_model_closes_owned_local_stream_and_keeps_borrowed_failure_open(monkeypatch):
    import intermine314.model as module

    owned = io.StringIO("broken XML")
    monkeypatch.setattr(module, "openAnything", lambda source: owned)
    with pytest.raises(ModelParseError):
        Model("model.xml")
    assert owned.closed
    borrowed = io.StringIO("broken XML")
    monkeypatch.setattr(module, "openAnything", lambda source: source)
    with pytest.raises(ModelParseError) as caught:
        Model(borrowed)
    assert not borrowed.closed
    assert (
        caught.value.message == "Error parsing model"
        and caught.value.source == "broken XML"
    )


def test_open_failure_retains_cause_without_unbound_cleanup(monkeypatch):
    import intermine314.model as module

    failure = OSError("cannot open model")

    def fail(source):
        raise failure

    monkeypatch.setattr(module, "openAnything", fail)
    with pytest.raises(ModelParseError) as caught:
        Model("missing.xml")
    assert caught.value.source == "missing.xml" and caught.value.cause is failure
    assert (
        str(caught.value)
        == "'Error parsing model':'missing.xml'OSError('cannot open model')"
    )
    assert str(ModelParseError("bad", "source")) == "'bad':'source'"


@pytest.mark.parametrize(
    "source",
    [
        "foo",
        "<model/>",
        '<model name="a"/>',
        '<root><model name="a" package="b"/><model name="c" package="d"/></root>',
        xml_model("<class/>"),
    ],
)
def test_invalid_models_raise_public_parse_error(source):
    with pytest.raises(ModelParseError) as caught:
        Model(source)
    assert caught.value.message == "Error parsing model"


def test_public_parse_model_can_replace_a_previously_parsed_class():
    model = Model(
        xml_model('<class name="A"><attribute name="before" type="int"/></class>')
    )
    model.parse_model(
        xml_model('<class name="A"><attribute name="after" type="int"/></class>')
    )
    model.vivify()
    assert "after" in model.get_class("A") and "before" not in model.get_class("A")
    with pytest.raises(ModelParseError, match="Duplicate class"):
        Model(xml_model('<class name="A"/><class name="A"/>'))


@pytest.mark.parametrize(
    "classes,match",
    [
        ('<class name="A" extends="A"/>', "cycle"),
        ('<class name="A" extends="B"/><class name="B" extends="A"/>', "cycle"),
        (
            '<class name="A"><reference name="b" referenced-type="Missing"/></class>',
            "Missing",
        ),
        (
            '<class name="A"><reference name="b" referenced-type="B" reverse-reference="missing"/></class><class name="B"/>',
            "missing",
        ),
        (
            '<class name="A"><reference name="b" referenced-type="B" reverse-reference="a"/></class><class name="B"><attribute name="a" type="int"/></class>',
            "reverse",
        ),
        (
            '<class name="A"><attribute name="x" type="int"/></class><class name="B" extends="A"><reference name="x" referenced-type="A"/></class>',
            "inherited",
        ),
    ],
)
def test_invalid_relationships_fail_without_internal_crashes(classes, match):
    with pytest.raises(ModelError, match=match):
        Model(xml_model(classes))


@pytest.mark.parametrize(
    "type_pair",
    [
        ("int", "java.lang.Integer"),
        ("java.lang.Boolean", "boolean"),
        ("float", "java.lang.Float"),
    ],
)
def test_compatible_inherited_boxed_and_primitive_fields_keep_identity(type_pair):
    parent_type, child_type = type_pair
    model = Model(
        xml_model(
            f'<class name="Child" extends="Parent"><attribute name="value" type="{child_type}"/></class><class name="Parent"><attribute name="value" type="{parent_type}"/></class>'
        )
    )
    parent = model.get_class("Parent")
    child = model.get_class("Child")
    assert child.get_field("value") is parent.get_field("value")
    assert child.get_field("value").declared_in is parent


@pytest.mark.parametrize("fieldtype", ["reference", "collection"])
def test_narrowed_inherited_relationships_keep_child_descriptor(fieldtype):
    model = Model(
        xml_model(
            f'<class name="Owner" extends="Base"><{fieldtype} name="target" referenced-type="Subtype"/></class><class name="Base"><{fieldtype} name="target" referenced-type="Target"/></class><class name="Subtype" extends="Target"/><class name="Target"/>'
        )
    )
    field = model.get_class("Owner").get_field("target")
    assert field.type_class is model.get_class("Subtype")
    assert field.declared_in is model.get_class("Owner")


@pytest.mark.parametrize(
    "classes",
    [
        '<class name="A"><reference name="b" referenced-type="B" reverse-reference="other"/></class><class name="B"><reference name="other" referenced-type="C"/></class><class name="C"/>',
        '<class name="Owner" extends="Base"><reference name="target" referenced-type="Subtype" reverse-reference="second"/></class><class name="Base"><reference name="target" referenced-type="Target" reverse-reference="first"/></class><class name="Subtype" extends="Target"><reference name="second" referenced-type="Owner"/></class><class name="Target"><reference name="first" referenced-type="Base"/></class>',
    ],
)
def test_incompatible_reverse_reference_targets_or_inheritance_are_rejected(classes):
    with pytest.raises(ModelError, match="reverse"):
        Model(xml_model(classes))


def test_node_iteration_logic_codes_and_codeless_nodes():
    first = ConstraintNode("Employee.age", ">", 20)
    second = ConstraintNode("Employee.name", "=", "Alice")
    third = ConstraintNode("Employee.fullTime", "=", True)
    tree = (first & second) | third
    assert isinstance(tree, ConstraintTree)
    assert list(tree) == [first, second, third]
    assert first.vargs == ("Employee.age", ">", 20) and first.kwargs == {}
    assert tree.as_logic() == "((A AND B) OR C)"
    assert tree.as_logic(iter(["X", "Y", "Z"])) == "((X AND Y) OR Z)"
    assert tree.as_logic(start="Z") == "((Z AND AA) OR AB)"
    many = reduce(
        lambda a, b: a & b, [ConstraintNode("Employee.age", ">", i) for i in range(28)]
    )
    assert " AND Z) AND AA) AND AB)" in many.as_logic()
    codeless = CodelessNode("Department.employees", "Manager")
    assert codeless.as_logic() == ""
    assert (first & codeless & second).as_logic() == "(A AND B)"
    assert (codeless | codeless).as_logic() == ""
    assert list(first & codeless) == [first, codeless]


class ProtocolQuery:
    def __init__(self, root):
        self.root = root
        self.cols = ()
        self.constraints = None

    def select(self, *cols):
        self.cols = tuple(str(c) for c in cols)
        return self

    def where(self, *args, **kwargs):
        self.constraints = (args, kwargs)
        return self

    def count(self):
        return 2

    def rows(self):
        yield ["Alice"]
        yield ["Bob"]

    def __iter__(self):
        yield {"name": "Alice"}
        yield {"name": "Bob"}


class ProtocolService:
    def __init__(self):
        self.queries = []

    def new_query(self, root):
        query = ProtocolQuery(root)
        self.queries.append(query)
        return query


def test_column_branches_select_where_iteration_and_count():
    service = ProtocolService()
    model = Model(fixture_bytes("model.xml"), service)
    column = model.Employee
    assert isinstance(column, Column)
    assert model.table == model.column
    assert str(model.table("Employee")) == "Employee"
    assert column.department is column.department
    assert column.department._parent is column
    assert str(column.department.company.name) == "Employee.department.company.name"
    assert column.select("name", "age").cols == ("name", "age")
    assert column.select().cols == ("Employee",)
    condition = column.age > 20
    assert condition.vargs == ("Employee.age", ">", 20)
    assert column.where(condition, name="Alice").constraints == (
        (condition,),
        {"name": "Alice"},
    )
    assert column.filter(name="Bob").constraints == ((), {"name": "Bob"})
    assert len(column) == 2
    assert [value for value in column.name] == ["Alice", "Bob"]
    assert [value for value in column] == [{"name": "Alice"}, {"name": "Bob"}]
    for invalid in (lambda: column.nonexistent, lambda: column.name.invalid):
        with pytest.raises(AttributeError):
            invalid()
    assert str(Column(model.make_path("Employee.name"), model)) == "Employee.name"


def test_column_subclass_branch_refresh_and_operator_primitives(model):
    employees = model.Department.employees
    old_branch = employees.age
    node = employees < model.Manager
    assert isinstance(node, CodelessNode)
    assert str(employees._parent.employees.title) == "Department.employees.title"
    assert employees._parent.employees.age is not old_branch
    name = model.Employee.name
    assert (name == None).vargs == ("Employee.name", "IS NULL")  # noqa: E711
    assert (name != None).vargs == ("Employee.name", "IS NOT NULL")  # noqa: E711
    assert (name == ["Alice"]).vargs == ("Employee.name", "ONE OF", ["Alice"])
    assert (name != ["Alice"]).vargs == ("Employee.name", "NONE OF", ["Alice"])
    assert (name % ("Alice", "Company")).vargs == (
        "Employee.name",
        "LOOKUP",
        "Alice",
        "Company",
    )
    assert (name == model.Employee.end).vargs == ("Employee.name", "IS", "Employee.end")


def test_root_column_subclass_mapping_updates_navigation(model):
    column = model.Employee
    original = column.age
    assert isinstance(column < model.Manager, CodelessNode)
    assert str(column.title) == "Employee.title"
    assert column.age is not original
    path_column = Column(model.make_path("Employee", {"Employee": "Manager"}), model)
    assert str(path_column.title) == "Employee.title"


def test_restored_model_infers_legacy_direct_query_profile(model):
    from intermine314.query import Query

    assert Query(model).compatibility == "legacy"
    assert Query(model, compatibility="native").compatibility == "native"
