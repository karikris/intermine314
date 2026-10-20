"""Explicit response/session ownership assertions."""
def assert_responses_closed(session):
    assert all(response.close_calls == 1 for response in session.responses)

def assert_borrowed_closed(session):
    from tests.fixtures.compatibility import assert_closed
    assert_closed(session)

def fresh_model():
    from intermine314.model import Model
    from tests.fixtures.compatibility import fixture_bytes
    return Model(fixture_bytes("model.xml"))
