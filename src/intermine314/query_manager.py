"""Historical saved-query helpers through the managed InterMine transport.

Adapted from client 1.13.0 under LICENSE-BSD and NOTICE. Registry/Service imports
are lazy; this module's account state is independent of plotting helpers.
"""

from __future__ import annotations

import json
from urllib.parse import quote
from xml.etree import ElementTree

from intermine314._helper_state import _HelperState

__all__ = ['save_mine_and_token', 'get_all_query_names', 'get_query', 'delete_query', 'post_query']

_state = None


def save_mine_and_token(m, t, *, registry=None, service=None, opener=None, **registry_options):
    """Validate and save an account; return the original class-only error message.

    Keyword options configure an owned Registry. Injected clients are borrowed;
    a Service must match the resolved mine. Replacement closes previous owned
    clients, and failed replacement leaves no active account configuration.
    """
    global mine, token, _state
    _HelperState.validate_options(registry, service, opener, registry_options)
    mine, token = m, t
    old, _state = _state, None
    if old is not None:
        old.close()
    candidate = _HelperState(m, t)
    phase = 'mine'
    try:
        candidate.connect(registry=registry, service=service, opener=opener,
                          registry_options=registry_options)
        phase = 'token'
        _names(candidate)
    except BaseException as error:
        try:
            candidate.close()
        except BaseException:
            pass  # Preserve the original configuration error or interruption.
        if isinstance(error, Exception):
            return 'An exception of type ' + type(error).__name__ + ' occurred. Check ' + phase
        raise
    _state = candidate
    return None


def _configured():
    if _state is None:
        raise RuntimeError('Call save_mine_and_token successfully before using account helpers')
    return _state


def _names(state, parameters=None):
    return json.loads(state.read('/user/queries', parameters))['queries'].keys()


def get_all_query_names():
    """Return saved names in server order, or ``'No saved queries'``."""
    names = _names(_configured())
    return ', '.join(names) if names else 'No saved queries'


def get_query(name):
    """Return raw XML/text, recognizing equivalent empty saved-query XML."""
    answer = _configured().read('/user/queries', {'filter': name, 'format': 'xml'})
    try:
        root = ElementTree.fromstring(answer)
    except ElementTree.ParseError:
        return answer
    if root.tag == 'saved-queries' and not len(root) and not (root.text or '').strip():
        return 'No such query available'
    return answer


def delete_query(name):
    """Delete an existing saved query; return the historical status string."""
    state = _configured()
    if name not in _names(state):
        return 'No such query available'
    state.discard('/user/queries/' + quote(name, safe=''), method='DELETE')
    return name + ' is deleted'


def post_query(value, *, overwrite=None):
    """PUT XML using the version-specific parameter and verify the saved name.

    The default duplicate-name path prompts interactively. Explicit True/False
    bypasses the prompt and chooses replacement or decline respectively.
    """
    if overwrite is not None and type(overwrite) is not bool:
        raise ValueError('overwrite must be None, True or False')
    root = ElementTree.fromstring(value)
    name = root.attrib['name']
    state = _configured()
    version = json.loads(state.read('/version'))
    parameter = 'query' if version >= 27 else 'xml'
    if name in _names(state):
        if overwrite is None:
            print('The query name exists')
            replace = input('Do you want to replace the old query? [y/n]') == 'y'
        else:
            replace = overwrite
        if not replace:
            print('Use a query name other than ' + name)
            return None
    parameters = {parameter: value}
    state.discard('/user/queries', parameters, method='PUT')
    if name not in _names(state, parameters):
        print('Note: name should contain no special symbol            and should be defined first')
        return 'Incorrect format'
    return name + ' is posted'
