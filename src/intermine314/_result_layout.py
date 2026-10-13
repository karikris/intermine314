"""Immutable, pickleable metadata shared only by built-in streamed results."""
from collections.abc import Mapping
from types import MappingProxyType


class _FrozenMap(Mapping):
    __slots__ = ('_data',)

    def __init__(self, values):
        object.__setattr__(self, '_data', MappingProxyType(dict(values)))

    def __setattr__(self, name, value):
        raise AttributeError('Stream metadata is immutable; rebind the row attribute instead')

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __reduce__(self):
        return type(self), (dict(self._data),)


def _row_parser(constructor, views):
    views = tuple(views)
    indexes = {}
    for index, view in enumerate(views):
        indexes[view] = index
        indexes[view.partition('.')[2] or view] = index
    indexes = _FrozenMap(indexes)

    def parse(payload):
        row = constructor(payload, views)
        row.index_map = indexes
        return row

    return parse
