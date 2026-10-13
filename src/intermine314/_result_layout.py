"""Immutable, pickleable metadata shared only by built-in streamed results."""
from collections.abc import Mapping
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _ObjectProjection:
    attributes: tuple
    references: _FrozenMap
    children: _FrozenMap

    @classmethod
    def compile(cls, views):
        attributes, references = [], {}
        for path in views:
            stripped = str(path).split('.', 1)[-1]
            if '.' in stripped:
                references.setdefault(stripped.split('.', 1)[0] + '.', []).append(stripped)
            else:
                attributes.append(stripped)
        return cls(tuple(attributes), _FrozenMap({key: tuple(paths) for key, paths in references.items()}),
                   _FrozenMap({key: cls.compile(paths) for key, paths in references.items()}))
