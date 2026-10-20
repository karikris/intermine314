from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, field
from urllib.parse import urlencode

from intermine314.query.spec import QuerySpec, query_spec_to_xml
from intermine314.service.transport import open_readonly


class _PreparedParameters(dict):
    """A normal request mapping with a reusable invariant wire prefix."""

    def __init__(self, fields, encoded, keys=None):
        super().__init__(fields)
        self._fields = fields
        self._encoded = encoded
        self._keys = frozenset(key for key, _ in fields) if keys is None else keys

    def encoded(self):
        from intermine314.service.session import encode_dict

        if any(key not in self or self[key] != value for key, value in self._fields):
            return urlencode(encode_dict(self), True).encode('utf-8')
        extra = {key: value for key, value in self.items() if key not in self._keys}
        suffix = urlencode(encode_dict(extra), True).encode('utf-8')
        return self._encoded + (b'&' + suffix if suffix else b'')


@dataclass(frozen=True)
class PreparedExecution:
    """Immutable wire fields, shared by count and every page in one execution.

    Service and model descriptor identity remain shared intentionally; no
    mutable query constraints or joins are retained in the execution snapshot.
    """

    service: object
    path: str
    fields: tuple
    encoded: bytes
    views: tuple
    cld: object
    row: str
    decimal_paths: tuple
    keys: frozenset = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'keys', frozenset(key for key, _ in self.fields))

    def results(self, row='dict', start=0, size=None, *, summary_path=None):
        params = _PreparedParameters(self.fields, self.encoded, self.keys)
        params['start'] = start
        if size is not None:
            params['size'] = size
        if summary_path is not None:
            params['summaryPath'] = summary_path
        options = {'decimal_paths': self.decimal_paths} if self.decimal_paths else {}
        return self.service.get_results(self.path, params, 'jsonrows' if summary_path is not None else self.row,
                                        list(self.views), self.cld, **options)

    def count(self):
        from intermine314.query.builder import ResultError

        try:
            return _read_count(self.service, self.path, self.encoded + b'&format=count')
        except ValueError as exc:
            raise ResultError(str(exc)) from exc


def _read_count(service, path, payload):
    with closing(open_readonly(service.opener, service.root + path, payload)) as conn:
        raw = conn.read()
    text = raw.decode('utf-8', errors='replace') if isinstance(raw, bytes) else str(raw)
    try:
        return int(text.strip())
    except ValueError as exc:
        raise ValueError('Server returned a non-integer count: ' + text) from exc


class QueryExecutor:
    """Execution adapter that exposes count()/results() over a QuerySpec."""

    def __init__(self, service, spec: QuerySpec):
        self.service = service
        self.spec = spec
        self.compatibility = spec.compatibility

    def __iter__(self):
        return self.results(row="dict")

    def get_results_path(self) -> str:
        return self.service.TEMPLATEQUERY_PATH if self.spec.template is not None else self.service.QUERY_PATH

    def to_query_params(self) -> dict[str, object]:
        if self.spec.template is not None:
            from intermine314.query.template import template_query_params

            return template_query_params(self.spec.name, self.spec.template.user_name, self.spec.constraints)
        return {"query": query_spec_to_xml(self.spec)}

    def prepare(self, row='dict', *, cld=None):
        from intermine314.service.session import encode_dict

        # Form encoding resolves mutable constraints into scalar/tuple values.
        fields = tuple((key, tuple(value) if isinstance(value, list) else value)
                       for key, value in encode_dict(self.to_query_params()).items())
        return PreparedExecution(
            self.service, self.get_results_path(), fields,
            urlencode(fields, True).encode('utf-8'), tuple(self.spec.views),
            self.spec.root_class if cld is None else cld, row, tuple(self.spec.decimal_paths),
        )

    def results(self, row="dict", start=0, size=None, *, cld=None, summary_path=None):
        return self.prepare(row, cld=cld).results(start=start, size=size, summary_path=summary_path)

    def count(self) -> int:
        prepared = self.prepare()
        return _read_count(self.service, prepared.path, prepared.encoded + b'&format=count')


class _QueryAdapter:
    """Use a custom query's live parameter/path protocol without a QuerySpec."""

    def __init__(self, query):
        self.query = query

    def results(self, row="dict", start=0, size=None, *, cld=None, summary_path=None):
        query = self.query
        params = query.to_query_params()
        params['start'] = start
        if size is not None:
            params['size'] = size
        if summary_path is not None:
            params['summaryPath'] = summary_path
        return query.service.get_results(query.get_results_path(), params, row, query.views,
                                         query.root if cld is None else cld)

    def count(self):
        params = self.query.to_query_params()
        params['format'] = 'count'
        return _read_count(self.query.service, self.query.get_results_path(),
                           urlencode(params, True).encode('utf-8'))


def execution_for(query):
    resolver = getattr(query, '_to_execution', None)
    execution = resolver() if callable(resolver) else None
    return execution if execution is not None else _QueryAdapter(query)


def builtin_query_methods(query, *names):
    """Only canonical query methods may bypass the caller's live protocol."""
    from intermine314.query.builder import Query
    from intermine314.query.template import Template

    return all(getattr(getattr(query, name, None), '__func__', None)
               in (getattr(Query, name), getattr(Template, name)) for name in names)
