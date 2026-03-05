from __future__ import annotations

from contextlib import closing
from urllib.parse import urlencode

from intermine314.query.spec import QuerySpec, query_spec_to_xml


class QueryExecutor:
    """Execution adapter that exposes count()/results() over a QuerySpec."""

    def __init__(self, service, spec: QuerySpec):
        self.service = service
        self.spec = spec

    def __iter__(self):
        return self.results(row="dict")

    def get_results_path(self) -> str:
        return self.service.QUERY_PATH

    def to_query_params(self) -> dict[str, str]:
        return {"query": query_spec_to_xml(self.spec)}

    def results(self, row="dict", start=0, size=None):
        params = self.to_query_params()
        params["start"] = start
        if size is not None:
            params["size"] = size
        return self.service.get_results(
            self.get_results_path(),
            params,
            row,
            list(self.spec.views),
            self.spec.root_class,
        )

    def count(self) -> int:
        params = self.to_query_params()
        params["format"] = "count"
        payload = urlencode(params, True).encode("utf-8")
        url = self.service.root + self.get_results_path()
        with closing(self.service.opener.open(url, payload)) as conn:
            raw = conn.read()
        if isinstance(raw, bytes):
            text = raw.decode("utf-8", errors="replace")
        else:
            text = str(raw)
        try:
            return int(text.strip())
        except ValueError as exc:
            raise ValueError("Server returned a non-integer count: " + text) from exc
