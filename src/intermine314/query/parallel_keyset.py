from __future__ import annotations

from itertools import islice

from . import constraints
from .pathfeatures import SortOrder, SortOrderList


def _resolve_keyset_path(query, keyset_path, *, query_error_cls):
    if keyset_path is None:
        if query.root is None:
            raise query_error_cls("Cannot infer keyset path when query root is undefined")
        keyset_path = query.root.name + ".id"
    keyset_path = query.prefix_path(str(keyset_path))
    key_path = query.model.make_path(keyset_path, query.get_subclass_dict())
    if not key_path.is_attribute():
        raise query_error_cls("Keyset path must be an attribute path: %s" % (keyset_path,))
    return keyset_path


def _iter_keyset_ids(query, keyset_path, keyset_batch_size):
    id_query = query.clone()
    id_query.clear_view()
    id_query.add_view(keyset_path)
    id_query.joins = []
    id_query._sort_order_list = SortOrderList()
    id_query.add_sort_order(keyset_path, SortOrder.ASC)

    last_seen = None
    cursor_constraint = None
    while True:
        if last_seen is not None:
            if cursor_constraint is None:
                cursor_constraint = constraints.BinaryConstraint(
                    keyset_path,
                    ">",
                    str(last_seen),
                    code="A",
                )
                cursor_constraint.path = id_query.prefix_path(cursor_constraint.path)
                if id_query.do_verification:
                    id_query.verify_constraint_paths([cursor_constraint])
                id_query.uncoded_constraints.append(cursor_constraint)
            else:
                cursor_constraint.value = str(last_seen)
        ids = []
        for rec in islice(id_query.results(row="dict", start=0, size=keyset_batch_size), keyset_batch_size):
            value = rec.get(keyset_path) if isinstance(rec, dict) else rec
            if value is None:
                continue
            token = str(value)
            if not ids or token != ids[-1]:
                ids.append(token)
        if not ids:
            break
        if last_seen is not None and ids[-1] == last_seen:
            break
        last_seen = ids[-1]
        yield ids


def run_parallel_keyset(
    query,
    *,
    row="dict",
    size=None,
    page_size=None,
    max_workers=None,
    ordered=True,
    prefetch=None,
    keyset_path=None,
    keyset_batch_size=None,
    query_error_cls,
):
    del page_size, max_workers, ordered, prefetch
    keyset_path = _resolve_keyset_path(query, keyset_path, query_error_cls=query_error_cls)
    yielded = 0
    chunk_query = query.clone()
    cursor_constraint = constraints.MultiConstraint(
        keyset_path,
        "ONE OF",
        [],
        code="A",
    )
    cursor_constraint.path = chunk_query.prefix_path(cursor_constraint.path)
    if chunk_query.do_verification:
        chunk_query.verify_constraint_paths([cursor_constraint])
    chunk_query.uncoded_constraints.append(cursor_constraint)
    chunk_query._sort_order_list = SortOrderList()
    chunk_query.add_sort_order(keyset_path, SortOrder.ASC)

    for ids in _iter_keyset_ids(query, keyset_path, keyset_batch_size):
        if size is not None and yielded >= size:
            break
        cursor_constraint.values = ids
        chunk_iter = chunk_query.results(row=row, start=0, size=None)
        for item in chunk_iter:
            yield item
            yielded += 1
            if size is not None and yielded >= size:
                return
