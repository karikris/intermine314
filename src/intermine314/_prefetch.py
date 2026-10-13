"""Opt-in bounded object prefetch without changing legacy lazy field access."""
from collections import deque
from itertools import islice

from intermine314._result_object import ResultObject
from intermine314.export.runtime import get_runtime
from intermine314.model import ComposedClass, Reference
from intermine314.parallel.policy import require_positive_int
from intermine314.service.resource_utils import close_resource_quietly


def _requests(window, fields, batch_size):
    groups = {}
    for obj in window:
        if not isinstance(obj, ResultObject):
            raise TypeError('prefetch_objects requires ResultObject inputs')
        for name in fields:
            field = obj._cld.get_field(name)
            if name in obj._data or name in obj._attr_cache or not obj._fetchable(field):
                continue
            root = field.declared_in if isinstance(obj._cld, ComposedClass) else obj._cld
            key = (id(obj._service), id(obj._cld.model), root.name)
            group = groups.setdefault(key, dict(service=obj._service, model=obj._cld.model,
                                                root=root, fields={}, recipients={}))
            group['fields'][name] = field
            group['recipients'].setdefault(obj.id, []).append((obj, name))
    for group in groups.values():
        identifiers = iter(group['recipients'])
        while chunk := list(islice(identifiers, batch_size)):
            yield group, chunk


def _fetch(request):
    from intermine314.query import Query

    group, identifiers = request
    root = group['root']
    query = Query(group['model'], service=group['service'], root=root)
    query.add_view(root.name + '.id')
    for name, field in group['fields'].items():
        path = root.name + '.' + name
        query.add_view(path + '.*' if isinstance(field, Reference) else path)
        if isinstance(field, Reference):
            query.outerjoin(path)
    query.add_constraint(root.name + '.id', 'ONE OF', identifiers)
    data = {}
    requested = set(identifiers)
    stream = query.results('jsonobjects')  # No joined-row size limit: collections stay complete.
    try:
        for obj in stream:
            if obj.id not in requested:
                raise ValueError('Prefetch response has a missing or unexpected objectId')
            if obj.id in data:
                raise ValueError('Prefetch response has a duplicate objectId')
            data[obj.id] = obj._data
    finally:
        close_resource_quietly(stream)
    # Reaching here validates the terminal status before accepting this request.
    return request, data


def _window(window, fields, batch_size, workers, runtime):
    # Validate the complete window before submitting work, including field names.
    requests = iter(list(_requests(window, fields, batch_size)))
    accepted = []
    with runtime.executor(workers, 'intermine314-prefetch') as pool:
        pending = deque()
        for request in islice(requests, workers):
            pending.append(pool.submit(_fetch, request))
        while pending:
            accepted.append(pending.popleft().result())
            request = next(requests, None)
            if request is not None:
                pending.append(pool.submit(_fetch, request))
    # A failed request leaves the current window unchanged. Original object
    # identities/cardinality and the existing nested identity maps are retained.
    for (group, identifiers), data in accepted:
        for identifier in identifiers:
            record = data.get(identifier, {})
            for obj, name in group['recipients'][identifier]:
                if name in obj._data or name in obj._attr_cache:
                    continue
                if name in record:
                    obj._data[name] = record[name]
                else:
                    from intermine314.model import Collection

                    obj._attr_cache[name] = [] if isinstance(group['fields'][name], Collection) else None


def prefetch_objects(objects, *fields, batch_size=256, max_workers=None, resource_profile=None, runtime=None):
    """Yield original objects after fetching missing first-level fields in bulk.

    Groups by exact Service/model/class and requests at most ``batch_size`` IDs
    per query. At most one input window (batch_size * workers objects) is held;
    workers default to the profile's metadata count. Collections have no joined
    row limit. Their payload size remains server/data dependent.

    Loaded and selected null fields never fetch. Input order, duplicate objects,
    cardinality and nested identity maps remain unchanged. A failed window makes
    no cache changes; earlier yielded windows remain populated. Keep the input
    iterator and Service open for the full operation. Dotted paths are excluded;
    prefetch the desired fields on nested objects in a subsequent call.
    """
    require_positive_int('batch_size', batch_size)
    if any(not isinstance(name, str) or not name or '.' in name for name in fields):
        raise ValueError('fields must be nonempty first-level field names')
    active = get_runtime(resource_profile, runtime)
    workers = active.profile.workers_for('metadata') if max_workers is None else require_positive_int('max_workers', max_workers)
    fields = tuple(dict.fromkeys(fields))

    def iterate():
        source = iter(objects)
        while window := list(islice(source, batch_size * workers)):
            with active.activate():
                _window(window, fields, batch_size, workers, active)
            # No runtime context, worker future or transport is held over yield.
            yield from window
    return iterate()
