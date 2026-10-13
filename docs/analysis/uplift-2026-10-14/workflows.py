"""Measure existing opt-in workflows with a controlled 5ms offline transport."""
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qs
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]


def run():
    import pyarrow as pa
    import pyarrow.parquet as pq

    from intermine314.export.runtime import shutdown_resources
    from intermine314.results import ResultObject, prefetch_objects
    from intermine314.service import Service
    from tests.fixtures.compatibility import FixtureSession, fixture_bytes

    class Session(FixtureSession):
        def __init__(self):
            super().__init__(FixtureSession.service().routes)
            self.routes['GET', '/service/templates'] = fixture_bytes('templates.xml')

        def request(self, method, url, data=None, **kwargs):
            time.sleep(.005)
            if method == 'POST':
                query = ElementTree.fromstring(parse_qs(data.decode())['query'][0])
                constraint = query.find('constraint')
                identifiers = ([int(node.text) for node in constraint.findall('value')]
                               if constraint.findall('value') else [int(constraint.get('value'))])
                rows = [{'class': 'Department', 'objectId': i, 'name': f'D{i}'} for i in identifiers]
                self.routes['POST', '/service/query/results'] = (
                    '{"results":[\n' + ',\n'.join(json.dumps(row) for row in rows) + '\n],"wasSuccessful":true}\n').encode()
            return super().request(method, url, data=data, **kwargs)

    def prefetch(batched):
        session = Session()
        with Service('https://offline.example/service', session=session) as service:
            cld = service.model.get_class('Department')
            objects = [ResultObject({'objectId': i}, cld) for i in range(32)]
            before = len(session.requests)
            started = time.perf_counter_ns()
            source = prefetch_objects(objects, 'name', batch_size=8, max_workers=2) if batched else objects
            result = [obj.name for obj in source]
            return time.perf_counter_ns() - started, len(session.requests) - before, result

    def metadata(parallel):
        sessions = [Session() for _ in range(4)]

        def discover(index):
            with Service(f'https://offline{index}.example/service', session=sessions[index]) as service:
                return service.model.name, sorted(service.templates)

        started = time.perf_counter_ns()
        if parallel:
            with ThreadPoolExecutor(max_workers=4) as pool:
                result = list(pool.map(discover, range(4)))
        else:
            result = [discover(i) for i in range(4)]
        return time.perf_counter_ns() - started, sum(len(s.requests) for s in sessions), result

    def reuse(shared):
        session = Session()
        started = time.perf_counter_ns()
        if shared:
            with Service('https://offline.example/service', session=session) as service:
                result = [service.model.name for _ in range(20)]
        else:
            result = []
            for _ in range(20):
                with Service('https://offline.example/service', session=session) as service:
                    result.append(service.model.name)
        return time.perf_counter_ns() - started, len(session.requests), result

    rows, rng = [], random.Random(314)
    for name, function in [('prefetch', prefetch), ('metadata', metadata), ('service_reuse', reuse)]:
        function(False)
        function(True)
        for repetition in range(21):
            variants = [False, True]
            rng.shuffle(variants)
            results = []
            for optimized in variants:
                elapsed, requests, result = function(optimized)
                results.append(result)
                rows.append(dict(workflow=name, optimized=optimized, repetition=repetition,
                                 ns=elapsed, requests=requests, simulated_latency_ms=5))
            assert results[0] == results[1]
    shutdown_resources()
    pq.write_table(pa.Table.from_pylist(rows), Path(__file__).with_name('workflows.parquet'), compression='zstd')
    import duckdb

    with duckdb.connect() as db:
        print(db.execute("SELECT workflow, optimized, median(ns)/1e6 AS ms, median(requests) AS requests FROM read_parquet(?) GROUP BY ALL ORDER BY workflow, optimized",
                         [str(Path(__file__).with_name('workflows.parquet'))]).fetchall())


if __name__ == '__main__':
    run()
