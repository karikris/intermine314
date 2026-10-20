"""Compare installed original/legacy/native contracts in isolated interpreters.

Original InterMine is installed only in --reference-python. Candidate interpreters
must not contain it. Parquet observations are queryable with benchmarks.evidence.
"""

import argparse
import gc
import importlib
import importlib.metadata
import importlib.util
import io
import json
import random
import subprocess
import sys
import tempfile
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET


def fixture_api(variant, fixture):
    package = 'intermine' if variant == 'original' else 'intermine314'
    if variant != 'original' and importlib.util.find_spec('intermine') is not None:
        raise RuntimeError('Candidate environment contains original InterMine')
    results = importlib.import_module(package + '.results')
    constraints = importlib.import_module(package + '.constraints')
    model_type = importlib.import_module(package + '.model').Model
    query_type = importlib.import_module(package + '.query').Query
    model = model_type(Path(fixture).read_text())
    options = {} if variant == 'original' else {'compatibility': variant}
    query = query_type(model, root='Employee', **options)
    query.add_view('Employee.name', 'Employee.age')
    query.add_constraint('name', 'ONE OF', ['Ada', 'Å & <'])
    return results, constraints, model, query


class FixtureOpener:
    def __init__(self, count=2):
        self.wire = (b'{"results":[\n' + b',\n'.join([b'["Ada",42]'] * count)
                     + b'\n],"wasSuccessful":true}\n')
        self.requests = 0
        self.streams = []

    def open(self, *args, **kwargs):
        self.requests += 1
        stream = io.BytesIO(self.wire)
        self.streams.append(stream)
        return stream

    def close(self):
        for stream in self.streams:
            stream.close()
        self.streams.clear()


def probe(variant, fixture):
    results, constraints, model, query = fixture_api(variant, fixture)
    values = [1, None, 'Å']
    encoded = results.encode_dict({'values': values, 'number': 12})
    con = constraints.MultiConstraint('Employee.name', 'ONE OF', values)
    clone = query.clone()
    clone.get_constraint('A').values.append('clone')
    node = ET.fromstring(query.to_xml())
    observation = {
        'text_bytes': results.encode_str('Å') == 'Å'.encode(),
        'scalar_preserved': results.encode_str(12) == 12,
        'encoded_collection_identity': encoded[b'values'] is values,
        'constraint_collection_identity': con.values is values and con.to_dict()['value'] is values,
        'clone_isolated': query.get_constraint('A').values == ['Ada', 'Å & <'],
        'xml_view': node.get('view'),
        'xml_values': [item.text for item in node.findall('constraint/value')],
    }
    opener = FixtureOpener()
    service = SimpleNamespace(root='https://offline.invalid', version=8, opener=opener,
                              compatibility=variant, model=model)

    def iterator(row='dict'):
        return results.ResultIterator(service, '/query/results', {}, row, ['Employee.name', 'Employee.age'])

    try:
        source = iterator()
        try:
            observation['length'] = len(source)
        except TypeError:
            observation['length'] = None
        opener.requests = 0
        observation['truth'] = bool(iterator())
        observation['truth_requests'] = opener.requests
        opener.requests = 0
        observation['rows'] = list(iterator())
        observation['list_requests'] = opener.requests
        source = iterator('rr')
        stream = iter(source)
        first = next(stream)
        observation['mutable_row_views'] = isinstance(first.views, list)

        class Replacement(results.ResultRow):
            pass

        source.row = Replacement
        observation['active_constructor_change'] = isinstance(next(stream), Replacement)
        list(stream)  # Consume and validate the footer.
        return observation
    finally:
        opener.close()



def measure(variant, fixture):
    """Warm local fixtures; Python allocation peaks exclude native allocations."""
    results, _, _, query = fixture_api(variant, fixture)
    opener = FixtureOpener(count=10000)
    service = SimpleNamespace(root='https://offline.invalid', version=8, opener=opener,
                              compatibility=variant)

    def stream(row='dict'):
        return results.ResultIterator(service, '/query/results', {}, row, ['Employee.name', 'Employee.age'])

    operations = {
        'query_xml': lambda: sum(len(query.to_xml()) for _ in range(200)),
        'dict_stream': lambda: sum(row['Employee.age'] for row in stream()),
        'dict_list': lambda: len(list(stream())),
        'rows_retained': lambda: len(list(stream('rr'))),
    }
    if variant != 'original':
        from intermine314.query import Query

        frame_query = Query(root='Employee', compatibility=variant)
        frame_query.views = ['Employee.name', 'Employee.age']
        batch = [{'Employee.name': 'Ada', 'Employee.age': 42}] * 1000
        frame_query.iter_batches = lambda **options: (batch for _ in range(10))
        for backend in ('polars', 'pandas'):
            operations['frame_' + backend] = lambda backend=backend: len(frame_query.dataframe(backend=backend))
    trials = []
    for case, operation in operations.items():
        value = operation()  # Import, codec and engine warm-up is not timed.
        opener.close()
        gc.collect()
        before = opener.requests
        started = time.perf_counter_ns()
        actual = operation()
        opener.close()
        elapsed = time.perf_counter_ns() - started
        requests = opener.requests - before
        assert actual == value
        gc.collect()
        tracemalloc.start()
        assert operation() == value
        opener.close()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        trials.append(dict(case=case, nanoseconds=elapsed, python_peak_bytes=peak,
                           requests=requests, result=value, units=200 if case == 'query_xml' else 10000))
    return trials


def live_checks(output):
    """Read-only bounded correctness samples; no public throughput benchmark."""
    from intermine314.export import ResourceRuntime, query_parquet, query_parquet_reader
    from intermine314.query import ParallelOptions
    from intermine314.service import Service

    observations = []
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    for mine, url in (
        ('LegumeMine', 'https://mines.legumeinfo.org/legumemine/service'),
        ('MaizeMine', 'https://maizemine.rnet.missouri.edu/maizemine/service'),
    ):
        for profile in ('native', 'legacy'):
            record = dict(mine=mine, profile=profile, checked_at=datetime.now(UTC).isoformat(),
                          source_revision=revision, status='passed', rows=None, templates=None, error=None)
            started = time.perf_counter()
            try:
                with (tempfile.TemporaryDirectory(prefix='intermine314-live-') as directory,
                      ResourceRuntime('cpu2') as runtime, runtime.activate(),
                      Service(url, compatibility=profile) as service):
                    query = service.select('Gene.id', 'Gene.primaryIdentifier').order_by('Gene.id')
                    serial = [row for row in query.results('dict', size=3)]
                    assert len(serial) == 3
                    options = ParallelOptions(max_workers=2, inflight_limit=2, page_size=2, ordered=True)
                    assert list(query.run_parallel(size=3, parallel_options=options)) == serial
                    path = query.to_parquet(Path(directory) / 'sample.parquet', size=3,
                                            single_file=True, parallel_options=options)
                    assert query_parquet(path).to_dicts() == serial
                    with query_parquet_reader(path) as reader:
                        assert reader.read_all().to_pylist() == serial
                    frame = query.dataframe(size=3)
                    assert (frame.to_dict('records') if profile == 'legacy' else frame.to_dicts()) == serial
                    templates = service.templates
                    for name in templates:
                        assert service.get_template(name).name == name
                    record.update(rows=len(serial), templates=len(templates))
            except Exception as exc:
                record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            record['seconds'] = time.perf_counter() - started
            observations.append(record)
            print(json.dumps(record), flush=True)
    import polars as pl

    output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(observations).write_parquet(output)
    if any(row['status'] != 'passed' for row in observations):
        raise RuntimeError('One or more live checks failed; inspect recorded observations')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=('original', 'legacy', 'native'))
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--reference-python', type=Path)
    parser.add_argument('--candidate-python', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--measure', action='store_true')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.live:
        if args.output is None:
            parser.error("live checks require output")
        return live_checks(args.output)
    if args.fixture is None:
        parser.error("offline probes require fixture")
    if args.worker:
        observation = measure(args.worker, args.fixture) if args.measure else probe(args.worker, args.fixture)
        package = 'intermine' if args.worker == 'original' else 'intermine314'
        location = Path(importlib.import_module(package).__file__).resolve()
        if not location.is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError('Probe requires an installed package inside its interpreter environment')
        print(json.dumps({'observations': observation, 'environment': {
            'package_file': str(location), 'version': importlib.metadata.version(package),
            'python': sys.version, 'python_executable': sys.executable,
        }}))
        return
    if not all((args.reference_python, args.candidate_python, args.output)):
        parser.error('reference-python, candidate-python and output are required')
    if args.repeats < 1:
        parser.error('repeats must be positive')
    observations, environments, trials = {}, {}, []
    rng = random.Random(314)
    with tempfile.TemporaryDirectory(prefix='intermine314-acceptance-') as directory:
        for repeat in range(args.repeats if args.measure else 1):
            variants = ['original', 'legacy', 'native']
            rng.shuffle(variants)
            for variant in variants:
                python = args.reference_python if variant == 'original' else args.candidate_python
                command = [str(python.absolute()), '-I', str(Path(__file__).resolve()),
                           '--worker', variant, '--fixture', str(args.fixture.resolve())]
                raw = subprocess.check_output(command + (['--measure'] if args.measure else []),
                                              cwd=directory, text=True, timeout=120)
                result = json.loads(raw)
                observations[variant] = result['observations']
                environments[variant] = result['environment']
                if args.measure:
                    trials.extend(dict(variant=variant, repeat=repeat, environment_json=json.dumps(result['environment']), **row)
                                  for row in result['observations'])
    import polars as pl

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.measure:
        pl.DataFrame(trials).write_parquet(args.output)
        print(f'Saved {len(trials)} warm fixture observations to {args.output}')
    else:
        assert observations['original'] == observations['legacy'], observations
        native = observations['native']
        assert native['length'] is None and native['truth_requests'] == 0 and native['list_requests'] == 1
        assert native['mutable_row_views'] is False and native['active_constructor_change'] is False
        pl.DataFrame([{'variant': variant, 'status': 'passed', 'details_json': json.dumps(value),
                       'environment_json': json.dumps(environments[variant])}
                      for variant, value in observations.items()]).write_parquet(args.output)
        print(json.dumps(observations, indent=2))


if __name__ == '__main__':
    main()
