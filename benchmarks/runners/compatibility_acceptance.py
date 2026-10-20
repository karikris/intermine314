"""Compare installed original/legacy/native contracts in isolated interpreters.

Original InterMine is installed only in --reference-python. Candidate interpreters
must not contain it. Parquet observations are queryable with benchmarks.evidence.
"""

import argparse
import importlib
import importlib.metadata
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=('original', 'legacy', 'native'))
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--reference-python', type=Path)
    parser.add_argument('--candidate-python', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.worker:
        observation = probe(args.worker, args.fixture)
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
    observations, environments = {}, {}
    with tempfile.TemporaryDirectory(prefix='intermine314-acceptance-') as directory:
        for variant in ('original', 'legacy', 'native'):
            python = args.reference_python if variant == 'original' else args.candidate_python
            raw = subprocess.check_output([str(python.absolute()), '-I', str(Path(__file__).resolve()),
                                           '--worker', variant, '--fixture', str(args.fixture.resolve())],
                                          cwd=directory, text=True, timeout=60)
            result = json.loads(raw)
            observations[variant] = result["observations"]
            environments[variant] = result["environment"]
    assert observations['original'] == observations['legacy'], observations
    native = observations['native']
    assert native['length'] is None and native['truth_requests'] == 0 and native['list_requests'] == 1
    assert native['mutable_row_views'] is False and native['active_constructor_change'] is False
    import polars as pl

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([{'variant': variant, 'status': 'passed', 'details_json': json.dumps(value),
                   'environment_json': json.dumps(environments[variant])}
                  for variant, value in observations.items()]).write_parquet(args.output)
    print(json.dumps(observations, indent=2))


if __name__ == '__main__':
    main()
