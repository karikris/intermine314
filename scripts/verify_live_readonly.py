"""Small, opt-in public read checks; never create accounts, lists or other state.

Run from an installed/editable checkout with --output path/to/live.parquet.
No throughput conclusions may be drawn from these three-row samples.
"""

import argparse
import hashlib
import json
import subprocess
import time
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import requests

MINES = (
    ('LegumeMine', 'https://mines.legumeinfo.org/legumemine/service'),
    ('MaizeMine', 'https://maizemine.rnet.missouri.edu/maizemine/service'),
    ('ThaleMine', 'https://bar.utoronto.ca/thalemine/service'),
    ('WheatMine', 'https://urgi.versailles.inrae.fr/WheatMine/service'),
    ('OakMine', 'https://urgi.versailles.inrae.fr/OakMine_PM1N/service'),
)


class BoundedSession(requests.Session):
    def send(self, request, **kwargs):
        kwargs['timeout'] = (4, 8)
        return super().send(request, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    from intermine314.export import query_parquet, query_parquet_reader
    from intermine314.query import ParallelOptions
    from intermine314.service import Service
    from intermine314.webservice import Service as LegacyService

    records = []
    source_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    options = ParallelOptions(max_workers=2, page_size=2, inflight_limit=2, ordered='ordered')
    for mine, root in MINES:
        for profile, service_type in [('native', Service), ('legacy', LegacyService)]:
            record = dict(mine=mine, root=root, profile=profile, checked_at=datetime.now(UTC).isoformat(),
                          source_sha=source_sha, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            started = time.perf_counter()
            try:
                with BoundedSession() as session, service_type(root, session=session, request_timeout=8) as service:
                    query = service.select('Gene.id', 'Gene.primaryIdentifier').order_by('Gene.id')
                    # Explicit dictionary format compares both compatibility profiles.
                    with closing(query.results(row='dict', size=3)) as results:
                        serial = list(results)
                    with closing(query.iter_rows(size=3, parallel_options=options)) as results:
                        parallel = list(results)
                    if serial != parallel or len(serial) != 3:
                        raise AssertionError('Three-row serial/ordered-parallel result mismatch')
                    with TemporaryDirectory(prefix='intermine-live-read-') as temp:
                        target = Path(temp) / 'sample.parquet'
                        query.to_parquet(target, size=3, single_file=True, parallel_options=options)
                        sql = query_parquet(target).to_dicts()
                        with query_parquet_reader(target, batch_size=2) as reader:
                            arrow = reader.read_all().to_pylist()
                        if serial != sql or serial != arrow:
                            raise AssertionError('Parquet/DuckDB/Arrow values differ')
                    names = sorted(service.templates)
                    for name in names:
                        service.get_template(name).to_query_params()
                    template_name = names[0] if names else None
                    template_rows = None
                    if template_name:
                        with closing(service.get_template(template_name).results(row='dict', size=3)) as results:
                            template_rows = sum(1 for _ in results)
                    record.update(status='ok', ws_version=service.version, rows=len(serial),
                                  serial_parallel_parquet_arrow_equal=True, templates_parsed=len(names),
                                  template_executed=template_name, template_rows=template_rows,
                                  result_sha256=hashlib.sha256(json.dumps(serial, sort_keys=True).encode()).hexdigest(),
                                  error_type=None, error=None)
            except Exception as exc:
                record.update(status='unverified', error_type=type(exc).__name__, error=str(exc)[:1200])
            record['elapsed_s'] = time.perf_counter() - started
            records.append(record)
            print(f'{mine} {profile}: {record["status"]} {record.get("error", "") or ""}', flush=True)
            if record['status'] != 'ok':
                break  # Do not repeat blocked or failing endpoints under another facade.
    import pyarrow as pa
    import pyarrow.parquet as pq

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), args.output, compression='zstd')


if __name__ == '__main__':
    main()
