"""The old baseline entry point must exercise real, current query APIs."""

import json
from types import SimpleNamespace
from urllib.parse import parse_qs

import polars as pl
import pytest

from benchmarks.runners import phase0_baselines
from benchmarks.runners.phase0_baselines import _export_query, _parser
from intermine314.export.resource_profile import ResourceProfile
from intermine314.query.parallel_offset import ParallelExecutionError
from tests.fixtures.compatibility.http import loopback_service

MODEL = b'''<model name="test" package="org.test">
<class name="Gene"><attribute name="identifier" type="java.lang.String"/>
<reference name="organism" referenced-type="Organism"/></class>
<class name="Organism"><attribute name="name" type="java.lang.String"/></class>
</model>'''
COLUMNS = ["Gene.identifier", "Gene.organism.name"]
ROWS = [["0001", "plant"], ["0002", None], ["0003", "plant"]]


@pytest.mark.parametrize("encoding", ["gzip", "zstd"])
@pytest.mark.parametrize("successful", [True, False])
def test_phase0_exports_joins_and_checks_late_status(tmp_path, encoding, successful):
    def page(request):
        params = parse_qs(request.data.decode())
        assert 'style="OUTER"' in params["query"][0]
        start, size = int(params["start"][0]), int(params["size"][0])
        body = ('{"results":[\n' + ',\n'.join(json.dumps(row) for row in ROWS[start:start + size])
                + '\n],"wasSuccessful":' + str(successful or start == 0).lower()
                + ',"error":"late failure"}\n')
        return body.encode(), encoding

    target = tmp_path / "export.parquet"
    target.write_bytes(b"existing output")
    args = _parser().parse_args([
        "--rows-target", "3", "--page-size", "2", "--max-workers", "2",
        "--ordered", "ordered", "--ordered-window-pages", "2",
    ])
    routes = {
        ("GET", "/service/version/ws"): b"30",
        ("GET", "/service/model"): (MODEL, encoding),
        ("POST", "/service/query/results"): page,
    }
    with loopback_service(routes) as (url, received):
        settings = SimpleNamespace(mine_url=url + "/service", root_class="Gene",
                                   views=COLUMNS, joins=["Gene.organism"])
        profile = ResourceProfile(max_workers=1, temp_dir=tmp_path / "scratch")
        if not successful:
            with pytest.raises(ParallelExecutionError):
                _export_query(args, settings, profile, target)
            assert target.read_bytes() == b"existing output"
        else:
            result = _export_query(args, settings, profile, target)
            con = result["duckdb_connection"]
            try:
                assert con.execute('SELECT * FROM results ORDER BY "Gene.identifier"').fetchall() == [tuple(row) for row in ROWS]
            finally:
                con.close()
            assert pl.read_parquet(target).columns == COLUMNS
        assert len([request for request in received if request.method == "POST"]) == 2
    assert list((tmp_path / "scratch").iterdir()) == []


def test_cli_persists_typed_metrics_in_parquet(tmp_path, monkeypatch):
    report = {'status': 'ok', 'rows': 3, 'elapsed_s': 0.25,
              'nested': {'passed': True, 'missing': None, 'empty': []}}
    monkeypatch.setattr(phase0_baselines, '_build_report', lambda args: (0, report))
    target = tmp_path / 'metrics.parquet'
    assert phase0_baselines.run(['--parquet-out', str(target)]) == 0
    frame = pl.read_parquet(target)
    values = {tuple(row['path']): row for row in frame.to_dicts()}
    assert values[('rows',)]['int_value'] == 3
    assert values[('elapsed_s',)]['float_value'] == 0.25
    assert values[('nested', 'passed')]['bool_value'] is True
    assert values[('nested', 'missing')]['value_type'] == 'NoneType'
    assert values[('nested', 'empty')]['value_type'] == 'list'
