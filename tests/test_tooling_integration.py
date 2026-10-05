"""Tooling boundaries which must work for installed distributions and samples."""
import http.client
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_python_security_floor_excludes_unpatched_connect_implementations():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert metadata["project"]["requires-python"] == ">=3.14.5"
    from intermine314 import VERSION

    assert metadata["project"]["version"] == VERSION


@pytest.mark.parametrize("host,headers", [
    ("bad\r\nhost", {}), ("example.test", {"Bad\nName": "value"}),
    ("example.test", {"Header": "bad\r\nvalue"}),
])
def test_supported_python_rejects_connect_injection_before_sending(host, headers):
    class Socket:
        def sendall(self, value):
            pytest.fail("unsafe CONNECT was sent")
    connection = http.client.HTTPConnection("proxy.example.test")
    connection.sock = Socket()
    # Set directly to test the stdlib safeguard urllib3 relies on in _tunnel.
    connection._tunnel_host = host
    connection._tunnel_port = 443
    connection._tunnel_headers = headers
    with pytest.raises((ValueError, http.client.InvalidURL)):
        connection._tunnel()


def test_sphinx_conf_respects_installed_package_resolution(tmp_path):
    package = tmp_path / "intermine314"
    package.mkdir()
    (package / "__init__.py").write_text('VERSION = "7.8.9"\n')
    script = (
        "import sys,runpy,json; "
        f"sys.path = [p for p in sys.path if p != {str(ROOT / 'src')!r}]; "
        f"sys.path.insert(0, {str(tmp_path)!r}); "
        f"conf=runpy.run_path({str(ROOT / 'docs/source/conf.py')!r}); "
        "import intermine314; "
        "print(json.dumps([conf['release'],intermine314.__file__]))"
    )
    result = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["7.8.9", str(package / "__init__.py")]


def test_sample_export_uses_supported_parallel_options_and_managed_connection(tmp_path):
    from intermine314.query.builder import ParallelOptions
    from samples import common
    class Query:
        def to_parquet(self, path, *, size, batch_size, parallel_options):
            assert isinstance(parallel_options, ParallelOptions)
            assert parallel_options.max_workers == 2
            assert parallel_options.ordered == "ordered"
            path = Path(path)
            path.mkdir()
            pl.DataFrame({"id": [2**53 + 1], "code": ["0007"]}).write_parquet(path / "part-00000.parquet")
            return str(path)
    path, con = common.export_parquet_and_open_duckdb(
        Query(), output_dir=tmp_path, parquet_name="parts", table_name="results", max_workers=2,
    )
    with con:
        assert con.execute("SELECT * FROM results").fetchall() == [(2**53 + 1, "0007")]
    assert common.parquet_head(path).to_dicts() == [{"id": 2**53 + 1, "code": "0007"}]
    with pytest.raises(Exception, match="closed"):
        con.execute("SELECT 1")


def test_offline_analytics_tool_runs_real_csv_parquet_sql_arrow_pipeline():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.analytics_smoke"], cwd=ROOT,
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"status": "ok", "rows": 2, "dataframe": "Polars", "storage": "Parquet"}


def test_sample_preview_uses_parallel_options_and_closes_early_iterator(capsys):
    from intermine314.query.builder import ParallelOptions
    from samples import common
    closed = []
    class Query:
        def run_parallel(self, *, row, start, size, parallel_options):
            assert isinstance(parallel_options, ParallelOptions)
            assert parallel_options.max_workers == 2
            def rows():
                try:
                    yield {"id": 1}
                finally:
                    closed.append(True)
            return rows()
    common.preview_rows(Query(), limit=1, max_workers=2)
    assert closed == [True]
    assert "'id': 1" in capsys.readouterr().out
