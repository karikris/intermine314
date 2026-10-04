"""Real result transports must reject failed pages before atomic publication."""

import json
from urllib.parse import parse_qs, urlsplit

import polars as pl
import pytest

from intermine314.query import ParallelOptions
from intermine314.query.parallel_offset import ParallelExecutionError
from intermine314.service import Service
from intermine314.service.errors import WebserviceError
from tests.fixtures.compatibility import SERVICE_ROOT, FixtureResponse, FixtureSession


class StatusSession(FixtureSession):
    def __init__(self, failed_offset):
        super().__init__(FixtureSession.service(version=30).routes)
        self.failed_offset = failed_offset

    def request(self, method, url, data=None, headers=None, **options):
        if urlsplit(url).path != "/service/query/results":
            return super().request(method, url, data, headers, **options)
        self._capture(method, url, data, headers, **options)
        parameters = parse_qs(data.decode())
        start, size = int(parameters["start"][0]), int(parameters["size"][0])
        rows = [[f"row-{index}", index] for index in range(start, start + size)]
        status = {"wasSuccessful": start != self.failed_offset, "error": "server failure after rows"}
        wire = '{"results":[\n' + ',\n'.join(json.dumps(row) for row in rows)
        wire += '\n],' + json.dumps(status)[1:]
        response = FixtureResponse(wire.encode())
        self.responses.append(response)
        return response


@pytest.mark.parametrize("profile", ["native", "legacy"])
@pytest.mark.parametrize("output", ["parquet", "parts", "csv"])
@pytest.mark.parametrize("failed_offset", [0, 2])
def test_failed_full_page_preserves_existing_export_and_cleans_staging(
    tmp_path, profile, output, failed_offset,
):
    target = tmp_path / ("results" if output == "parts" else "results." + output)
    previous = pl.DataFrame({"Employee.name": ["previous"], "Employee.age": [99]})
    if output == "parts":
        target.mkdir()
        previous.write_parquet(target / "part-0.parquet")
        (target / "keep.txt").write_text("existing directory contents")
    elif output == "parquet":
        previous.write_parquet(target)
    else:
        target.write_text("Employee.name,Employee.age\nprevious,99\n")
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    scratch = tmp_path / "scratch"
    session = StatusSession(failed_offset)
    with Service(SERVICE_ROOT, session=session, compatibility=profile) as service:
        query = service.select("Employee.name", "Employee.age")
        with pytest.raises(ParallelExecutionError) as caught:
            query.export(
                target, format="csv" if output == "csv" else "parquet",
                single_file=output != "parts", size=4, batch_size=1,
                parallel_options=ParallelOptions(page_size=2, max_workers=1, inflight_limit=1, ordered=True),
                temp_dir=scratch,
            )
    assert caught.value.offset == failed_offset
    assert isinstance(caught.value.__cause__, WebserviceError)
    after = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before
    assert list(scratch.iterdir()) == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["results" if output == "parts" else target.name, "scratch"]
    calls = [call for call in session.requests if call.path.endswith("query/results")]
    assert [int(parse_qs(call.data.decode())["start"][0]) for call in calls] == list(range(0, failed_offset + 1, 2))
    assert all(response.close_calls == 1 for response in session.responses)
    assert session.close_calls == 0
