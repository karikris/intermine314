from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from threading import Thread

import pytest

from intermine314.export import ResourceProfile, ResourceRuntime
from intermine314.lists._csv import identifier_body
from intermine314.service.errors import WebserviceError
from intermine314.service.session import InterMineURLOpener
from tests.support.lists import client


@pytest.mark.parametrize('status', [200, 500])
def test_upload_has_exact_length_and_one_http_attempt(tmp_path, status):
    observed = []
    expected = ('"0007"\n"κ"\n"a ""quote"""').encode()
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = self.headers.get('Content-Length')
            body = self.rfile.read(int(length or 0))
            observed.append((length, self.headers.get('Transfer-Encoding'), body))
            self.send_response(status)
            self.send_header('Content-Length', '2')
            self.end_headers()
            self.wfile.write(b'{}')
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    opener = InterMineURLOpener()
    try:
        with ResourceRuntime(ResourceProfile(temp_dir=str(tmp_path))) as runtime, runtime.activate():
            with identifier_body(StringIO('id\n0007\nκ\n"a ""quote"""\n'), 'id', None) as body:
                path = Path(body.name)
                assert path.is_file() and not body.closed
                uri = f'http://127.0.0.1:{server.server_port}/upload'
                if status == 500:
                    with pytest.raises(WebserviceError):
                        opener.post_plain_text(uri, body)
                else:
                    assert opener.post_plain_text(uri, body) == b'{}'
            assert body.closed and not path.exists()
    finally:
        opener.close()
        server.shutdown()
        server.server_close()
        thread.join(3)
    assert observed == [(str(len(expected)), None, expected)]
    assert list(tmp_path.iterdir()) == []


def test_late_invalid_identifier_causes_zero_mutation_requests():
    service, session = client()
    source = StringIO('id\n' + '0007\n' * 70000 + '"line\nbreak"\n')
    with pytest.raises(ValueError, match='line breaks'):
        service.list_manager().create_list(csv_input=source, csv_column='id', list_type='Employee')
    assert not any(request.method == 'POST' for request in session.requests)
    assert not source.closed


def test_identifier_chunks_preserve_boundaries_and_large_unicode_values():
    import csv

    identifiers = ['0007', 'κ' * 40000, 'a "quote"', *[f'{i:09d}' for i in range(70000)]]
    source = StringIO()
    writer = csv.writer(source)
    writer.writerow(['id'])
    writer.writerows([value] for value in identifiers)
    source.seek(0)
    expected = '\n'.join('"' + value.replace('"', '""') + '"' for value in identifiers).encode()
    with identifier_body(source, 'id', None) as body:
        assert body.read() == expected
        assert body.tell() == len(expected)
    assert not source.closed
