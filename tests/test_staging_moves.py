import errno
from pathlib import Path

import pytest

from intermine314.export import ParquetOptions, parquet
from tests.support.parquet import query_with_rows


def test_validated_same_filesystem_parts_are_moved_without_copying(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Same-filesystem staging must not copy completed parts')

    monkeypatch.setattr(parquet.shutil, 'copytree', forbidden)
    target = tmp_path / 'result'
    query_with_rows([{'Gene.id': 1}], ['Gene.id']).to_parquet(target)
    assert (target / 'part-00000.parquet').is_file()
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize('directory', [False, True])
def test_cross_filesystem_staging_copies_and_retains_source_for_owner_cleanup(tmp_path, monkeypatch, directory):
    source = tmp_path / 'source'
    if directory:
        source.mkdir()
        (source / 'data').write_bytes(b'complete')
    else:
        source.write_bytes(b'complete')

    def cross_device(*args):
        raise OSError(errno.EXDEV, 'cross-device')

    monkeypatch.setattr(parquet.os, 'replace', cross_device)
    target = tmp_path / 'staged'
    parquet._move_or_copy(source, target)
    assert source.exists()
    assert (target / 'data' if directory else target).read_bytes() == b'complete'


@pytest.mark.parametrize('failure', [OSError(errno.EIO, 'disk failed'), KeyboardInterrupt('interrupted')])
def test_staging_move_failure_preserves_existing_output(tmp_path, monkeypatch, failure):
    target = tmp_path / 'result'
    query_with_rows([{'Gene.id': 7}], ['Gene.id']).to_parquet(target)
    before = (target / 'part-00000.parquet').read_bytes()
    replace = parquet.os.replace

    def fail(source, destination):
        if Path(source).name == 'parts':
            raise failure
        return replace(source, destination)

    monkeypatch.setattr(parquet.os, 'replace', fail)
    with pytest.raises(type(failure)):
        query_with_rows([{'Gene.id': 8}], ['Gene.id']).to_parquet(
            target, parquet_options=ParquetOptions(compact=False),
        )
    assert (target / 'part-00000.parquet').read_bytes() == before
    assert list(tmp_path.iterdir()) == [target]
