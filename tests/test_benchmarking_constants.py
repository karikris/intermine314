from benchmarks import bench_constants as bc


def test_resolve_matrix_rows_constants():
    assert bc.resolve_matrix_rows_constant("MATRIX_ROWS") == bc.rows_to_csv(bc.MATRIX_ROWS)
    assert bc.resolve_matrix_rows_constant("batch_size_test_chunk_rows") == bc.rows_to_csv(
        bc.BATCH_SIZE_TEST_CHUNK_ROWS
    )


def test_resolve_matrix_rows_passthrough():
    assert bc.resolve_matrix_rows_constant("1,2,3") == "1,2,3"
