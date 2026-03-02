
from benchmarks import bench_constants as bc


class TestBenchmarkConstants:
    def test_resolve_matrix_rows_constants(self):
        assert bc.resolve_matrix_rows_constant("MATRIX_ROWS") == \
            bc.rows_to_csv(bc.MATRIX_ROWS)
        assert bc.resolve_matrix_rows_constant("batch_size_test_chunk_rows") == \
            bc.rows_to_csv(bc.BATCH_SIZE_TEST_CHUNK_ROWS)

    def test_resolve_matrix_rows_passthrough(self):
        assert bc.resolve_matrix_rows_constant("1,2,3") == "1,2,3"

    def test_shared_constants_exist(self):
        assert bc.DEFAULT_MATRIX_GROUP_SIZE > 0
        assert "auto" in bc.AUTO_WORKER_TOKENS
        assert bc.DEFAULT_PARQUET_COMPRESSION
        assert bc.DEFAULT_MATRIX_STORAGE_DIR
        assert bc.DEFAULT_RUNNER_PREFLIGHT_TIMEOUT_SECONDS > 0
        assert bc.DEFAULT_RUNNER_IMPORT_REPETITIONS > 0
        assert bc.DEFAULT_RUNNER_LOG_LEVEL
        assert bc.DEFAULT_RUNNER_DEBUG_LOG_LEVEL
        assert bc.DEFAULT_RUNNER_PARALLEL_PROFILE
        assert bc.DEFAULT_PARITY_SAMPLE_MODE in bc.VALID_PARITY_SAMPLE_MODES
        assert bc.DEFAULT_PARITY_SAMPLE_SIZE > 0
        assert bc.WARMUP_ROWS > 0
        assert bc.PROGRESS_LOG_INTERVAL_ROWS > 0
        assert bc.RETRY_BACKOFF_INITIAL_SECONDS > 0
        assert bc.RETRY_BACKOFF_MAX_SECONDS >= bc.RETRY_BACKOFF_INITIAL_SECONDS
        assert bc.BATCH_SIZE_TEST_ROWS > 0
        assert bc.BATCH_SIZE_TEST_CHUNK_ROWS
        assert tuple(bc.BATCH_SIZE_TEST_CHUNK_ROWS) == \
            (1_000, 2_500, 5_000, 7_500, 10_000)

