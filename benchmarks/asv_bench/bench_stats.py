from __future__ import annotations

from benchmarks.bench_utils import stat_summary


class TimeStatSummary:
    params = [100, 1_000, 10_000]
    param_names = ["sample_size"]

    def setup(self, sample_size: int) -> None:
        self.values = [float(index) for index in range(sample_size)]

    def time_stat_summary(self, sample_size: int) -> None:
        del sample_size
        stat_summary(self.values)
