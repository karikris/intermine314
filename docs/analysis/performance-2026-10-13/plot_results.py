"""Create a shareable figure from DuckDB views over the audit's Parquet data."""
import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'intermine314-matplotlib-cache'))

import duckdb
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
con = duckdb.connect(str(HERE/'analysis.duckdb'), read_only=True)
fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), layout='constrained')
for delay in (5, 30):
    data = con.execute("SELECT workers,speedup_vs_one_worker FROM scaling WHERE stage='http_fetch' AND delay_ms=? ORDER BY workers", [delay]).fetchall()
    axes[0].plot([r[0] for r in data], [r[1] for r in data], marker='o', label=f'{delay} ms handler delay')
axes[0].set(title='Fetch scaling depends on latency', ylabel='Speedup vs one worker', xlabel='HTTP workers')
for stage, size, label in [('duckdb_query', 122880, 'DuckDB / 122,880 rows per group'),
                           ('duckdb_query', 500000, 'DuckDB / 500,000 rows per group'),
                           ('polars_query', None, 'Polars / 122,880 rows per group')]:
    data = con.execute('SELECT workers,speedup_vs_one_worker FROM scaling WHERE stage=? AND row_group_size IS NOT DISTINCT FROM ? ORDER BY workers', [stage,size]).fetchall()
    axes[1].plot([r[0] for r in data], [r[1] for r in data], marker='o', label=label)
axes[1].set(title='Native queries need enough work units', ylabel='Speedup vs one thread', xlabel='Native engine threads')
for ordered in (True, False):
    data = con.execute("SELECT workers,peak_rss_bytes/1048576 FROM measured_summary WHERE stage='duckdb_compact' AND variant='copy_zstd' AND ordered=? ORDER BY workers", [ordered]).fetchall()
    axes[2].plot([r[0] for r in data], [r[1] for r in data], marker='o', label='Ordered' if ordered else 'Unordered')
axes[2].set(title='Compaction memory changes with ordering', ylabel='Maximum sampled process RSS (MiB)', xlabel='DuckDB threads')
for ax in axes:
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, loc='best')
    ax.set_xticks([1,4,8,12,16,18,20])
    ax.set_xlim(0,21)
    ax.axvspan(18.5,21,color='#f5cfaa',alpha=.5)
fig.suptitle('Intel Core Ultra 7 265K — 18 CPUs visible in WSL; warm synthetic stage benchmarks', fontsize=14)
fig.supxlabel('Seven timed repetitions per configuration. Shading: beyond guest CPU count. /tmp is tmpfs; physical disk and WAN performance are unmeasured.', fontsize=9)
fig.savefig(HERE/'worker_scaling.svg')
fig.savefig(HERE/'worker_scaling.png',dpi=150)
plt.close(fig)
con.close()
