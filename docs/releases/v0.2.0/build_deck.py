"""Build the 16:9 release deck from versioned Parquet evidence.

Presentation-only dependencies: reportlab==5.0.1 and duckdb==1.5.6.
Current-source preview: publication is deferred; existing PyPI 0.2.0 is distinct.
"""

import argparse
import json
from pathlib import Path

import duckdb
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

W, H = 1280, 720
INK = "#11233A"
NAVY = "#0B1728"
PAPER = "#F5F6F2"
WHITE = "#FFFFFF"
TEAL = "#087E78"
MINT = "#4DE0B7"
MUTED = "#56677B"
LINE = "#D9E2E4"
ORANGE = "#BF622C"
PURPLE = "#735798"
TOTAL = 15



def register_fonts(directory):
    for font, file in (
        ("Body", "Lato-Regular.ttf"),
        ("Bold", "Lato-Bold.ttf"),
        ("Heavy", "Lato-Heavy.ttf"),
    ):
        pdfmetrics.registerFont(TTFont(font, str(directory / file)))
    pdfmetrics.registerFont(
        TTFont("Code", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
    )


class Deck:
    def __init__(self, output):
        self.c = canvas.Canvas(str(output), pagesize=(W, H), pageCompression=1)
        self.c.setTitle("intermine314 0.2.0 | Familiar calls. Modern analytics.")
        self.c.setAuthor("intermine314 / Kris Kari")
        self.c.setSubject(
            "Release highlights, measured evidence and migration guidance"
        )
        self.page = 0
        self.dark = False
        self.layout = []

    def rect(self, x, y, w, h, fill, stroke=None, radius=0):
        self.c.setFillColor(HexColor(fill))
        self.c.setStrokeColor(HexColor(stroke or fill))
        if radius:
            self.c.roundRect(x, H - y - h, w, h, radius, fill=1, stroke=bool(stroke))
        else:
            self.c.rect(x, H - y - h, w, h, fill=1, stroke=bool(stroke))

    def text(self, x, y, value, size=22, font="Body", color=None):
        color = color or (PAPER if self.dark else INK)
        width = pdfmetrics.stringWidth(str(value), font, size)
        if x < 0 or y < 0 or x + width > W - 24 or y + size > H - 8:
            raise ValueError(f"Slide {self.page}: text outside page: {value!r}")
        self.layout.append(
            dict(page=self.page, text=str(value), x=x, y=y, width=width, size=size)
        )
        self.c.setFillColor(HexColor(color))
        self.c.setFont(font, size)
        self.c.drawString(x, H - y - size, str(value))

    def para(self, x, y, value, width, size=22, color=None, font="Body", leading=None):
        leading = leading or size * 1.32
        cursor = y
        for paragraph in value.split("\n"):
            words, line = paragraph.split(), ""
            for word in words:
                candidate = (line + " " + word).strip()
                if pdfmetrics.stringWidth(candidate, font, size) > width and line:
                    self.text(x, cursor, line, size, font, color)
                    cursor += leading
                    line = word
                else:
                    line = candidate
            if line:
                self.text(x, cursor, line, size, font, color)
            cursor += leading
        return cursor

    def line(self, x1, y1, x2, y2, color=LINE, width=1):
        self.c.setStrokeColor(HexColor(color))
        self.c.setLineWidth(width)
        self.c.line(x1, H - y1, x2, H - y2)

    def pill(self, x, y, value, color=TEAL, dark=False):
        width = pdfmetrics.stringWidth(value, "Bold", 13) + 24
        self.rect(x, y, width, 29, color if dark else "#E1EEEA", radius=14)
        self.text(x + 12, y + 5, value, 13, "Bold", WHITE if dark else color)
        return width

    def slide(self, kicker, title, subtitle="", source="", dark=False):
        if self.page:
            self.c.showPage()
        self.page += 1
        self.dark = dark
        self.rect(0, 0, W, H, NAVY if dark else PAPER)
        self.rect(64, 42, 32, 4, MINT if dark else TEAL)
        self.text(109, 33, kicker.upper(), 14, "Bold", MINT if dark else TEAL)
        self.text(64, 87, title, 42, "Heavy")
        if subtitle:
            self.para(64, 149, subtitle, 1140, 21, "#AFBED0" if dark else MUTED)
        self.line(64, 661, 1216, 661, "#28394F" if dark else LINE)
        footer = "#AFBED0" if dark else MUTED
        self.text(64, 680, "intermine314 / 0.2.0 source preview", 11, "Bold", footer)
        if source:
            self.text(335, 680, source, 11, color=footer)
        self.text(1152, 680, f"{self.page:02d} / {TOTAL}", 13, color=footer)
        self.c.bookmarkPage(f"slide-{self.page}")
        self.c.addOutlineEntry(title, f"slide-{self.page}")

    def card(self, x, y, w, h, value, label, detail, accent=TEAL):
        self.rect(x, y, w, h, WHITE, LINE, radius=14)
        self.rect(x + 22, y + 24, 28, 4, accent)
        self.text(x + 22, y + 47, value, 48, "Heavy", accent)
        self.para(x + 22, y + 115, label, w - 44, 23, INK, "Bold")
        self.para(x + 22, y + 183, detail, w - 44, 18, MUTED)

    def bar(self, x, y, label, value, maximum, width, color, shown):
        self.text(x, y, label, 20, "Bold")
        self.rect(x, y + 38, width, 45, "#E5EAE7", radius=5)
        self.rect(x, y + 38, width * value / maximum, 45, color, radius=5)
        self.text(x, y + 96, shown, 22, "Bold", color)

    def link(self, x, y, label, url, size=18):
        self.text(x, y, label, size, "Body", MINT if self.dark else TEAL)
        width = pdfmetrics.stringWidth(label, "Body", size)
        self.c.linkURL(url, (x, H - y - size - 3, x + width, H - y + 3), relative=0)


def evidence(root):
    folder = root / 'docs/analysis/original-comparison'
    with duckdb.connect() as con:
        def records(name):
            cursor = con.execute('SELECT * FROM read_parquet(?)', [str(folder / name)])
            columns = [item[0] for item in cursor.description]
            return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
        summaries = records('summary.parquet')
        storage = records('storage.parquet')
        environment = records('environment.parquet')[0]
        cursor = con.execute('SELECT "case",variant,median(nanoseconds)/1e6 AS ms, '
                             'median(python_peak_bytes) AS peak FROM read_parquet(?) GROUP BY ALL',
                             [str(folder / 'api-trials.parquet')])
        api = {(r[0], r[1]): {'ms': r[2], 'peak': r[3]} for r in cursor.fetchall()}
    return ({(r['fixture'], r['rows'], r['case']): r for r in summaries},
            {(r['fixture'], r['rows'], r['format']): r['bytes'] for r in storage}, environment, api)


def table(d, headers, rows, widths, x=64, y=226, row_height=66, size=21):
    total = sum(widths)
    d.rect(x, y, total, 46, TEAL, radius=6)
    cursor = x
    for value, width in zip(headers, widths, strict=True):
        d.text(cursor + 14, y + 10, value, 16, 'Bold', WHITE)
        cursor += width
    for index, row in enumerate(rows):
        top = y + 46 + index * row_height
        if index % 2 == 0:
            d.rect(x, top, total, row_height, WHITE)
        cursor = x
        for column, (value, width) in enumerate(zip(row, widths, strict=True)):
            d.text(cursor + 14, top + 18, str(value), size,
                   'Bold' if column == 0 else 'Body', INK if column == 0 else MUTED)
            cursor += width


def build(root, output):
    summary, storage, env, api = evidence(root)
    def m(case, n=1000000, fixture='repetitive', field='seconds'):
        return summary[(fixture, n, case)][field]
    def b(fmt, fixture='repetitive', n=1000000):
        return storage[(fixture, n, fmt)]
    def ratio(old, new):
        return f'{old / new:.1f}×'
    def mb(value):
        return f'{value / 1024**2:.1f}'
    def sec(value):
        return f'{value:.3f}'
    d = Deck(output)
    d.c.setSubject('Original InterMine 1.13.0 versus current intermine314 source; publication deferred')
    revision = env['source_revision'][:8]

    d.slide('0.2 release series / October 2026', 'Familiar calls.',
            source=f'Current source {revision} / publication deferred', dark=True)
    d.text(64, 145, 'Modern analytics.', 65, 'Heavy', MINT)
    d.text(64, 260, 'intermine314', 35, 'Bold')
    d.text(64, 308, '0.2.0', 100, 'Heavy')
    d.para(66, 447, 'Measured against original InterMine 1.13.0: speed, memory, storage and the cost of migration.',
           700, 26, '#BDCBDA')
    d.pill(66, 564, 'CURRENT SOURCE PREVIEW — NOT A NEW PYPI RELEASE', dark=True)
    for i, label in enumerate(('PARQUET', 'DUCKDB', 'ARROW', 'POLARS')):
        y = 232 + i * 85
        d.rect(889, y, 292, 62, '#142B42', '#28465A', radius=12)
        d.text(919, y + 17, label, 21, 'Bold', MINT if i == 0 else PAPER)
        if i < 3:
            d.line(1035, y + 62, 1035, y + 85, '#456474', 2)

    d.slide('Measured outcomes', 'The largest gains arrive when data is reused.',
            'One million annotation rows, eight columns. Five fresh-process repetitions per workflow.', '[B1] Repetitive fixture / medians')
    saving = 100 * (1 - b('modern_parquet_4') / b('original_stream_csv'))
    rss_saving = 100 * (1 - m('modern_parquet_4', field='peak_rss_bytes') / m('original_pandas_csv', field='peak_rss_bytes'))
    d.card(64, 232, 360, 326, ratio(m('read_csv_pandas'), m('read_parquet_duckdb')),
           'Faster repeated aggregation', 'CSV + pandas versus Parquet + DuckDB. Warm OS cache; same grouped result.')
    d.card(458, 232, 360, 326, f'{saving:.0f}%', 'Less disk than plain CSV',
           'Zstandard Parquet versus streamed CSV. Data entropy and compression change the saving.')
    d.card(852, 232, 360, 326, f'{rss_saving:.0f}%', 'Lower export peak RSS',
           'Four-worker Parquet versus original dataframe → CSV. Streaming CSV has a lower memory floor.')
    d.para(64, 592, 'These are measured workflow comparisons, not a universal package speedup. Small jobs and direct CSV export can favor the original client.', 1144, 20, MUTED)

    d.slide('Compatibility', 'Keep the familiar entry point.',
            'The historical facade restores legacy behavior around a shared modern runtime.', '[A1] Compatibility contract and scoped audit')
    d.rect(64, 225, 656, 213, NAVY, radius=12)
    for i, line in enumerate(('from intermine314.webservice import Service',
                              'service = Service(mine_url)',
                              'query = service.new_query("Gene")',
                              'query.add_view("Gene.primaryIdentifier")',
                              'for row in query.rows(): ...')):
        d.text(85, 249 + i * 32, line, 19, 'Code', MINT if i == 0 else PAPER)
    d.para(760, 229, '338 passing scopes\n122 tested departures\n460 inventoried symbols', 437, 29, INK, 'Bold', 53)
    d.para(64, 479, 'Queries, constraints, templates, rows, lists and metadata retain familiar entry points. Legacy len(), truth checks and mutable row metadata retain their expected behavior.', 650, 22)
    d.para(760, 457, 'For query.dataframe(), install the pandas extra. Output uses Arrow dtypes. The audit counts are scoped assertions, not a promise of complete API equivalence.', 437, 21, MUTED)

    d.slide('Execution', 'One data pipeline behind both interfaces.',
            'A modern storage workflow is explicit; import replacement alone does not activate every optimization.', '[A2] Query, export and resource runtime')
    stages = [('Fetch', 'HTTP pages'), ('Validate', 'Status + types'), ('Persist', 'Parquet'), ('Query', 'DuckDB'), ('Return', 'Arrow / frames')]
    for i, (title, subtitle) in enumerate(stages):
        x = 64 + i * 233
        d.rect(x, 259, 211, 132, WHITE, LINE, radius=10)
        d.text(x + 18, 280, title, 28, 'Bold', TEAL)
        d.text(x + 18, 332, subtitle, 19)
        if i < 4:
            d.line(x + 211, 325, x + 231, 325, TEAL, 3)
    d.para(64, 449, 'Bounded pages and typed Arrow conversion avoid retaining every result as Python dictionaries. Parquet becomes the reusable data source; SQL pushes filtering and aggregation into DuckDB.', 705, 25)
    d.para(835, 449, 'Polars frames by default in native mode. Optional pandas at the legacy boundary. Explicit ownership closes streams, readers and connections.', 369, 22, MUTED)

    d.slide('Import-level performance', 'Compatibility and native modes have different costs.',
            'Warm local fixtures, same Python 3.14.5. No HTTP, storage or native-buffer RSS in these timings.', '[B2] API fixtures / five repeats / milliseconds')
    rows = []
    for case, label in [('query_xml', '200 XML serializations'), ('dict_stream', '10k streamed dictionaries'),
                        ('dict_list', '10k dictionaries → list'), ('rows_retained', '10k retained row objects')]:
        old, legacy, native = [api[(case, variant)]['ms'] for variant in ('original', 'legacy', 'native')]
        rows.append([label, f'{old:.2f}', f'{legacy:.2f}', f'{native:.2f}', ratio(old, legacy)])
    table(d, ['Operation', 'Original', 'Legacy facade', 'Native', 'Import-level gain'], rows,
          [392, 160, 190, 150, 260], row_height=67, size=21)
    d.para(64, 578, 'Legacy list() can issue two requests, matching original behavior. Native iterators are unsized and consume one request; native gains require choosing that interface.', 1140, 21, MUTED)

    d.slide('Export time', 'CSV is cheap to write. Parquet pays for future reads.',
            'HTTP response → output file, including decoding and conversion. No artificial network latency.', '[B1] Repetitive fixture / seconds / lower is better')
    cases = [('original_stream_csv', 'Original → streaming CSV'), ('original_pandas_csv', 'Original → pandas → CSV'),
             ('modern_parquet_1', 'intermine314 → Parquet / 1 worker'), ('modern_parquet_4', 'intermine314 → Parquet / 4 workers')]
    table(d, ['Workflow', '100k rows', '1M rows', '1M requests'],
          [[label, sec(m(case, 100000)), sec(m(case)), str(int(m(case, field='requests')))] for case, label in cases],
          [590, 182, 182, 198], row_height=67)
    d.para(64, 580, 'Original CSV paths use one streaming response. Parquet exports use bounded 10k-row pages and atomic publication. This compares complete workflows, not identical storage formats.', 1142, 21, MUTED)

    d.slide('Memory', 'Memory savings depend on data shape.',
            'Peak process RSS in MiB includes Python objects, imported engines and native buffers.', '[B1] Two fixtures / MiB / lower is better')
    table(d, ['Export workflow', '100k annot.', '1M annot.', '1M diverse'],
          [[label, mb(m(case, 100000, field='peak_rss_bytes')), mb(m(case, field='peak_rss_bytes')),
            mb(m(case, fixture='diverse', field='peak_rss_bytes'))] for case, label in cases],
          [590, 182, 182, 198], row_height=67)
    d.para(64, 578, 'For diverse 1M rows, four-worker Parquet used 7% more RSS than original pandas. Streaming CSV stayed smallest. Queue byte estimates bound admitted work, not total process RSS.', 1143, 21, MUTED)

    d.slide('Storage', 'Compressible annotations benefit most.',
            'The same one million typed records per format. Every persisted value was checked after decoding.', '[B1] Actual file sizes / MiB / shared scale')
    chart_maximum = max(b('original_stream_csv', fixture) for fixture in ('repetitive', 'diverse'))
    for x, fixture, title in [(64, 'repetitive', 'Repeated annotations'), (664, 'diverse', 'Diverse identifiers + sequences')]:
        d.text(x, 226, title, 27, 'Bold')
        maximum = b('original_stream_csv', fixture)
        for i, (fmt, label, color) in enumerate([('original_stream_csv', 'Plain CSV', MUTED),
                                               ('original_csv_gzip6', 'CSV + gzip level 6', PURPLE),
                                               ('modern_parquet_4', 'Parquet + Zstandard', TEAL)]):
            value = b(fmt, fixture)
            y = 295 + i * 88
            d.text(x, y, label, 20, 'Bold')
            d.rect(x, y + 34, 400 * value / chart_maximum, 26, color, radius=3)
            d.text(x + 419, y + 30, mb(value), 21, 'Bold', color)
        gain = 100 * (1 - b('modern_parquet_4', fixture) / maximum)
        d.text(x, 579, f'{gain:.1f}% smaller than plain CSV', 25, 'Bold', TEAL)
    d.para(64, 622, 'gzip is a meaningful control. Parquet also retains schema and supports column selection; compression alone does not explain its value.', 1150, 17, MUTED)

    d.slide('Repeated analysis', 'Read only the columns needed for the question.',
            'Group by chromosome; count records and sum lengths. Same result, warm filesystem cache.', '[B1] 1M rows / median seconds / lower is better')
    query_cases = [('read_csv_python', 'CSV + Python streaming'), ('read_csv_pandas', 'CSV + pandas / two columns'),
                   ('read_csv_duckdb', 'CSV + DuckDB / control'), ('read_parquet_duckdb', 'Parquet + package DuckDB')]
    table(d, ['Read + aggregate', 'Annotations', 'Diverse data', 'Annotations RSS'],
          [[label, sec(m(case)), sec(m(case, fixture='diverse')), mb(m(case, field='peak_rss_bytes')) + ' MiB']
           for case, label in query_cases], [580, 190, 190, 192], row_height=67)
    d.para(64, 579, 'The DuckDB-over-CSV control separates engine improvements from file-format improvements. A new connection is included per operation; imports and interpreter startup are excluded.', 1140, 21, MUTED)

    d.slide('Time saved across a job', 'Repeated analysis can repay the conversion cost.',
            'Original pandas/CSV vs Parquet/DuckDB: export + N analyses. Modeled totals, not a live-mine forecast.', '[B1] Original pandas/CSV versus 4-worker Parquet/DuckDB')
    rows = []
    for fixture, label in [('repetitive', 'Annotations'), ('diverse', 'Diverse')]:
        for n, nlabel in [(100000, '100k'), (1000000, '1M')]:
            export_old, export_new = m('original_pandas_csv', n, fixture), m('modern_parquet_4', n, fixture)
            old_q, new_q = m('read_csv_pandas', n, fixture), m('read_parquet_duckdb', n, fixture)
            totals = []
            for repeats in (1, 10, 100):
                saved = export_old + repeats * old_q - export_new - repeats * new_q
                totals.append(f'{saved:+.2f} s')
            rows.append([f'{label} / {nlabel}', *totals])
    table(d, ['Dataset', '1 analysis', '10 analyses', '100 analyses'], rows,
          [470, 227, 227, 228], row_height=67)
    d.para(64, 579, 'Positive values are time saved; negative values mean the modern workflow took longer. Network conditions, cold storage, query shape and concurrent load can change these estimates.', 1140, 21, MUTED)

    d.slide('Parallel processing', 'A 16-worker budget is shared across stages.',
            'Profiles cpu16, cpu8, cpu6, cpu4 and cpu2 adapt to available CPUs and memory.', '[A2] ResourceProfile defaults; [B1] measured worker comparison')
    table(d, ['Stage', 'Starting allocation', 'Design constraint'],
          [['Public HTTP / metadata', '4 / 4', 'Per-host permits and bounded pages'],
           ['Arrow CPU / I/O', '4 / 2', 'Convert bounded batches'],
           ['Ordered / unordered COPY', '4 / 8', 'One copy; order only when required'],
           ['DuckDB SQL / Polars', '16 / 12', 'One main plan; shared admission']],
          [390, 245, 517], row_height=65, size=21)
    gain = m('modern_parquet_1') / m('modern_parquet_4')
    d.para(64, 571, f'Observed 1 → 4 HTTP workers: {gain:.2f}× on the 1M annotation fixture. More threads are not automatically faster. Host: Ultra 7 265K, WSL2 exposes 18 CPUs, not all 20 cores.', 1140, 21, MUTED)

    d.slide('Correctness under load', 'Performance improvements retain validation.',
            'The export is accepted only after its pages and final schema are valid.', '[A2] Runtime contracts and installed regression checks')
    blocks = [('Validate every page', 'Read terminal server status before accepting rows. Failures leave existing output intact.'),
              ('Account for waiting results', 'Completed pages waiting for earlier pages stay charged to outstanding-page and byte estimates.'),
              ('Decode compressed metadata', 'Sized and whole-response reads decode supported compression consistently.'),
              ('Preserve typed values', 'Arrow and Parquet preserve explicit types; unsafe precision conversions fail instead of silently rounding.')]
    for i, (title, detail) in enumerate(blocks):
        x, y = 64 + (i % 2) * 591, 228 + (i // 2) * 194
        d.rect(x, y, 560, 168, WHITE, LINE, radius=10)
        d.text(x + 22, y + 20, title, 26, 'Bold', TEAL)
        d.para(x + 22, y + 66, detail, 516, 21)

    d.slide('Validation and boundaries', 'Evidence supports a scoped replacement.',
            'The core pipeline and historical facade share tests; the compatibility boundary remains explicit.', '[A1] Completion evidence / exact runtime source')
    d.card(64, 228, 360, 320, '2,805', 'Source tests passed', 'Combined extras on Python 3.14.5. Linux CI; no claim of exhaustive platform coverage.')
    d.card(458, 228, 360, 320, '1,003', 'Installed regressions', 'Executed outside the checkout against the wheel. Seven independent install modes verified.')
    d.card(852, 228, 360, 320, '2 mines', 'Live read-only smoke', 'LegumeMine and MaizeMine: serial/parallel values, exports and template parsing. Small samples only.')
    d.para(64, 577, 'Not certified here: free-threaded Python, subinterpreters, Windows/macOS, authenticated writes or all legacy argument combinations. Pandas uses Arrow dtypes; native iterators are unsized.', 1140, 21, MUTED)

    d.slide('Adoption', 'Start with the facade. Persist data when it will be reused.',
            'Keep existing query-building code; adopt columnar storage where its measured benefits apply.', '[A1] Migration guide / local candidate only')
    d.rect(64, 228, 1152, 260, NAVY, radius=12)
    code = ['from intermine314.webservice import Service',
            'from intermine314.export import query_parquet',
            'query = Service(mine_url).new_query("Gene")',
            'query.add_view("Gene.primaryIdentifier")',
            'query.to_parquet("genes.parquet", single_file=True)',
            'frame = query_parquet("genes.parquet", "SELECT count(*) FROM results")']
    for i, line in enumerate(code):
        d.text(88, 249 + i * 35, line, 21, 'Code', MINT if i in (0, 1) else PAPER)
    d.para(64, 527, 'Use the pandas extra for legacy dataframe() calls. Review documented departures and resource settings before moving large workflows.', 680, 23)
    d.para(805, 527, 'Publication is deferred. Existing PyPI 0.2.0 is an earlier artifact; the benchmarked code is a prepared source build.', 400, 21, MUTED)

    d.slide('Methods and sources', 'Measured, reproducible and explicit about limits.',
            'No comparison in this deck uses intermine314 0.1.7 as its performance baseline.', '[B1/B2] Raw Parquet evidence + reproduction commands')
    refs = [('[B1] Complete workflow trials, storage and method', 'docs/analysis/original-comparison/README.md'),
            ('[B2] Original / legacy / native API trials', 'docs/analysis/original-comparison/api-trials.parquet'),
            ('[A1] Compatibility coverage and departures', 'docs/analysis/v021-readiness/completion.md'),
            ('[A2] Runtime resource policy', 'src/intermine314/export/resource_profile.py')]
    for i, (label, path) in enumerate(refs):
        d.link(64, 222 + i * 52, label, 'https://github.com/karikris/intermine314/blob/master/' + path, 23)
    d.para(64, 454, 'Both clients use CPython 3.14.5. Original InterMine 1.13.0 needs a benchmark-only MutableMapping alias to import Service. No installed package source was patched. Candidate includes its optional speed extra.', 1140, 21)
    d.para(64, 548, 'Synthetic 100k / 1M rows; two entropy profiles; five isolated runs; randomized stage order. Full-value hashes checked. Warm OS cache, no fsync, no WAN. RSS includes native memory. Python-version gains are not isolated.', 1140, 21, MUTED)
    if d.page != TOTAL:
        raise ValueError('Unexpected slide count')
    d.c.save()
    return d.layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--fonts', type=Path, default=Path('/usr/share/fonts/truetype/lato'))
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    register_fonts(args.fonts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / 'intermine314-v0.2.0.pdf'
    layout = build(args.root, output)
    (args.output_dir / 'layout.json').write_text(json.dumps(layout, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
