"""Build the 16:9 release deck from versioned Parquet evidence.

Presentation-only dependencies: reportlab==5.0.1 and duckdb==1.5.6.
Sales presentation for the unreleased 0.2.1 candidate. Metrics retain their original source provenance.
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
PURPLE = "#735798"
TOTAL = 10



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
        self.c.setTitle("intermine314 0.2.1 | Familiar InterMine. Modern performance.")
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
        self.text(64, 680, "intermine314 / 0.2.1 release preview", 11, "Bold", footer)
        if source:
            self.text(335, 680, source, 11, color=footer)
        self.text(1152, 680, f"{self.page:02d} / {TOTAL}", 13, color=footer)
        self.c.bookmarkPage(f"slide-{self.page}")
        self.c.addOutlineEntry(title, f"slide-{self.page}")

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
    return ({(r['fixture'], r['rows'], r['case']): r for r in summaries},
            {(r['fixture'], r['rows'], r['format']): r['bytes'] for r in storage}, environment)


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
    summary, storage, env = evidence(root)

    def metric(case, fixture='repetitive', field='seconds'):
        return summary[(fixture, 1000000, case)][field]

    def size(fmt, fixture='repetitive'):
        return storage[(fixture, 1000000, fmt)] / 1024**2

    d = Deck(output)
    d.c.setSubject('intermine314 0.2.1 sales presentation; compatibility advances and measured workflow benefits; release preview')
    d.c.setKeywords('intermine314 0.2.1; benchmark source ' + env['source_revision'])
    source = 'Measured against original InterMine 1.13.0 / five-run medians'
    query_gain = metric('read_csv_pandas') / metric('read_parquet_duckdb')
    export_gain = metric('original_pandas_csv') / metric('modern_parquet_4')

    # 01: Product promise.
    d.slide('Introducing 0.2.1', 'intermine314', dark=True)
    d.text(1020, 87, '0.2.1', 44, 'Heavy', MINT)
    d.text(64, 203, 'Familiar InterMine.', 66, 'Heavy', PAPER)
    d.text(64, 288, 'Modern performance.', 66, 'Heavy', MINT)
    d.para(68, 422, 'Keep your workflows.\nUnlock columnar analytics.', 700, 31, '#BDCBDA', leading=45)
    d.pill(68, 564, 'PYTHON 3.14.5+', dark=True)
    for i, (label, width) in enumerate([('PARQUET', 238), ('DUCKDB', 282), ('ARROW', 326), ('POLARS', 370)]):
        x, y = 1216 - width, 227 + i * 82
        d.rect(x, y, width, 61, '#142B42', '#28465A', radius=8)
        d.text(x + 24, y + 16, label, 23, 'Bold', MINT if i == 0 else PAPER)

    # 02: Low-friction migration, with the actual compatibility boundary visible.
    d.slide('Built for InterMine users', 'A familiar way to work.',
            'The historical Service interface stays at the center.')
    d.text(64, 231, 'YOUR EXISTING IMPORT', 15, 'Bold', MUTED)
    d.rect(64, 268, 1152, 87, '#E9EDE9', radius=8)
    d.text(90, 294, 'from intermine.webservice import Service', 29, 'Code', MUTED)
    d.text(64, 388, 'YOUR NEW IMPORT', 15, 'Bold', TEAL)
    d.rect(64, 425, 1152, 87, NAVY, radius=8)
    d.text(90, 451, 'from intermine314.webservice import Service', 29, 'Code', MINT)
    d.text(64, 557, 'Queries     Templates     Lists     Models     Registry', 27, 'Bold')
    d.para(64, 616, 'Update dependency declarations; review documented API and dtype differences.', 1150, 18, MUTED)

    # 03: Compatibility progression, not a performance comparison with old releases.
    d.slide('Two releases. One direction.', 'Closer to the InterMine code you already use.',
            '0.2.0 broadened support. 0.2.1 restores more original behavior.', '[C1] Release notes and compatibility acceptance')
    stages = [
        (64, 'Earlier releases', 'Narrower legacy support', ['More migration work'], False),
        (456, '0.2.0', 'Broader API coverage', ['Queries, XML and templates', 'Lists, summaries and registry', 'Legacy row / object defaults'], False),
        (848, '0.2.1', 'Closer legacy behavior', ['Optional pandas output', 'Result sizing and mutability', 'Live result customization'], True),
    ]
    for x, version, title, features, highlight in stages:
        d.rect(x, 244, 368, 318, NAVY if highlight else WHITE, None if highlight else LINE, radius=12)
        d.text(x + 24, 269, version, 40 if version != 'Earlier releases' else 30, 'Heavy', MINT if highlight else TEAL)
        d.para(x + 24, 341, title, 320, 25, PAPER if highlight else INK, 'Bold')
        for j, line in enumerate(features):
            d.text(x + 24, 420 + j * 37, line, 20, color='#BDCBDA' if highlight else MUTED)
    d.text(64, 603, 'More familiar calls. Fewer changes to established workflows.', 27, 'Bold', TEAL)

    # 04: Concrete changes make the compatibility proposition credible.
    d.slide('What 0.2.1 adds', 'Compatibility goes beyond method names.',
            'Legacy behavior at the boundary. Native analytics underneath.', '[C1] Historical facade; native optimizations retained')
    table(d, ['Existing code relies on…', '0.2.0', '0.2.1'], [
        ['dataframe() output', 'Polars', 'pandas in legacy mode*'],
        ['Result length / truth', 'Unsized streams', 'Original sizing behavior'],
        ['Mutable row metadata', 'Shared / immutable', 'Mutable in legacy mode'],
        ['Live customization', 'Bound at stream opening', 'Changes respected'],
    ], [430, 320, 402], y=231, row_height=70, size=23)
    d.text(64, 584, 'Also restored: public encoders and typed constraint values.', 24, 'Bold', TEAL)
    d.para(64, 627, '*Requires the pandas extra; output uses Arrow dtypes. Native mode retains Polars.', 1150, 17, MUTED)

    # 05: Two measured speed claims, with their denominators attached.
    d.slide('Faster from data to answers', 'Spend less time moving and reading data.',
            'One million annotation rows. Same data. Same analytical result.', source)
    for x, gain, title, before, after in [
        (64, query_gain, 'Faster aggregation', 'CSV + pandas', 'Parquet + DuckDB'),
        (656, export_gain, 'Faster export', 'Original dataframe → CSV', 'intermine314 → Parquet'),
    ]:
        d.rect(x, 237, 560, 317, WHITE, LINE, radius=12)
        d.text(x + 26, 259, f'{gain:.1f}×', 86, 'Heavy', TEAL)
        d.text(x + 28, 373, title, 32, 'Bold')
        d.text(x + 28, 437, before, 23, color=MUTED)
        d.text(x + 28, 483, after, 23, 'Bold', TEAL)
    d.para(64, 599, 'Synthetic local benchmark; warm cache. Small jobs and streaming CSV can favor the original client.', 1150, 19, MUTED)

    # 06: Honest storage comparison with a shared absolute scale.
    d.slide('Smaller data. Easier reuse.', '64–97% less disk than plain CSV.',
            'Measured at one million rows across two data profiles.', '[B1] MiB / shared scale / all persisted values verified')
    maximum = max(size('original_stream_csv', fixture) for fixture in ('repetitive', 'diverse'))
    for x, fixture, title in [(64, 'repetitive', 'Repeated annotations'), (656, 'diverse', 'Diverse identifiers + sequences')]:
        d.text(x, 229, title, 27, 'Bold')
        for i, (fmt, label, color) in enumerate([
            ('original_stream_csv', 'CSV', MUTED),
            ('original_csv_gzip6', 'CSV + gzip', PURPLE),
            ('modern_parquet_4', 'Parquet + Zstandard', TEAL),
        ]):
            value, y = size(fmt, fixture), 300 + i * 91
            d.text(x, y, label, 22, 'Bold')
            d.rect(x, y + 38, 398 * value / maximum, 28, color, radius=3)
            d.text(x + 421, y + 33, f'{value:.1f}', 25, 'Bold', color)
    d.text(64, 607, 'Typed columns. Selective reads. A reusable analytical dataset.', 27, 'Bold', TEAL)

    # 07: The scaling proposition avoids promising that more threads always win.
    d.slide('Scale with control', 'Use your CPU. Manage your memory.',
            'Stage-aware workers and bounded page queues.', '[A2] Default CPU budget / [B1] measured memory tradeoff')
    d.text(64, 225, '16', 119, 'Heavy', TEAL)
    d.text(68, 365, 'Default native CPU budget', 26, 'Bold')
    d.text(68, 415, 'Other profiles: 8 · 6 · 4 · 2', 24, color=MUTED)
    stages = [('HTTP', '4'), ('Arrow CPU', '4'), ('DuckDB SQL', '16'), ('Polars', '12')]
    for i, (stage, workers) in enumerate(stages):
        x, y = 670 + (i % 2) * 280, 239 + (i // 2) * 137
        d.text(x, y, workers, 52, 'Heavy', TEAL)
        d.text(x, y + 66, stage, 23, 'Bold')
    d.line(64, 522, 1216, 522)
    d.text(64, 550, '38% lower export peak RSS', 32, 'Bold', TEAL)
    d.text(64, 596, '1M annotation rows, versus original pandas → CSV.', 22, color=MUTED)
    d.para(64, 630, 'Diverse data used 7% more RSS; streaming CSV stayed smallest. Stage allocations are not additive.', 1150, 16, MUTED)

    # 08: One architectural picture, not a specification list.
    d.slide('A modern analytical foundation', 'Query once. Keep getting value.',
            'Python 3.14.5+ with native columnar engines.', '[A2] Shared export and analytics pipeline', dark=True)
    steps = [('InterMine', 'Fetch'), ('Parquet', 'Persist'), ('DuckDB', 'Analyze'), ('Arrow', 'Exchange'), ('Polars', 'Explore')]
    for i, (name, action) in enumerate(steps):
        x = 64 + i * 234
        d.rect(x, 284, 214, 137, '#142B42', '#28465A', radius=10)
        d.text(x + 20, 308, name, 30, 'Bold', MINT)
        d.text(x + 20, 365, action, 23, color=PAPER)
        if i < 4:
            d.line(x + 214, 352, x + 233, 352, MINT, 3)
    d.text(64, 497, 'Validated pages. Typed data. Failure-safe export.', 34, 'Heavy')
    d.text(64, 565, 'One shared pipeline for legacy and native interfaces.', 27, color='#BDCBDA')

    # 09: Evidence, with compatibility scope clearly stated.
    d.slide('Confidence behind the upgrade', 'Built on tested public contracts.',
            'Validated against original InterMine and fresh package installations.', '[C1] Current candidate validation and scoped API audit')
    metrics = [('2,805', 'Source tests passed'), ('1,003', 'Installed regressions'), ('2 mines', 'Live read-only checks')]
    for i, (number, label) in enumerate(metrics):
        x = 64 + i * 394
        d.text(x, 260, number, 66, 'Heavy', TEAL)
        d.para(x, 356, label, 350, 28, INK, 'Bold')
    d.line(64, 452, 1216, 452)
    d.text(64, 493, '338 passing API scopes', 35, 'Bold')
    d.text(64, 548, 'Up from 331 in 0.2.0.', 27, color=TEAL)
    d.para(64, 606, '460 inventoried symbols; 122 have tested departures. Counts describe scoped checks, not complete equivalence.', 1150, 18, MUTED)

    # 10: A clean closing pitch, with evidence one click away.
    d.slide('intermine314 0.2.1', 'Keep your InterMine expertise.', dark=True)
    d.text(64, 182, 'Expand what your data can do.', 55, 'Heavy', MINT)
    d.para(67, 326, 'Familiar APIs.\nReusable Parquet.\nFast native analytics.', 870, 38, PAPER, 'Bold', leading=62)
    d.link(68, 553, 'Explore the project', 'https://github.com/karikris/intermine314', 26)
    d.link(427, 553, 'Compatibility guide', 'https://github.com/karikris/intermine314/blob/master/docs/source/release_readiness.rst', 26)
    d.link(838, 553, 'Benchmark evidence', 'https://github.com/karikris/intermine314/blob/master/docs/analysis/original-comparison/README.md', 26)
    d.text(68, 620, '0.2.1 release preview · Full methods and qualifications accompany this deck.', 17, color='#AFBED0')
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
    output = args.output_dir / 'intermine314-v0.2.1.pdf'
    layout = build(args.root, output)
    (args.output_dir / 'layout.json').write_text(json.dumps(layout, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
