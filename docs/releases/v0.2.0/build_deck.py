"""Build the 16:9 release deck from versioned Parquet evidence.

Presentation-only dependencies: reportlab==5.0.1 and duckdb==1.5.6.
Run from docs/releases/v0.2.0 after publication verification is recorded.
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
        self.text(64, 680, "intermine314 / 0.2.0", 13, "Bold", footer)
        if source:
            self.text(300, 680, source, 12, color=footer)
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
    c = duckdb.connect()
    rows = c.execute(
        'SELECT "case",variant,median(ns)/1e6,median(total_bytes),median(parts) '
        'FROM read_parquet(?) GROUP BY "case",variant',
        [str(root / "docs/analysis/release-v020-vs017/trials.parquet")],
    ).fetchall()
    trials = {(r[0], r[1]): dict(ms=r[2], bytes=r[3], parts=r[4]) for r in rows}
    rows = c.execute(
        "SELECT \"case\",speedup FROM read_parquet(?) WHERE variant='candidate'",
        [str(root / "docs/analysis/uplift-2026-10-14/final_summary.parquet")],
    ).fetchall()
    uplift = dict(rows)
    memory = c.execute(
        "SELECT * FROM read_parquet(?) LIMIT 1",
        [str(root / "docs/analysis/uplift-2026-10-14/phase2_memory_trials.parquet")],
    ).description
    assert memory
    return trials, uplift


def build(root, output, publication):
    trials, uplift = evidence(root)

    def t(case, variant):
        return trials[(case, variant)]["ms"]

    d = Deck(output)
    # 01 — A concise release proposition.
    d.slide(
        "Release / 19 October 2026",
        "Familiar calls.",
        source="Python 3.14.5+ / published on PyPI",
        dark=True,
    )
    d.text(64, 145, "Modern analytics.", 65, "Heavy", MINT)
    d.text(64, 259, "intermine314", 34, "Bold")
    d.text(64, 300, "0.2.0", 109, "Heavy")
    d.para(
        66,
        444,
        "Legacy InterMine workflows, shared execution and a resource-aware columnar pipeline.",
        690,
        26,
        "#BDCBDA",
    )
    for i, label in enumerate(("PARQUET", "DUCKDB", "ARROW", "POLARS")):
        y = 232 + i * 85
        d.rect(889, y, 292, 62, "#142B42", "#28465A", radius=12)
        d.text(919, y + 17, label, 21, "Bold", MINT if i == 0 else PAPER)
        if i < 3:
            d.line(1035, y + 62, 1035, y + 85, "#456474", 2)

    # 02 — Explain the product before its measurements.
    d.slide(
        "Architecture",
        "One shared runtime, end to end.",
        "Keep the query workflow. Move analytics into native columnar engines.",
        "[A1] Architecture and migration guide",
    )
    stages = [
        ("InterMine", "Query / templates"),
        ("Typed batches", "Validate / convert"),
        ("Parquet", "Persist / compact"),
        ("DuckDB", "Filter / aggregate"),
        ("Arrow + Polars", "Read / analyze"),
    ]
    for i, (name, caption) in enumerate(stages):
        x = 64 + i * 234
        d.rect(
            x,
            255,
            214,
            154,
            TEAL if i == 2 else WHITE,
            None if i == 2 else LINE,
            radius=14,
        )
        d.text(x + 17, 282, name, 22, "Bold", WHITE if i == 2 else INK)
        d.para(x + 17, 332, caption, 179, 18, WHITE if i == 2 else MUTED)
        if i < 4:
            d.text(x + 218, 312, "›", 25, "Bold", TEAL)
    d.para(
        65,
        475,
        "Python coordinates transport, compatibility and ownership. Polars, DuckDB and Arrow do the native analytical work.",
        1080,
        25,
    )
    d.pill(65, 572, "CSV INGESTION AND SPAWNED PARQUET TRANSFORMS ALSO SUPPORTED")

    # 03 — Counts describe evidence scopes, never application compatibility.
    d.slide(
        "Compatibility",
        "Legacy workflows, canonical execution.",
        "Change the package import and retain the historical service entry point.",
        "[A1] 460-symbol audit / 40 supplemental bindings",
    )
    d.rect(64, 231, 683, 176, NAVY, radius=12)
    d.text(88, 257, "from intermine314.webservice import Service", 20, "Code", MINT)
    d.text(88, 306, 'query = service.select("Gene.symbol")', 20, "Code", PAPER)
    d.text(88, 354, 'query.export("genes.parquet")', 20, "Code", PAPER)
    d.para(
        66,
        449,
        "Queries and expressions · saved XML · templates · lists and summaries · identifier jobs · registry helpers",
        676,
        25,
    )
    d.card(
        785,
        231,
        430,
        340,
        "460",
        "historical symbols inventoried",
        "331 passing scopes; 129 tested departures. Coverage is scoped, not universal equivalence.",
    )
    d.text(
        65,
        606,
        "Native Service keeps dictionaries; the historical facade keeps legacy row/object defaults.",
        19,
        color=MUTED,
    )

    # 04 — Worker counts are a policy, not a benchmark promise.
    d.slide(
        "Parallel processing",
        "A CPU budget with stage-specific workers.",
        "Default native CPU budget: 16. Smaller profiles: 8, 6, 4 and 2.",
        "[A1] Resource profiles / advisory package admission",
    )
    workers = [
        ("Public-mine HTTP", "4", "Per-host permits"),
        ("Arrow CPU / I/O", "4 / 2", "Separate native pools"),
        ("DuckDB SQL", "16", "One main native query"),
        ("Polars", "12", "One streaming plan"),
        ("Coarse Python tasks", "4", "Spawned path-based work"),
        ("Mutable resource", "1", "One owner per target"),
    ]
    for i, (label, value, detail) in enumerate(workers):
        x, y = 64 + i % 3 * 389, 232 + i // 3 * 174
        d.rect(x, y, 371, 155, WHITE, LINE, radius=12)
        d.text(x + 20, y + 17, value, 43, "Heavy", TEAL)
        d.text(x + 20, y + 77, label, 22, "Bold")
        d.text(x + 20, y + 113, detail, 17, color=MUTED)
    d.para(
        65,
        606,
        "Allocations are stage choices, not additive pools. Actual CPU visibility, engine initialization and server limits still matter.",
        1138,
        18,
        MUTED,
    )

    # 05 — Show the bound's lifetime explicitly.
    d.slide(
        "Memory and scheduling",
        "Keep work charged until it is consumed.",
        "Pending, completed and emitting pages all belong to the outstanding-page envelope.",
        "[A1] Inflight accounting / streaming XML / shared result layouts",
    )
    states = [
        ("FETCH", "Pending requests"),
        ("WAIT", "Completed pages"),
        ("EMIT", "Ordered consumption"),
    ]
    for i, (state, detail) in enumerate(states):
        x = 64 + i * 390
        d.rect(x, 238, 367, 150, WHITE, LINE, radius=12)
        d.text(x + 22, 264, state, 36, "Heavy", TEAL)
        d.text(x + 22, 327, detail, 22)
    d.line(86, 419, 1190, 419, TEAL, 3)
    d.text(
        276,
        436,
        "Reservation remains active across the whole page lifecycle",
        24,
        "Bold",
        TEAL,
    )
    d.para(
        65,
        505,
        "Shared immutable row layouts reduce repeated metadata. Sized decompressed XML reads and bounded upload/transform queues reduce avoidable buffering.",
        1130,
        25,
    )
    d.text(
        65,
        610,
        "Byte caps are estimates, not RSS guarantees; one oversized page may proceed to avoid deadlock.",
        18,
        color=MUTED,
    )

    # 06 — Real reproduced correctness, not synthetic marketing.
    d.slide(
        "Reliability",
        "Three concrete failure modes addressed.",
        "Accept validated data; preserve previous output when work fails.",
        "[A1] Fresh deterministic probes against published 0.1.7 source",
    )
    for i, (number, title, body) in enumerate(
        (
            (
                "01",
                "Server status",
                "A page ending in wasSuccessful=false is rejected before its rows are accepted.",
            ),
            (
                "02",
                "Compressed XML",
                "Sized reads decompress response bytes before metadata reaches the XML parser.",
            ),
            (
                "03",
                "Failed exports",
                "Staging and validation preserve the existing Parquet output after producer failure.",
            ),
        )
    ):
        x = 64 + i * 391
        d.rect(x, 235, 369, 348, WHITE, LINE, radius=14)
        d.text(x + 24, 260, number, 52, "Heavy", TEAL)
        d.text(x + 24, 346, title, 25, "Bold")
        d.para(x + 24, 404, body, 319, 22, MUTED)
    d.text(
        65,
        608,
        "Publication is guarded per target; portable directory fallback is not a crash/cross-process transaction.",
        17,
        color=MUTED,
    )

    # 07 — Whole-directory bytes include the new manifest.
    old = trials[("export_directory", "0.1.7")]
    new = trials[("export_directory", "candidate")]
    saved = 100 * (1 - new["bytes"] / old["bytes"])
    d.slide(
        "Storage / published 0.1.7 comparison",
        "Smaller Parquet. Far fewer small files.",
        "100,000-row integer/string fixture; total bytes include the candidate manifest.",
        "[A1] Matched modern dependencies / 21 paired local trials",
    )
    d.bar(
        64,
        245,
        "0.1.7 source",
        old["bytes"],
        old["bytes"],
        630,
        "#8291A0",
        f"{old['bytes']:,.0f} bytes / 100 files",
    )
    d.bar(
        64,
        419,
        "0.2.0 runtime",
        new["bytes"],
        old["bytes"],
        630,
        TEAL,
        f"{new['bytes']:,.0f} bytes / 1 data file",
    )
    d.text(804, 258, f"{saved:.0f}%", 100, "Heavy", TEAL)
    d.text(810, 385, "less total storage", 27, "Bold")
    d.text(810, 458, "100 → 1", 53, "Heavy", TEAL)
    d.text(810, 532, "data files", 24)
    d.text(
        65,
        613,
        "Compaction trades write time for layout: this fixture took 75% more time. Savings depend on the dataset.",
        18,
        color=MUTED,
    )

    # 08 — A defensible comparison with published source.
    before, after = t("dict_list", "0.1.7"), t("dict_list", "candidate")
    d.slide(
        "Results / published 0.1.7 comparison",
        "One materialization. One response.",
        "10,000 dictionary rows × 20 fields, on identical modern dependencies.",
        "[A1] Median of 21 paired processes / local in-memory response",
    )
    d.text(64, 240, f"{100 * (1 - after / before):.0f}%", 109, "Heavy", TEAL)
    d.text(68, 376, "less elapsed time", 29, "Bold")
    d.text(68, 462, f"{before:.2f} ms → {after:.2f} ms", 37, "Heavy")
    d.rect(752, 239, 463, 311, WHITE, LINE, radius=14)
    d.text(780, 268, "2 → 1", 78, "Heavy", TEAL)
    d.text(784, 377, "requests", 27, "Bold")
    d.para(
        783,
        434,
        "Unsized streams remove the extra count request during list(results).",
        400,
        23,
        MUTED,
    )
    d.para(
        66,
        591,
        "Network latency is excluded. Result iterators no longer support len(); use explicit query counts when needed.",
        1120,
        19,
        MUTED,
    )

    # 09 — Deliberately distinct baseline, visible at both top and bottom.
    d.slide(
        "Optimization / development-baseline comparison",
        "Focused improvements across the pipeline.",
        "Development uplift: timing baseline 8e584c5; retained-allocation comparison uses the phase 1 baseline.",
        "[A2] Timing: 8e584c5; allocation: phase 1 → 2 / neither is PyPI 0.1.7",
    )
    cards = [
        (f"{uplift['keyed_rows']:.2f}×", "Keyed-row access", "10k rows × 20 fields"),
        (
            f"{uplift['numeric_export']:.2f}×",
            "Eligible numeric export",
            "Single staged numeric part",
        ),
        (
            f"{uplift['repeated_sql']:.2f}×",
            "Repeated small SQL",
            "Opt-in borrowed connection",
        ),
        ("84.7%", "Less retained Python allocation", "Shared layout; phase 1 → 2"),
    ]
    for i, (value, label, detail) in enumerate(cards):
        x = 64 + i * 291
        d.card(x, 245, 274, 330, value, label, detail, ORANGE)
    d.para(
        65,
        605,
        "Python allocation is not RSS. SQL setup reuse is opt-in. Whole-job gains depend on stage costs and input data.",
        1140,
        18,
        MUTED,
    )

    # 10 — Workflow benefits are opt-in demonstrations.
    d.slide(
        "Workflow design / illustrative latency fixtures",
        "Fewer requests. Less repeated setup.",
        "Opt-in workflow demonstrations with simulated 5 ms request latency; equal results checked.",
        "[A3] Workflow fixtures / not public-mine throughput measurements",
    )
    for i, (value, title, detail) in enumerate(
        (
            ("32 → 4", "Object reads", "Batch prefetch; 180.2 → 22.9 ms"),
            ("40 → 2", "Service reuse", "Reuse metadata; 222.1 → 11.1 ms"),
            ("4 mines", "Concurrent discovery", "Same 12 requests; 67.4 → 19.1 ms"),
        )
    ):
        d.card(64 + i * 391, 243, 369, 330, value, title, detail, PURPLE)
    d.para(
        65,
        606,
        "Prefetch is explicit. Cache initialization and resource lifetime remain guarded; mutations retain one owner.",
        1140,
        18,
        MUTED,
    )

    # 11 — Performance trade-offs belong in the main narrative.
    d.slide(
        "Performance boundaries",
        "Match the path to the workload.",
        "Published 0.1.7 source comparison; identical modern dependencies, local fixtures.",
        "[A1] 21 paired processes / custom writers bypass HTTP columnar fast path",
    )
    columns = [(80, "Workload"), (697, "0.1.7"), (865, "0.2.0"), (1047, "Time change")]
    for x, label in columns:
        d.text(x, 228, label, 18, "Bold", MUTED)
    names = [
        ("dict_list", "Dictionary materialization"),
        ("dict_iter", "Dictionary streaming"),
        ("export_single_numeric", "Custom numeric single-file export"),
        ("export_directory", "Custom directory export"),
    ]
    for i, (case, label) in enumerate(names):
        y = 274 + i * 69
        d.line(65, y - 12, 1215, y - 12)
        old_ms, new_ms = t(case, "0.1.7"), t(case, "candidate")
        delta = 100 * (new_ms / old_ms - 1)
        d.text(80, y, label, 21, "Bold")
        d.text(697, y, f"{old_ms:.2f} ms", 20)
        d.text(865, y, f"{new_ms:.2f} ms", 20)
        d.text(1047, y, f"{delta:+.0f}%", 23, "Bold", TEAL if delta < 0 else ORANGE)
    d.para(
        65,
        580,
        "These measurements do not establish a universal speedup or memory reduction. Directory compaction improves layout while adding write work.",
        1130,
        22,
        MUTED,
    )

    # 12 — Verification facts refer to the release workflow.
    d.slide(
        "Release assurance",
        "Validated from source to installed wheel.",
        "The matching tag triggers exact-commit checks before verified artifacts reach PyPI.",
        "[A4] Release CI / artifact policy / PyPI hash verification",
    )
    for i, (value, title, detail) in enumerate(
        (
            ("2,813", "Source tests", "Full local suite + API trace"),
            ("904", "Installed regressions", "Source-package imports excluded"),
            ("6", "Installation modes", "Wheel/sdist, extras and docs"),
        )
    ):
        d.card(64 + i * 391, 235, 369, 310, value, title, detail)
    d.para(
        65,
        574,
        "Version consistency → master ancestry → exact artifact ID → SHA-256 verification → trusted-publisher upload",
        1140,
        23,
        TEAL,
        "Bold",
    )
    d.text(
        65,
        623,
        "Validated platform: Linux CPython 3.14.5+. Other OSes and free-threaded builds remain outside certification.",
        17,
        color=MUTED,
    )

    # 13 — Adoption needs explicit, actionable boundaries.
    d.slide(
        "Migration",
        "A clear upgrade contract.",
        "Use the migration guide alongside the familiar API surface.",
        "[A1] Migration checklist / release notes",
    )
    changes = [
        (
            "01",
            "Choose the service profile",
            "Historical webservice import: legacy defaults. Native service import: dictionaries.",
        ),
        (
            "02",
            "Move pandas workflows to Polars",
            "dataframe() returns Polars; update package imports and dependency declarations.",
        ),
        (
            "03",
            "Review stream and helper changes",
            "Unsized streams; immutable streamed metadata; removed old policy/Tor helper modules.",
        ),
        (
            "04",
            "Upgrade the Python environment",
            "Python 3.14.5+; modern engine minima; PyArrow is a core dependency.",
        ),
    ]
    for i, (number, title, body) in enumerate(changes):
        y = 221 + i * 99
        d.text(66, y + 1, number, 32, "Heavy", TEAL)
        d.text(130, y, title, 24, "Bold")
        d.text(130, y + 40, body, 19, color=MUTED)

    # 14 — Finish with a useful action, not a generic thank-you.
    d.slide(
        "Available on PyPI",
        "Start with your existing query.",
        "Adopt the modern pipeline one workflow at a time.",
        "Release 0.2.0 / 19 October 2026",
        dark=True,
    )
    d.rect(64, 231, 1152, 102, "#142B42", "#29445A", radius=12)
    d.text(90, 265, "python -m pip install intermine314==0.2.0", 28, "Code", MINT)
    d.para(
        67,
        387,
        "1. Retain familiar service and query calls.\n2. Export validated data to Parquet.\n3. Use DuckDB and Polars for repeated analysis.",
        1000,
        28,
        PAPER,
        leading=54,
    )
    d.link(
        68,
        580,
        "pypi.org/project/intermine314/0.2.0",
        "https://pypi.org/project/intermine314/0.2.0/",
        24,
    )
    d.text(
        68,
        621,
        "karikris/intermine314 · Python 3.14.5+ · MIT AND BSD-2-Clause",
        17,
        color="#AFBED0",
    )

    # 15 — Accessible, clickable sources and explicit methodology.
    d.slide(
        "Evidence appendix",
        "Every headline has a named baseline.",
        "Measurements are local observations, not public-server capacity or whole-job guarantees.",
        "Audited 14 Oct 2026 / release verified 19 Oct 2026",
    )
    base = "https://github.com/karikris/intermine314/blob/v0.2.0/"
    refs = [
        (
            "[A1] Published 0.1.7 comparison + migration",
            base + "docs/analysis/release-v020-vs017/report.md",
            "Matched modern dependencies; 21 pairs; fixtures, memory limits and raw Parquet.",
        ),
        (
            "[A2] Development uplift + phase 2 memory comparison",
            base + "docs/analysis/uplift-2026-10-14/report.md",
            "Separate 21-pair timings; shared-layout retained allocation; opt-in SQL reuse.",
        ),
        (
            "[A3] Opt-in workflow demonstrations",
            base + "docs/analysis/uplift-2026-10-14/phase5-experiments.md",
            "Simulated 5 ms request latency; prefetch, discovery and Service reuse.",
        ),
        (
            "[A4] Exact release workflow + publication evidence",
            publication["workflow_url"],
            "Tagged commit, six installation modes, installed regressions and artifact hashes.",
        ),
    ]
    for i, (label, url, detail) in enumerate(refs):
        y = 223 + i * 87
        d.link(67, y, label, url, 21)
        d.text(67, y + 35, detail, 17, color=MUTED)
    d.para(
        67,
        589,
        "Benchmark host: WSL2, 18 visible CPUs; CPython 3.14.5; JIT off. No attribution to Python 3.14 itself: both sides used the same interpreter.",
        1130,
        19,
        MUTED,
    )
    if d.page != TOTAL:
        raise ValueError("Unexpected slide count")
    d.c.save()
    return d.layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument(
        "--fonts", type=Path, default=Path("/usr/share/fonts/truetype/lato")
    )
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    register_fonts(args.fonts)
    publication = json.loads((here / "publication.json").read_text())
    if publication["version"] != "0.2.0" or publication["status"] != "verified":
        raise ValueError("A verified 0.2.0 publication record is required")
    output = here / "intermine314-v0.2.0.pdf"
    layout = build(args.root, output, publication)
    (here / "layout.json").write_text(json.dumps(layout, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
