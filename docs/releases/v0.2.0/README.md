# intermine314 0.2.0 release presentation

[Open the PDF deck](intermine314-v0.2.0.pdf) ·
[Speaker notes](speaker-notes.md) ·
[PyPI](https://pypi.org/project/intermine314/0.2.0/) ·
[Release workflow](https://github.com/karikris/intermine314/actions/runs/39236010477)

A 15-slide, 16:9 presentation for scientific users, maintainers and technical
decision makers. Suggested presentation time: 12–15 minutes. Graphics and charts
are vector-based; fonts are embedded. The appendix contains clickable sources.

The deck covers shared legacy APIs, the native analytical pipeline, stage worker
budgets, memory accounting, failure-safe exports, measured Parquet compaction,
result materialization, focused development improvements and migration.
Measured slowdowns are included. Figures are explicitly separated between
published 0.1.7 source, development-uplift baselines and opt-in latency fixtures.
The retained-allocation figure specifically compares phase 1 with phase 2.

Publication was verified against the exact release artifacts and PyPI hashes.
`publication.json` records provenance; `pypi-smoke.json` records a clean public
PyPI installation and analytical smoke check. `evidence.parquet` contains typed
metric provenance. `validation.json` records PDF checks. Historical benchmarks
remain available at the immutable release revision below.

To rebuild the PDF from the repository root, use an isolated presentation
environment. These dependencies are not added to the package:

```bash
uv venv --python .venv/bin/python /tmp/intermine314-deck-env
uv pip install --python /tmp/intermine314-deck-env/bin/python \
  'reportlab==5.0.1' 'duckdb==1.5.6'
/tmp/intermine314-deck-env/bin/python docs/releases/v0.2.0/build_deck.py \
  --output-dir /tmp/intermine314-deck-rebuild
```

The generator extracts the three original Parquet inputs from Git revision
`6c5077ec0e61127b94fec277e25e1330e5441e87` into temporary storage, reads them through
DuckDB and closes both storage and connection. The clone must contain that
revision. It requires the verified publication record and an explicit output
directory; the tracked PDF and measurements remain frozen. It expects Lato fonts under
`/usr/share/fonts/truetype/lato` and DejaVu Sans Mono under
`/usr/share/fonts/truetype/dejavu`; `--fonts` overrides the Lato directory.
Lato and DejaVu retain their respective font licenses. Their embedded font
subsets provide portable PDF viewing without installing those fonts.

`layout.json` records text geometry for inspection. Optional rendering inspection
uses PyMuPDF in a separate presentation environment. `preview.png` is a contact
sheet of all 15 slides. The PDF and supporting evidence are documentation assets
and are excluded from the published package artifacts.
