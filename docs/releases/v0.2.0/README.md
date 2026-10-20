# intermine314 0.2 release series: original-client comparison

[Open the PDF deck](intermine314-v0.2.0.pdf) ·
[Speaker notes](speaker-notes.md) ·
[Measurements and reproduction](../../analysis/original-comparison/README.md)

This rebuilt 15-slide, 16:9 presentation compares **original InterMine 1.13.0**
with the current intermine314 source. It covers API timings, export time, absolute
process memory, Parquet/CSV/gzip storage, repeated analysis, modeled job savings,
worker budgets and compatibility boundaries. Charts are vector graphics; fonts
are embedded and source links are clickable.

**This is an unpublished source preview.** Publication was deferred at the user's
request on 21 October 2026. Version declarations remain 0.2.0. The already
published [PyPI 0.2.0](https://pypi.org/project/intermine314/0.2.0/) is a different,
earlier build. PyPI [forbids filename reuse](https://pypi.org/help/#file-name-reuse),
including after deletion; a future publication needs an unused version and a
fresh validated build.

`prepared-artifacts.json` identifies the local wheel and source archive, hashes,
clean build revision and seven verified installation modes. The files are in
the ignored workspace directory `dist/prepared-2026-10-21/`; they are suitable for
local evaluation, not an overwrite upload. Repository runtime files match the
installed candidate used by the benchmark, checked by content digest.

`publication.json`, `publication.parquet` and `pypi-smoke.json` remain historical
records of the **19 October** release. They do not certify publication of this
preview. The previous PDF and its source remain recoverable from Git revision
`75716284e42733d83eb12bf29a838aeda63ef68e`.

The new deck reads current comparison Parquet through DuckDB. `evidence.parquet`
binds headline claims to their measured denominators; `analysis.duckdb` contains
views over the comparison files (open it from the repository root).
`validation.json` records PDF checks; `preview.png` shows all slides.

Rebuild in an isolated presentation environment; these are not package
dependencies:

```bash
uv venv --python .venv/bin/python /tmp/intermine314-deck-env
uv pip install --python /tmp/intermine314-deck-env/bin/python \
  'reportlab==5.0.1' 'duckdb==1.5.6'
/tmp/intermine314-deck-env/bin/python docs/releases/v0.2.0/build_deck.py \
  --output-dir /tmp/intermine314-deck-rebuild
```

Lato is expected under `/usr/share/fonts/truetype/lato` (`--fonts` overrides it),
and DejaVu Sans Mono under `/usr/share/fonts/truetype/dejavu`. Embedded subsets
retain the fonts' respective licenses. Optional PDF inspection uses PyMuPDF.
The presentation and measurement assets are excluded from wheel and sdist.
