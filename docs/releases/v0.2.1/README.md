# intermine314 0.2.1 — sales presentation

[Open the PDF](intermine314-v0.2.1.pdf) · [Slide preview](preview.png) ·
[Speaker notes](speaker-notes.md) · [Benchmark methods](../../analysis/original-comparison/README.md)

A concise, ten-slide presentation for scientific users and technical decision
makers. The story moves from familiar InterMine calls to stronger compatibility,
faster analysis, smaller storage and controlled parallel execution. The visual
style uses large type, restrained colors, vector charts and embedded fonts.

The deck distinguishes two compatibility advances:

- **0.2.0:** broader historical API coverage, legacy row/object defaults and a
  shared modern analytical pipeline.
- **0.2.1:** optional pandas output in legacy mode, original result sizing,
  mutable metadata, live customization, public encoders and typed constraints.

The original-client measurements are unchanged. Headline comparisons name their
baselines; the memory slide retains the diverse-data counterexample. Detailed
methods and limitations accompany the slides in the speaker notes and benchmark
report. The earlier 15-slide deck is replaced by this presentation, with about
58% fewer words. The requested sentence about an earlier performance baseline
has been removed.

**Release preview:** this is presentation branding for the implemented 0.2.1
candidate. Package version declarations remain 0.2.0 and publication remains
deferred. No version, tag, dependency or release artifact was changed by this edit.
Published 0.2.0 records remain in [the prior release directory](../v0.2.0/README.md).

Rebuild using the isolated presentation environment:

```bash
uv pip install --python /tmp/intermine314-deck-env/bin/python \
  'reportlab==5.0.1' 'duckdb==1.5.6'
/tmp/intermine314-deck-env/bin/python docs/releases/v0.2.1/build_deck.py \
  --output-dir /tmp/intermine314-sales-rebuild
```

Create that environment first with `uv venv --python .venv/bin/python
/tmp/intermine314-deck-env` if needed. Lato fonts are expected under
`/usr/share/fonts/truetype/lato` (`--fonts` overrides this), and DejaVu Sans Mono
under `/usr/share/fonts/truetype/dejavu`; embedded subsets retain their licenses.

`evidence.parquet` retains measured claim provenance. `compatibility.parquet`
records the release-to-release behavior comparison. `analysis.duckdb` provides
views over this evidence (open it from the repository root). `layout.json` and
`validation.json` record PDF checks. All presentation assets are excluded from
package distributions. The superseded deck is recoverable from Git revision
`45a161da8b96593b5d0a75301d8381a1c0684528`.
