# Historical plotting helpers — task 8.3

`intermine314.bar_chart` restores all five functions from the pinned InterMine
1.13.0 [bar_chart.py](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/bar_chart.py).
Adaptations use the shipped BSD-2-Clause license and NOTICE. Original positional
calls, chart labels, annotations, `show()` calls and `None` plotting returns are
preserved. No plotting call writes a CSV or other data file.

```python
from intermine314 import bar_chart

bar_chart.save_mine_and_token("humanmine", "account-token")
bar_chart.plot_go_vs_p("my-gene-list")
bar_chart.plot_go_vs_count("my-gene-list")
rows = bar_chart.get_query(query_xml)
bar_chart.query_to_barchart_log(query_xml, "true")
```

`save_mine_and_token(m, t, *, registry=None, service=None, opener=None,
**registry_options)` shares the configured managed transport architecture of the
saved-query helpers. Keyword options configure an owned Registry; supplied
Registry, Service, opener and session objects remain borrowed. Module account
state is independent of `query_manager`. Successful replacement closes previous
owned resources; failed replacement leaves no active plotting configuration.
Source error messages report only the exception class and `Check mine` or
`Check token`. Tokens do not enter helper log messages.

A local account view supplies the existing `ListManager` and `List` enrichment
implementation with the resolved root, version, compatibility profile and a
token-appending opener. It creates no extra HTTP client and works even when an
injected custom opener supplies no Service. Authentication is cleared on a
borrowed opener clone by the shared helper state; each list/enrichment URL uses
the selected saved token. Borrowed identities, headers, session ownership,
timeout, TLS, proxy/Tor configuration and custom opener behavior are preserved.
The original GO helpers constructed an unauthenticated Service, and original
raw-query requests omitted the token. Including the selected token repairs
private-list/query access, rather than silently accessing another client's
account. Selected credentials go only to the resolved mine.

GO plots call `calculate_enrichment(widget="go_enrichment_for_gene")`, retain at
most five records and close the actual stream immediately. They never consume
the sixth record merely to break the loop. As a consequence, a result with five
records does not consume or validate the subsequent stream/footer. This is a
bounded early-close repair, not a claim to validate unseen records. Identifiers
are horizontal. Both plots set x label `GO Term` and y margin 0.1:

| Function | Title | Y label / values | Annotation |
| --- | --- | --- | --- |
| `plot_go_vs_p` | `GO Term vs p-value (Label: Gene count)` | `p_value` / `p-value` | `matches` |
| `plot_go_vs_count` | `GO Term vs Count (Label: Annotation)` | `Number of Genes` / `matches` | `populationAnnotationCount` |

Raw `get_query(xml)` returns tab-split row arrays followed by the historical
empty-string sentinel when the wire ends with LF. It reads managed raw TSV
rather than the whitespace-stripping flat-file iterator, preserving spaces,
empty cells, Unicode and CR characters. An empty wire returns `['']`, as in the
source. A final row without LF is now split and plotted correctly; upstream
left that row as a string and omitted it from the chart. Only the final sentinel
is removed for plotting; malformed internal blank rows are rejected.

Query charts use column two for x, column three for y, and the first row's first
cell for the title. XML views two/three become the axis labels. Exactly
`resp == 'true'` applies natural log then two-decimal rounding with lazy Polars
expressions and y label `log(PATH)`. Other values preserve numeric y and `PATH`.
Ticks are vertical, each bar is annotated with its numeric y value, and y margin
is 0.1. Numeric whitespace, including the preserved CR from CRLF input, is
trimmed in a Polars expression before casting, matching source `float` parsing;
x labels retain their original whitespace. Polars uses natural log and
half-to-even rounding, verified against the
[pinned expression source](https://github.com/pola-rs/polars/blob/1bd8ec12/py-polars/src/polars/expr/expr.py)
and actual results: `[1, 2, 0, -1]` becomes `[0.0, 0.69, -inf, nan]`.
Nonfinite heights and annotation strings remain inspectable in the figure;
their bars and annotations are invisible because they have no finite plotting
position. Expected Matplotlib internal affine warnings during construction are
filtered locally, and actual Agg rendering succeeds with warnings treated as
errors. Log transformation itself uses no NumPy operations.

Empty enrichment, missing lists, missing/null required enrichment fields, empty
query data, short TSV rows, nonnumeric y values and invalid/insufficient XML views
raise `ValueError` before creating a figure. Missing configuration raises
`RuntimeError`. Wire/protocol errors and interrupts propagate with managed
responses closed. These explicit errors replace upstream indexing/parser
crashes and do not claim acceptance of malformed inputs.

Ordinary module imports, account configuration and raw queries import neither
analytics libraries nor Matplotlib. Plotting imports Polars and the optional
Matplotlib package lazily; missing Matplotlib raises an actionable error asking
for `pip install "intermine314[plots]"` before any plot network access. Pandas,
the original client and owned NumPy dataframe/math operations are absent.

Evidence: `tests/test_bar_chart_helpers.py` uses strict offline routes, native
and legacy profiles, actual Agg axes/patches/annotations and canvas rendering,
observed five-record consumption, borrowed Service and token/basic-auth opener
identities, custom opener-only injection, owned replacement cleanup, read and
iteration interrupts, independent account state, missing optional dependency
and analytics-blocking subprocesses. Fixtures are `plot-query-results.txt`,
`plot-enrichment.json`, and the existing registry/account snapshots. Plot-only
tests skip individually without the optional extra; raw/configuration/import
and missing-dependency cases continue running. All figure cases were executed
locally with Matplotlib 3.11.2.
