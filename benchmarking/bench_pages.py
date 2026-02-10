from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarking.bench_runs_md import render_benchmark_run_markdown


DEFAULT_BENCHMARK_SITE_DIR = "docs/benchmarks"
_RUNS_INDEX_NAME = "runs-index.json"
_RUNS_SUBDIR = "runs"


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return text or "benchmark"


def _load_runs_index(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(loaded, list):
        return [entry for entry in loaded if isinstance(entry, dict)]
    return []


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_metadata(report: dict[str, Any], json_report_path: str) -> dict[str, Any]:
    runtime = report.get("environment", {}).get("runtime_config", {})
    timestamp = str(report.get("environment", {}).get("timestamp_utc", "unknown"))
    target = str(runtime.get("benchmark_target", "manual"))
    mine_url = str(runtime.get("mine_url", "unknown"))
    repetitions = runtime.get("repetitions", "n/a")
    return {
        "timestamp_utc": timestamp,
        "target": target,
        "mine_url": mine_url,
        "repetitions": repetitions,
        "source_json_report_path": json_report_path,
    }


def _run_id(metadata: dict[str, Any], existing: set[str]) -> str:
    timestamp = str(metadata.get("timestamp_utc", ""))
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except Exception:
        parsed = datetime.now(timezone.utc)
    target = _slugify(str(metadata.get("target", "manual")))
    base = f"{parsed.strftime('%Y%m%d-%H%M%S')}-{target}"
    candidate = base
    suffix = 2
    while candidate in existing:
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def _format_inline(text: str) -> str:
    parts = re.split(r"(`[^`]+`)", text)
    rendered: list[str] = []
    for part in parts:
        if part.startswith("`") and part.endswith("`") and len(part) >= 2:
            rendered.append(f"<code>{html.escape(part[1:-1])}</code>")
        else:
            rendered.append(html.escape(part))
    return "".join(rendered)


def _split_table_row(row: str) -> list[str]:
    stripped = row.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_alignment_row(cells: list[str]) -> bool:
    if not cells:
        return False
    for cell in cells:
        token = cell.replace("-", "").replace(":", "").strip()
        if token:
            return False
    return True


def _render_table_html(lines: list[str]) -> str:
    rows = [_split_table_row(line) for line in lines if line.strip()]
    if not rows:
        return ""
    headers = rows[0]
    body = rows[1:]
    if body and _is_alignment_row(body[0]):
        body = body[1:]
    width = len(headers)
    body_norm = [(row + [""] * max(0, width - len(row)))[:width] for row in body]
    out = ["<div class='table-wrap'><table>", "<thead><tr>"]
    out.extend(f"<th>{_format_inline(cell)}</th>" for cell in headers)
    out.append("</tr></thead><tbody>")
    for row in body_norm:
        out.append("<tr>")
        out.extend(f"<td>{_format_inline(cell)}</td>" for cell in row)
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def _markdown_lines_to_html(lines: list[str]) -> str:
    out: list[str] = []
    idx = 0
    total = len(lines)
    while idx < total:
        line = lines[idx].rstrip()
        stripped = line.strip()
        if not stripped:
            idx += 1
            continue

        if stripped.startswith("#### "):
            out.append(f"<h4>{_format_inline(stripped[5:])}</h4>")
            idx += 1
            continue
        if stripped.startswith("### "):
            out.append(f"<h3>{_format_inline(stripped[4:])}</h3>")
            idx += 1
            continue
        if stripped.startswith("## "):
            out.append(f"<h2>{_format_inline(stripped[3:])}</h2>")
            idx += 1
            continue
        if stripped.startswith("# "):
            out.append(f"<h1>{_format_inline(stripped[2:])}</h1>")
            idx += 1
            continue

        if stripped.startswith("- "):
            items: list[str] = []
            while idx < total and lines[idx].strip().startswith("- "):
                items.append(lines[idx].strip()[2:].strip())
                idx += 1
            out.append("<ul>")
            out.extend(f"<li>{_format_inline(item)}</li>" for item in items)
            out.append("</ul>")
            continue

        if stripped.startswith("|"):
            table_lines: list[str] = []
            while idx < total and lines[idx].strip().startswith("|"):
                table_lines.append(lines[idx].strip())
                idx += 1
            out.append(_render_table_html(table_lines))
            continue

        paragraph_parts = [stripped]
        idx += 1
        while idx < total:
            probe = lines[idx].strip()
            if not probe or probe.startswith(("#", "-", "|")):
                break
            paragraph_parts.append(probe)
            idx += 1
        out.append(f"<p>{_format_inline(' '.join(paragraph_parts))}</p>")

    return "\n".join(part for part in out if part)


def _page_template(*, title: str, subtitle: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f7f9fc;
      --text: #13233a;
      --muted: #5c6b82;
      --card: #ffffff;
      --line: #d6e0eb;
      --accent: #0f6cab;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: linear-gradient(120deg, #ecf3fb 0%, #f8fafc 45%, #eef6f1 100%);
      color: var(--text);
      font-family: "Segoe UI", "Noto Sans", sans-serif;
      line-height: 1.5;
    }}
    header {{
      padding: 2rem 1.5rem 1rem;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.72);
      backdrop-filter: blur(6px);
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    header h1 {{ margin: 0; font-size: 1.55rem; }}
    header p {{ margin: .3rem 0 0; color: var(--muted); }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 1.2rem 1.2rem 2rem; }}
    .table-wrap {{ overflow-x: auto; margin: 0.8rem 0 1.3rem; background: var(--card); border: 1px solid var(--line); border-radius: 10px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.93rem; min-width: 780px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 0.45rem 0.55rem; text-align: left; vertical-align: top; }}
    th {{ background: #eef4fb; position: sticky; top: 0; z-index: 1; }}
    h2, h3, h4 {{ margin-top: 1.25rem; margin-bottom: 0.45rem; }}
    ul {{ margin-top: 0.2rem; }}
    code {{
      background: #edf2f7;
      border: 1px solid #d2dbe7;
      border-radius: 4px;
      padding: 0 0.28rem;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.9em;
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .back-link {{ margin-bottom: 1rem; display: inline-block; }}
    .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: .8rem; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 0.8rem;
    }}
    .meta {{ color: var(--muted); font-size: 0.9rem; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p>{html.escape(subtitle)}</p>
  </header>
  <main>
    {body}
  </main>
</body>
</html>
"""


def _render_index(entries: list[dict[str, Any]]) -> str:
    blocks: list[str] = [
        "<p>Latest benchmark runs published from <code>benchmarking/benchmarks.py</code>.</p>",
    ]
    if not entries:
        blocks.append("<p>No benchmark runs published yet.</p>")
        return _page_template(
            title="intermine314 Benchmark Runs",
            subtitle="GitHub Pages benchmark history",
            body="\n".join(blocks),
        )

    blocks.append("<div class='card-grid'>")
    for entry in entries:
        blocks.append(
            "<article class='card'>"
            f"<h3><a href='{html.escape(str(entry.get('page', '#')))}'>{html.escape(str(entry.get('run_id', 'run')))}</a></h3>"
            f"<p class='meta'>Target: <code>{html.escape(str(entry.get('target', 'manual')))}</code></p>"
            f"<p class='meta'>Mine: <code>{html.escape(str(entry.get('mine_url', 'unknown')))}</code></p>"
            f"<p class='meta'>Timestamp: <code>{html.escape(str(entry.get('timestamp_utc', 'unknown')))}</code></p>"
            f"<p class='meta'>Repetitions: <code>{html.escape(str(entry.get('repetitions', 'n/a')))}</code></p>"
            f"<p><a href='{html.escape(str(entry.get('report_json', '#')))}'>Run JSON</a></p>"
            "</article>"
        )
    blocks.append("</div>")
    return _page_template(
        title="intermine314 Benchmark Runs",
        subtitle="GitHub Pages benchmark history",
        body="\n".join(blocks),
    )


def append_benchmark_run_pages(
    site_dir: Path,
    report: dict[str, Any],
    *,
    json_report_path: str,
) -> dict[str, str]:
    site_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = site_dir / _RUNS_SUBDIR
    runs_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / ".nojekyll").write_text("", encoding="utf-8")

    index_json_path = site_dir / _RUNS_INDEX_NAME
    entries = _load_runs_index(index_json_path)
    existing_ids = {str(entry.get("run_id", "")) for entry in entries}

    meta = _run_metadata(report, json_report_path=json_report_path)
    run_id = _run_id(meta, existing_ids)
    run_json_rel = f"{_RUNS_SUBDIR}/{run_id}.json"
    run_html_rel = f"{_RUNS_SUBDIR}/{run_id}.html"
    run_json_path = site_dir / run_json_rel
    run_html_path = site_dir / run_html_rel

    _save_json(run_json_path, report)
    markdown_lines = render_benchmark_run_markdown(report, json_report_path=run_json_rel)
    body = "<p><a class='back-link' href='../index.html'>&larr; Back to benchmark index</a></p>\n"
    body += _markdown_lines_to_html(markdown_lines)
    run_title = f"intermine314 Benchmark {meta['timestamp_utc']}"
    run_subtitle = f"Target {meta['target']} on {meta['mine_url']}"
    run_html_path.write_text(
        _page_template(title=run_title, subtitle=run_subtitle, body=body),
        encoding="utf-8",
    )

    entry = {
        "run_id": run_id,
        "timestamp_utc": meta["timestamp_utc"],
        "target": meta["target"],
        "mine_url": meta["mine_url"],
        "repetitions": meta["repetitions"],
        "page": run_html_rel,
        "report_json": run_json_rel,
        "source_json_report_path": meta["source_json_report_path"],
    }
    entries = [entry] + [old for old in entries if str(old.get("run_id")) != run_id]
    _save_json(index_json_path, entries)

    index_html_path = site_dir / "index.html"
    index_html_path.write_text(_render_index(entries), encoding="utf-8")
    return {
        "site_dir": str(site_dir),
        "index_html": str(index_html_path),
        "run_html": str(run_html_path),
        "run_json": str(run_json_path),
        "index_json": str(index_json_path),
    }
