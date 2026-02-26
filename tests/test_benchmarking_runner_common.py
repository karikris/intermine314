import json
from pathlib import Path

import pytest

from benchmarks.runners import common
from benchmarks.runners.common import (
    probe_direct,
    run_import_baseline_subprocess,
    stat_summary,
    tor_proxy_observability_fields,
    validate_tor_proxy_url,
)
from intermine314.service.errors import TorConfigurationError


def test_validate_tor_proxy_url_accepts_dns_safe_scheme():
    proxy = validate_tor_proxy_url("socks5h://127.0.0.1:9050", context="test proxy")
    assert proxy == "socks5h://127.0.0.1:9050"


def test_validate_tor_proxy_url_rejects_dns_unsafe_scheme():
    with pytest.raises(TorConfigurationError, match="socks5h://"):
        validate_tor_proxy_url("socks5://127.0.0.1:9050", context="test proxy")


def test_tor_proxy_observability_fields_expose_scheme_and_dns_policy():
    fields = tor_proxy_observability_fields("socks5h://127.0.0.1:9050")
    assert fields["proxy_url"] == "socks5h://127.0.0.1:9050"
    assert fields["tor_proxy_scheme"] == "socks5h"
    assert fields["tor_dns_safety"] == "enforced"


def test_stat_summary_shape():
    summary = stat_summary([1.0, 2.0, 3.0])
    assert summary["n"] == 3.0
    assert summary["mean"] == 2.0
    assert summary["min"] == 1.0
    assert summary["max"] == 3.0
    assert summary["median"] == 2.0
    assert "stddev" in summary


def test_run_import_baseline_subprocess_parses_json_line(tmp_path):
    snippet = (
        "import json;"
        "print(json.dumps({'seconds':0.01,'module_count':7,'tracemalloc_peak_bytes':128}))"
    )
    payload = run_import_baseline_subprocess(
        import_snippet=snippet,
        repetitions=2,
        source_root=tmp_path,
    )
    assert payload["repetitions"] == 2
    assert len(payload["runs"]) == 2
    assert payload["seconds"]["mean"] == 0.01
    assert payload["module_count"]["mean"] == 7.0
    assert payload["tracemalloc_peak_bytes"]["mean"] == 128.0


def test_probe_direct_dns_failure_classification(monkeypatch):
    monkeypatch.setattr(common.socket, "getaddrinfo", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("dns")))
    probe = probe_direct("https://example.org/service", timeout_seconds=0.1)
    assert probe["mode"] == "direct"
    assert probe["reason"] == "dns_failed"
    assert probe["err_type"] == "OSError"
