from __future__ import annotations

from types import SimpleNamespace

from benchmarks.runners import run_live


def test_run_live_skips_in_ci_when_not_opted_in(monkeypatch, capsys):
    monkeypatch.setenv("CI", "1")
    monkeypatch.delenv("RUN_LIVE", raising=False)

    code = run_live.run([])

    assert code == run_live.SKIP_EXIT_CODE
    assert "preflight_skip reason=ci_disabled" in capsys.readouterr().out


def test_run_live_returns_skip_when_preflight_fails(monkeypatch, capsys):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("RUN_LIVE", "1")
    monkeypatch.setattr(
        run_live,
        "_parse_benchmark_args",
        lambda _args: SimpleNamespace(mine_url="https://example.org/mine", benchmark_target="auto"),
    )
    monkeypatch.setattr(run_live, "_candidate_urls", lambda _args: ["https://example.org/mine"])
    monkeypatch.setattr(run_live, "_preflight", lambda *_args, **_kwargs: (False, None))

    code = run_live.run([])

    assert code == run_live.SKIP_EXIT_CODE
    assert "preflight_skip reason=environment" in capsys.readouterr().out


def test_run_live_preflight_only_short_circuits_benchmark(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("RUN_LIVE", "1")
    monkeypatch.setattr(
        run_live,
        "_parse_benchmark_args",
        lambda _args: SimpleNamespace(mine_url="https://example.org/mine", benchmark_target="auto"),
    )
    monkeypatch.setattr(run_live, "_candidate_urls", lambda _args: ["https://example.org/mine"])
    monkeypatch.setattr(run_live, "_preflight", lambda *_args, **_kwargs: (True, "https://example.org/mine"))
    monkeypatch.setattr(
        run_live,
        "_run_benchmark",
        lambda _args: (_ for _ in ()).throw(AssertionError("benchmark should not run in preflight-only mode")),
    )

    code = run_live.run(["--preflight-only"])

    assert code == run_live.SUCCESS_EXIT_CODE


def test_run_live_injects_selected_candidate_into_benchmark_args(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("RUN_LIVE", "1")
    monkeypatch.setattr(
        run_live,
        "_parse_benchmark_args",
        lambda _args: SimpleNamespace(mine_url="https://seed.example/mine", benchmark_target="auto"),
    )
    monkeypatch.setattr(run_live, "_candidate_urls", lambda _args: ["https://seed.example/mine"])
    monkeypatch.setattr(run_live, "_preflight", lambda *_args, **_kwargs: (True, "https://picked.example/mine"))

    captured = {}

    def _capture(args):
        captured["args"] = list(args)
        return run_live.SUCCESS_EXIT_CODE

    monkeypatch.setattr(run_live, "_run_benchmark", _capture)

    code = run_live.run(["--benchmark-target", "auto", "--mine-url", "https://seed.example/mine"])

    assert code == run_live.SUCCESS_EXIT_CODE
    assert captured["args"][-2:] == ["--mine-url", "https://picked.example/mine"]
