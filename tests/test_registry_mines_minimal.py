from __future__ import annotations

import pytest

import intermine314.registry.mines as registry_mines


def test_normalize_service_root_strips_service_suffix():
    host, path = registry_mines._normalize_service_root("HTTPS://bar.utoronto.ca/thalemine/service")
    assert host == "bar.utoronto.ca"
    assert path == "/thalemine"


def test_matches_profile_uses_host_and_path_prefix():
    profile = {
        "host_patterns": ["bar.utoronto.ca"],
        "path_prefixes": ["/thalemine"],
    }
    assert registry_mines._matches_profile(profile, "bar.utoronto.ca", "/thalemine")
    assert registry_mines._matches_profile(profile, "bar.utoronto.ca", "/thalemine/v1")
    assert not registry_mines._matches_profile(profile, "example.org", "/thalemine")
    assert not registry_mines._matches_profile(profile, "bar.utoronto.ca", "/other")


def test_resolve_mine_user_agent_uses_config_matcher():
    user_agent = registry_mines.resolve_mine_user_agent("https://bar.utoronto.ca/thalemine/service")
    assert isinstance(user_agent, str)
    assert "Mozilla/5.0" in user_agent


def test_resolve_production_plan_enforces_elt_workflow():
    plan = registry_mines.resolve_production_plan(
        "https://maizemine.rnet.missouri.edu/maizemine/service",
        size=10_000,
        workflow="elt",
        production_profile="auto",
    )
    assert plan["workflow"] == "elt"
    assert int(plan["workers"]) > 0

    with pytest.raises(ValueError, match="workflow"):
        registry_mines.resolve_production_plan(
            "https://maizemine.rnet.missouri.edu/maizemine/service",
            size=10_000,
            workflow="legacy",
            production_profile="auto",
        )
