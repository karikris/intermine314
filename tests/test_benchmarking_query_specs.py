from types import SimpleNamespace

from benchmarks.benchmarks import _enforce_two_outer_join_shape, resolve_query_benchmark_specs


class TestBenchmarkQuerySpecs:
    def test_enforce_two_outer_join_shape_limits_to_two_first_hops(self):
        root = "Gene"
        views = [
            "Gene.primaryIdentifier",
            "Gene.symbol",
            "Gene.transcripts.primaryIdentifier",
            "Gene.proteins.length",
            "Gene.pathways.name",
        ]
        joins = [
            "Gene.transcripts.CDS",
            "Gene.proteins.proteinDomainRegions",
            "Gene.pathways",
        ]

        selected_views, selected_joins = _enforce_two_outer_join_shape(root, views, joins)
        assert len(selected_joins) == 2
        assert all(join.count(".") == 1 for join in selected_joins)
        assert len(selected_views) >= 3

    def test_resolve_query_benchmark_specs_applies_complex_join_shape(self):
        args = SimpleNamespace(query_root=None, query_views=None, query_joins=None)
        target_settings = {
            "root_class": "Gene",
            "views": [
                "Gene.primaryIdentifier",
                "Gene.symbol",
                "Gene.transcripts.primaryIdentifier",
                "Gene.proteins.length",
                "Gene.pathways.name",
            ],
            "joins": [
                "Gene.transcripts.CDS",
                "Gene.proteins.proteinDomainRegions",
                "Gene.pathways",
            ],
        }

        specs = resolve_query_benchmark_specs(args=args, target_settings=target_settings)
        assert "simple" in specs
        assert "complex" in specs
        assert specs["simple"]["joins"] == []
        assert len(specs["complex"]["joins"]) == 2
        assert all(view.count(".") == 1 for view in specs["simple"]["views"])

