import unittest
from types import SimpleNamespace

from benchmarking.benchmarks import _enforce_two_outer_join_shape, resolve_query_benchmark_specs


class TestBenchmarkQuerySpecs(unittest.TestCase):
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
        self.assertEqual(len(selected_joins), 2)
        self.assertTrue(all(join.count(".") == 1 for join in selected_joins))
        self.assertGreaterEqual(len(selected_views), 3)

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
        self.assertIn("simple", specs)
        self.assertIn("complex", specs)
        self.assertEqual(specs["simple"]["joins"], [])
        self.assertEqual(len(specs["complex"]["joins"]), 2)
        self.assertTrue(all(view.count(".") == 1 for view in specs["simple"]["views"]))


if __name__ == "__main__":
    unittest.main()
