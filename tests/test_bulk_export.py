import unittest

from intermine314.bulk_export import default_oakmine_targeted_tables, rank_template_names


class TestBulkExportHelpers(unittest.TestCase):
    def test_rank_template_names_by_keyword_hits(self):
        names = [
            "Protein_GO_Associations",
            "Protein_InterPro_Domains",
            "Gene_Summary",
            "Protein_GO_InterPro",
        ]
        ranked = rank_template_names(names, ["go", "interpro"])
        self.assertEqual(ranked[0], "Protein_GO_InterPro")
        self.assertIn("Protein_GO_Associations", ranked)
        self.assertIn("Protein_InterPro_Domains", ranked)

    def test_default_oakmine_targeted_tables_shape(self):
        tables = default_oakmine_targeted_tables()
        names = [table.name for table in tables]
        self.assertEqual(names, ["core_protein", "edge_go", "edge_domain"])
        self.assertTrue(all(table.views for table in tables))
        self.assertTrue(any("Protein.GOTerms.primaryIdentifier" in table.views for table in tables))

