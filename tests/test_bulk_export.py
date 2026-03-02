
from intermine314.export.targeted import default_oakmine_targeted_tables, rank_template_names


class TestBulkExportHelpers:
    def test_rank_template_names_by_keyword_hits(self):
        names = [
            "Protein_GO_Associations",
            "Protein_InterPro_Domains",
            "Gene_Summary",
            "Protein_GO_InterPro",
        ]
        ranked = rank_template_names(names, ["go", "interpro"])
        assert ranked[0] == "Protein_GO_InterPro"
        assert "Protein_GO_Associations" in ranked
        assert "Protein_InterPro_Domains" in ranked

    def test_default_oakmine_targeted_tables_shape(self):
        tables = default_oakmine_targeted_tables()
        names = [table.name for table in tables]
        assert names == ["core_protein", "edge_go", "edge_domain"]
        assert all(table.views for table in tables)
        assert any("Protein.GOTerms.primaryIdentifier" in table.views for table in tables)
