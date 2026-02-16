import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from intermine314.export import targeted as targeted_export


class _FakeList:
    def __init__(self, name):
        self.name = name


class _FakeIDQuery:
    def __init__(self):
        self.views = []

    def add_view(self, *views):
        self.views.extend(list(views))

    def run_parallel(self, **kwargs):
        return iter(())


class _FakeService:
    def __init__(self):
        self.created = []
        self.deleted = []

    def list_templates(self, include=None, limit=None):
        return []

    def new_query(self, _root_class):
        return _FakeIDQuery()

    def create_batched_lists(
        self,
        _identifiers,
        list_type,
        chunk_size,
        name_prefix,
        description,
        tags,
    ):
        _ = (list_type, chunk_size, name_prefix, description, tags)
        name = f"tmp_list_{len(self.created) + 1:03d}"
        self.created.append(name)
        return [_FakeList(name)]

    def delete_lists(self, names):
        self.deleted.extend(list(names))


def _fake_prepared_tables(*args, **kwargs):
    _ = (args, kwargs)
    table = targeted_export.TargetedTableSpec(
        name="core",
        root_class="Gene",
        views=["Gene.primaryIdentifier"],
        joins=[],
        template_names=[],
        template_keywords=[],
    )
    return [
        targeted_export._PreparedTableSpec(
            table=table,
            template_name=None,
            template=None,
            valid_views=["Gene.primaryIdentifier"],
            valid_joins=[],
        )
    ]


def _iter_chunks_of(n):
    def _inner(id_iter, *, list_chunk_size, row_limit):
        _ = (id_iter, list_chunk_size, row_limit)
        for idx in range(1, n + 1):
            yield [f"id{idx}"], idx

    return _inner


def _fake_chunk_report(**kwargs):
    idx = int(kwargs["chunk_index"])
    return {
        "chunk_index": idx,
        "list_name": kwargs["list_name"],
        "path": f"/tmp/chunk-{idx:03d}.parquet",
        "bytes": idx * 10,
        "rows": idx,
        "seconds": 0.25 * idx,
        "used_template": None,
        "fallback_views": ["Gene.primaryIdentifier"],
        "fallback_joins": [],
    }


class TestTargetedReporting(unittest.TestCase):
    @staticmethod
    def _table_specs():
        return [
            {
                "name": "core",
                "root_class": "Gene",
                "views": ["Gene.primaryIdentifier"],
                "joins": [],
                "template_names": [],
                "template_keywords": [],
            }
        ]

    def test_summary_mode_uses_bounded_chunk_samples(self):
        service = _FakeService()
        with patch("intermine314.export.targeted._prepare_table_plans", side_effect=_fake_prepared_tables):
            with patch("intermine314.export.targeted._iter_identifier_chunks", side_effect=_iter_chunks_of(5)):
                with patch("intermine314.export.targeted._export_table_chunk", side_effect=_fake_chunk_report):
                    report = targeted_export.export_targeted_tables_with_lists(
                        service=service,
                        root_class="Gene",
                        identifier_path="Gene.primaryIdentifier",
                        output_dir="/tmp/targeted-summary",
                        table_specs=self._table_specs(),
                        report_mode="summary",
                        report_sample_size=2,
                    )

        self.assertEqual(report["report_mode"], "summary")
        self.assertEqual(report["chunk_count"], 5)
        self.assertEqual(report["created_lists_total"], 5)
        self.assertEqual(len(report["created_lists"]), 2)
        self.assertTrue(report["created_lists_truncated"])
        table_report = report["tables"]["core"]
        self.assertEqual(table_report["chunk_count"], 5)
        self.assertEqual(len(table_report["chunks"]), 2)
        self.assertTrue(table_report["chunk_details_truncated"])
        self.assertEqual(report["totals"]["core"]["files"], 5)
        self.assertEqual(report["totals"]["core"]["rows"], 15)
        self.assertEqual(report["totals"]["core"]["bytes"], 150)
        self.assertEqual(len(service.deleted), 5)

    def test_full_mode_keeps_all_chunk_details(self):
        service = _FakeService()
        with patch("intermine314.export.targeted._prepare_table_plans", side_effect=_fake_prepared_tables):
            with patch("intermine314.export.targeted._iter_identifier_chunks", side_effect=_iter_chunks_of(3)):
                with patch("intermine314.export.targeted._export_table_chunk", side_effect=_fake_chunk_report):
                    report = targeted_export.export_targeted_tables_with_lists(
                        service=service,
                        root_class="Gene",
                        identifier_path="Gene.primaryIdentifier",
                        output_dir="/tmp/targeted-full",
                        table_specs=self._table_specs(),
                        report_mode="full",
                        report_sample_size=1,
                    )

        self.assertEqual(report["report_mode"], "full")
        self.assertEqual(len(report["created_lists"]), 3)
        self.assertFalse(report["created_lists_truncated"])
        table_report = report["tables"]["core"]
        self.assertEqual(len(table_report["chunks"]), 3)
        self.assertFalse(table_report["chunk_details_truncated"])
        self.assertEqual(report["totals"]["core"]["rows"], 6)

    def test_summary_mode_can_stream_chunk_details_to_jsonl(self):
        service = _FakeService()
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "chunks.jsonl"
            with patch("intermine314.export.targeted._prepare_table_plans", side_effect=_fake_prepared_tables):
                with patch("intermine314.export.targeted._iter_identifier_chunks", side_effect=_iter_chunks_of(4)):
                    with patch("intermine314.export.targeted._export_table_chunk", side_effect=_fake_chunk_report):
                        report = targeted_export.export_targeted_tables_with_lists(
                            service=service,
                            root_class="Gene",
                            identifier_path="Gene.primaryIdentifier",
                            output_dir=Path(tmp) / "out",
                            table_specs=self._table_specs(),
                            report_mode="summary",
                            report_sample_size=1,
                            chunk_details_jsonl_path=jsonl_path,
                        )

            self.assertEqual(report["chunk_details_jsonl_path"], str(jsonl_path))
            lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 4)
            first = json.loads(lines[0])
            self.assertEqual(first["table"], "core")
            self.assertEqual(first["chunk_index"], 1)


if __name__ == "__main__":
    unittest.main()
