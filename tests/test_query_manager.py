import unittest
from types import SimpleNamespace

from intermine314.service.query_manager import (
    INCORRECT_FORMAT,
    NO_SAVED_QUERIES,
    NO_SUCH_QUERY_AVAILABLE,
    QueryManager,
)


class QueryManagerTest(unittest.TestCase):
    def test_query_manager_instances_do_not_share_tokens(self):
        first = QueryManager()
        second = QueryManager()

        first._resolve_service_root = lambda _mine: "https://example.org/service"
        first._get_user_queries = lambda _root, _token: {"queries": {}}
        second._resolve_service_root = lambda _mine: "https://example.org/service"
        second._get_user_queries = lambda _root, _token: {"queries": {}}

        first.save_mine_and_token("mine_a", "token_a")
        second.save_mine_and_token("mine_b", "token_b")

        self.assertEqual(first.get_saved_credentials(), ("mine_a", "token_a"))
        self.assertEqual(second.get_saved_credentials(), ("mine_b", "token_b"))

    def test_clear_credentials_only_clears_instance_state(self):
        manager = QueryManager()
        manager._resolve_service_root = lambda _mine: "https://example.org/service"
        manager._get_user_queries = lambda _root, _token: {"queries": {}}
        manager.save_mine_and_token("mine_a", "token_a")

        manager.clear_credentials()

        self.assertEqual(manager.get_saved_credentials(), ("", ""))

    def test_get_all_query_names_handles_empty_and_non_empty_payloads(self):
        manager = QueryManager()
        manager._mine = "mine_a"
        manager._token = "token_a"
        manager._resolve_service_root = lambda _mine: "https://example.org/service"
        manager._get_user_queries = lambda _root, _token: {"queries": {"query1": {}, "query2": {}}}
        self.assertEqual(manager.get_all_query_names(), "query1, query2")

        manager._get_user_queries = lambda _root, _token: {"queries": {}}
        self.assertEqual(manager.get_all_query_names(), NO_SAVED_QUERIES)

    def test_get_query_and_delete_query_status_messages(self):
        manager = QueryManager()
        manager._mine = "mine_a"
        manager._token = "token_a"
        manager._resolve_service_root = lambda _mine: "https://example.org/service"
        manager._get_user_queries = lambda _root, _token: {"queries": {"query1": {}}}

        manager._request = lambda *_args, **_kwargs: SimpleNamespace(text="<saved-queries></saved-queries>")
        self.assertEqual(manager.get_query("query_missing"), NO_SUCH_QUERY_AVAILABLE)

        manager._request = lambda *_args, **_kwargs: SimpleNamespace(text="<query name='query1'/>")
        self.assertEqual(manager.get_query("query1"), "<query name='query1'/>")

        self.assertEqual(manager.delete_query("query_missing"), NO_SUCH_QUERY_AVAILABLE)

        manager._request = lambda *_args, **_kwargs: SimpleNamespace()
        self.assertEqual(manager.delete_query("query1"), "query1 is deleted")

    def test_save_mine_and_token_error_messages(self):
        manager = QueryManager()
        manager._resolve_service_root = lambda _mine: (_ for _ in ()).throw(RuntimeError("bad mine"))
        result = manager.save_mine_and_token("mine_a", "token_a")
        self.assertIn("Check mine", result)

        manager = QueryManager()
        manager._resolve_service_root = lambda _mine: "https://example.org/service"
        manager._get_user_queries = lambda _root, _token: (_ for _ in ()).throw(RuntimeError("bad token"))
        result = manager.save_mine_and_token("mine_a", "token_a")
        self.assertIn("Check token", result)

    def test_post_query_rejects_invalid_insert_when_not_visible_after_put(self):
        manager = QueryManager()
        manager._mine = "mine_a"
        manager._token = "token_a"
        manager._resolve_service_root = lambda _mine: "https://example.org/service"
        manager._service_version = lambda *_args, **_kwargs: 27
        manager._get_user_queries = lambda _root, _token: {"queries": {"existing": {}}}
        manager._request = lambda *_args, **_kwargs: SimpleNamespace()

        result = manager.post_query('<query name="query3"></query>', input_func=lambda _prompt: "n")
        self.assertEqual(result, INCORRECT_FORMAT)


if __name__ == "__main__":
    unittest.main()
