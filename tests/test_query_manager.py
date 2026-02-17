import unittest

import pytest

from intermine314 import query_manager as qm

pytestmark = pytest.mark.filterwarnings(
    "ignore:intermine314.query_manager module-level functions are deprecated.*:DeprecationWarning"
)


class QueryManagerTest(unittest.TestCase):
    def setUp(self):
        qm.reset_state()
        qm.save_mine_and_token("mock", "x")

    def tearDown(self):
        qm.reset_state()

    def test_get_all_query_names(self):
        # Function returns none if there is no error and mine is nonempty
        self.assertEqual(qm.get_all_query_names(), "query1")

    def test_get_query(self):
        # Function returns none if the query exists in user account
        self.assertEqual(qm.get_query("query1"), "c1, c2")
        # Function returns a message if query doesn't exists in user account
        self.assertEqual(qm.get_query("query3"), "No such query available")

    def test_delete_query(self):
        # deletes a query 'query1' if it exists and returns a message
        self.assertEqual(qm.delete_query("query1"), "query1 is deleted")
        # returns a message if query doesn't exists in user account
        self.assertEqual(qm.delete_query("query3"), "No such query available")

    def test_post_query(self):
        # posts a query if xml is right
        self.assertEqual(qm.post_query('<query name="query3"></query>'), "query3 is posted")
        # can't post if xml is wrong and returns a message
        self.assertEqual(qm.post_query('<query name="query4"></query>'), "Incorrect format")

    def test_query_manager_instances_do_not_share_tokens(self):
        first = qm.QueryManager()
        second = qm.QueryManager()

        first.save_mine_and_token("mock", "token_a")
        second.save_mine_and_token("mock", "token_b")

        self.assertEqual(first.get_saved_credentials(), ("mock", "token_a"))
        self.assertEqual(second.get_saved_credentials(), ("mock", "token_b"))

    def test_reset_state_clears_legacy_credentials(self):
        qm.save_mine_and_token("mock", "secret_token")
        self.assertEqual(qm.get_saved_credentials(), ("mock", "secret_token"))

        qm.reset_state()

        self.assertEqual(qm.get_saved_credentials(), ("", ""))


if __name__ == "__main__":
    unittest.main()
