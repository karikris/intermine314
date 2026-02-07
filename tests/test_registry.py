import unittest

from intermine314 import registry
from intermine314.webservice import Registry


class RegistryTest(unittest.TestCase):
    @staticmethod
    def _pick_live_mine():
        try:
            mines = Registry().all_mines()
        except Exception as exc:
            raise unittest.SkipTest(f"Registry unavailable: {exc}")

        if not mines:
            raise unittest.SkipTest("Registry returned no mines")

        preferred = ("FlyMine", "MaizeMine", "ThaleMine", "LegumeMine", "OakMine", "WheatMine")
        by_lower = {}
        for mine in mines:
            if isinstance(mine, dict) and mine.get("name"):
                by_lower[mine["name"].lower()] = mine["name"]

        for name in preferred:
            if name.lower() in by_lower:
                return by_lower[name.lower()]

        for mine in mines:
            if isinstance(mine, dict) and mine.get("name"):
                return mine["name"]

        raise unittest.SkipTest("No mine names available from registry")

    def test_getInfo(self):
        mine = self._pick_live_mine()
        # function returns none is everything runs fine
        self.assertEqual(registry.getInfo(mine), None)
        # function returns a message if anything goes wrong,
        # example: mine is not correct
        self.assertEqual(registry.getInfo("__not_a_real_mine__"), "No such mine available")

    def test_getData(self):
        mine = self._pick_live_mine()
        # function returns none is everything runs fine
        self.assertEqual(registry.getData(mine), None)
        # function returns a message if anything goes wrong,
        # example: mine is not correct
        self.assertEqual(registry.getData("__not_a_real_mine__"), "No such mine available")

    def test_getMines(self):
        self._pick_live_mine()
        # function returns none is everything runs fine
        self.assertEqual(registry.getMines(), None)
        # function returns a message if anything goes wrong,
        # example: organism name is not correct
        self.assertEqual(registry.getMines("__not_a_real_organism__"), "No such mine available")


if __name__ == "__main__":
    unittest.main()
