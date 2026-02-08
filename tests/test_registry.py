import unittest

from intermine314 import registry
from intermine314.webservice import Registry


class RegistryTest(unittest.TestCase):
    INVALID_MINE = "__not_a_real_mine__"
    INVALID_ORGANISM = "__not_a_real_organism__"
    EXPECTED_ERROR = "No such mine available"

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
        self.assertIsNone(registry.getInfo(mine))
        self.assertEqual(registry.getInfo(self.INVALID_MINE), self.EXPECTED_ERROR)

    def test_getData(self):
        mine = self._pick_live_mine()
        self.assertIsNone(registry.getData(mine))
        self.assertEqual(registry.getData(self.INVALID_MINE), self.EXPECTED_ERROR)

    def test_getMines(self):
        self._pick_live_mine()
        self.assertIsNone(registry.getMines())
        self.assertEqual(registry.getMines(self.INVALID_ORGANISM), self.EXPECTED_ERROR)


if __name__ == "__main__":
    unittest.main()
