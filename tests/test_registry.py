import unittest

from intermine314 import registry
from intermine314.service import Registry


class RegistryTest(unittest.TestCase):
    INVALID_MINE = "__not_a_real_mine__"
    INVALID_ORGANISM = "__not_a_real_organism__"

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

    def test_get_info(self):
        mine = self._pick_live_mine()
        info = registry.get_info(mine)
        self.assertIsInstance(info, dict)
        with self.assertRaises(registry.RegistryLookupError):
            registry.get_info(self.INVALID_MINE)

    def test_get_data(self):
        mine = self._pick_live_mine()
        data = registry.get_data(mine)
        self.assertIsInstance(data, list)
        with self.assertRaises(registry.RegistryLookupError):
            registry.get_data(self.INVALID_MINE)

    def test_get_mines(self):
        self._pick_live_mine()
        mines = registry.get_mines()
        self.assertIsInstance(mines, list)
        bad = registry.get_mines(self.INVALID_ORGANISM)
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
