import unittest


class LiveRegistryTest(unittest.TestCase):
    @unittest.skip("Registry live API coverage is currently disabled in this suite")
    def testAccessRegistry(self):
        pass


if __name__ == "__main__":
    unittest.main()
