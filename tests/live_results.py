from intermine314.errors import WebserviceError
from intermine314.webservice import Service
from functools import reduce
import unittest
import csv
import os
import uuid


class LiveResultsTest(unittest.TestCase):
    TEST_ROOT = os.getenv("TESTMODEL_URL", "http://localhost:8080/intermine-demo/service")

    SERVICE = Service(TEST_ROOT)

    def setUp(self):
        self.manager_q = self.SERVICE.select("Manager.age", "Manager.name")
        self.manager_age_sum = 1383

    def testLazyReferenceFetchingManagers(self):
        departments = self.SERVICE.select("Department.*").results()
        managers = [d.manager.name for d in departments]
        expected = [
            "EmployeeA1",
            "EmployeeB1",
            "EmployeeB3",
            "Jennifer Taylor-Clarke",
            "David Brent",
            "Keith Bishop",
            "Glynn Williams",
            "Neil Godwin",
            "Tatjana Berkel",
            "Sinan Turçulu",
            "Bernd Stromberg",
            "Timo Becker",
            "Dr. Stefan Heinemann",
            "Burkhardt Wutke",
            "Frank Möllers",
            "Charles Miner",
            "Michael Scott",
            "Angela",
            "Lonnis Collins",
            "Meredith Palmer",
            "Juliette Lebrac",
            "Gilles Triquet",
            "Jacques Plagnol Jacques",
            "Didier Leguélec",
            "Joel Liotard",
            "Bwa'h Ha Ha",
            "Quote Leader",
            "Separator Leader",
            "Slash Leader",
            "XML Leader",
        ]

        self.assertEqual(expected, managers)

    def testLazyReferenceFetchingEmployeeDetails(self):
        dave = self.SERVICE.select("Employee.*").where(name="David Brent").one()
        self.assertEqual("Sales", dave.department.name)
        self.assertIsNotNone(dave.address)

        # Can handle null references.
        b1 = self.SERVICE.select("Employee.*").where(name="EmployeeB1").one()
        self.assertIsNone(b1.address)

    def testLazyCollectionFetching(self):
        results = self.SERVICE.select("Department.*").results()
        age_sum = reduce(lambda x, y: x + reduce(lambda a, b: a + b.age, y.employees, 0), results, 0)
        self.assertEqual(5924, age_sum)

        # Can handle empty collections as well as populated ones.
        banks = self.SERVICE.select("Bank.*").results()
        self.assertEqual([1, 0, 0, 2, 2], [len(bank.corporateCustomers) for bank in banks])

    def assertManagerAgeIsSum(self, fmt, accessor):
        total = sum(accessor(x) for x in self.manager_q.results(row=fmt))
        self.assertEqual(self.manager_age_sum, total)

    def test_attr_access(self):
        for synonym in ["object", "objects", "jsonobjects"]:
            self.assertManagerAgeIsSum(synonym, lambda row: row.age)

    def test_rr_indexed_access(self):
        self.assertManagerAgeIsSum("rr", lambda row: row["age"])
        self.assertManagerAgeIsSum("rr", lambda row: row[0])

    def test_row_as_function(self):
        self.assertManagerAgeIsSum("rr", lambda row: row("age"))
        self.assertManagerAgeIsSum("rr", lambda row: row(0))

    def test_dict_row(self):
        self.assertManagerAgeIsSum("dict", lambda row: row["Manager.age"])

    def test_list_row(self):
        self.assertManagerAgeIsSum("list", lambda row: row[0])

    def test_json_rows(self):
        self.assertManagerAgeIsSum("jsonrows", lambda row: row[0]["value"])

    def test_csv(self):
        results = self.manager_q.results(row="csv")
        reader = csv.reader(results, delimiter=",", quotechar='"')
        self.assertEqual(self.manager_age_sum, sum(int(row[0]) for row in reader))

    def test_tsv(self):
        results = self.manager_q.results(row="tsv")
        reader = csv.reader(results, delimiter="\t")
        self.assertEqual(self.manager_age_sum, sum(int(row[0]) for row in reader))

    def testModelClassAutoloading(self):
        q = self.SERVICE.model.Manager.select("name", "age")
        expected_sum = 1383

        self.assertEqual(expected_sum, sum(map(lambda x: x.age, q.results(row="object"))))

    def testSearchRes(self):
        res, facs = self.SERVICE.search("david")
        self.assertEqual(2, len(res))

    def testSearchFacets(self):
        res, facs = self.SERVICE.search("david")
        self.assertTrue("Manager" in facs["Category"])
        self.assertEqual(1, facs["Category"]["Manager"])

    def testSearchWithFacet(self):
        res, facs = self.SERVICE.search("david", Category="Department")
        self.assertEqual(1, len(res))
        self.assertEqual("Sales", res[0]["fields"]["name"])

    def test_user_registration(self):
        username = "mayfly-{0}@noreply.intermine.org".format(uuid.uuid4())
        password = "yolo"
        try:
            s = Service(self.SERVICE.root, username, password)
            s.deregister(s.get_deregistration_token())
        except Exception:
            pass

        s = self.SERVICE.register(username, password)

        self.assertEqual(s.root, self.SERVICE.root)
        self.assertEqual(2, len(s.get_all_lists()))

        drt = s.get_deregistration_token()
        s.deregister(drt)

        self.assertRaises(WebserviceError, s.get_all_lists)

    def test_templates(self):
        names = self.SERVICE.templates.keys()
        self.assertTrue(len(names))

        t0 = self.SERVICE.get_template("CEO_Rivals")
        c = t0.count()
        self.assertTrue(c, msg="{0.name} should return some results".format(t0))


if __name__ == "__main__":
    unittest.main()
