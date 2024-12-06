import json
import unittest

from sage.all import GF, Integer, PolynomialRing

from instance_generation import Instance, gen_zipfian_instance


class TestInstanceCalcIfNice(unittest.TestCase):

    def test_empty(self):
        self.assertTrue(Instance.calc_if_nice(4, []))

    def test_nice(self):
        self.assertTrue(Instance.calc_if_nice(3, [1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4]))

    def test_close_dealers(self):
        self.assertFalse(Instance.calc_if_nice(2, [1, 2, 1, 2, 1, 2, 2]))

    def test_same_dealers(self):
        self.assertFalse(Instance.calc_if_nice(2, [5, 5, 5, 6, 6, 6, 5, 6]))

    def test_same_under_threshold(self):
        self.assertTrue(Instance.calc_if_nice(4, [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3]))

    def test_close_under_threshold(self):
        self.assertTrue(
            Instance.calc_if_nice(
                4,
                [
                    1,
                    1,
                    2,
                    2,
                    2,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    6,
                    6,
                ],
            )
        )


class TestInstanceSerialization(unittest.TestCase):
    def test_serialization(self):
        f = GF(13)
        pR = PolynomialRing(f, "x")

        inst = gen_zipfian_instance(f, pR, 2, 2, 3, 10, 1.5, 10, disable=True)
        json.dumps(inst.to_serializable())

    def test_pipeline(self):
        f = GF(13)
        pR = PolynomialRing(f, "x")

        inst = gen_zipfian_instance(f, pR, 2, 2, 3, 10, 1.5, 10, disable=True)

        data = inst.to_serializable()

        new_inst = Instance.from_json(data)

        self.assertEqual(inst.agreement, new_inst.agreement)
        self.assertEqual(inst.codeword, new_inst.codeword)


if __name__ == "__main__":
    unittest.main()
