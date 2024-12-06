import unittest

from instance_generation import Instance


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


if __name__ == "__main__":
    unittest.main()
