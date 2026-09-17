import unittest

from rb2301_ca1.runtime_utils import select_base_ros_domain_id


class RuntimeUtilityTests(unittest.TestCase):
    def test_requested_domain_block_is_preserved(self):
        self.assertEqual(select_base_ros_domain_id(40, 4), 40)

    def test_domain_block_must_fit_ros_range(self):
        with self.assertRaises(ValueError):
            select_base_ros_domain_id(231, 4)

    def test_automatic_domain_avoids_zero(self):
        selected = select_base_ros_domain_id(None, 4)
        self.assertGreaterEqual(selected, 20)
        self.assertLessEqual(selected + 3, 232)


if __name__ == "__main__":
    unittest.main()
