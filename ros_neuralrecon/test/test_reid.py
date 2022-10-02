#!/usr/bin/env python3

import unittest

from geometry_msgs.msg import Point
from vision_msgs.msg import BoundingBox2D
from std_msgs.msg import Float64MultiArray


class TestRosNeuralRecon(unittest.TestCase):
    """
    TODO: build test 

    Run instructions:
        python3 test_reid.py

    @author(Smells Like ML)
    """

    def __init__(self, *args, **kwargs):
        super(TestRosNeuralRecon, self).__init__(*args, **kwargs)

    def test_from_ros(self):
        # TODO: Design actual test
        self.assertEqual(1,1)


if __name__ == "__main__":
    unittest.main()
