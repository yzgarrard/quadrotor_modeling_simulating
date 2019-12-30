import unittest
from src.Quadrotor import Quadrotor
import numpy as np
import math


class QuadrotorTests(unittest.TestCase):

    def test_world_gravity(self):
        """
        Sanity test. Should always be [0, 0, 9.81] regardless of quadrotor orientation
        :return:
        """
        testquadrotor = Quadrotor(mass=1,
                                  arm_lengths=[0.225, 0.225, 0.225, 0.225],
                                  k_f=0.000002,
                                  k_w=0.000002,
                                  omega=[0, 0, 0, 0],
                                  roll=math.pi,
                                  pitch=0,
                                  yaw=0)
        self.assertTrue(np.array_equal(testquadrotor.calculate_world_force(),
                                       np.array([0, 0, 9.81])))

    def test_hover_thrust(self):
        """
        Check that a given set of rotor speeds counters gravity
        :return:
        """
        testquadrotor = Quadrotor(mass=1,
                                  arm_lengths=[0.225, 0.225, 0.225, 0.225],
                                  k_f=0.000002,
                                  k_w=0.000002,
                                  omega=[1107.3617295, -1107.3617295, 1107.3617295, -1107.3617295],
                                  roll=0,
                                  pitch=0,
                                  yaw=0)

        self.assertTrue(np.all(np.isclose(testquadrotor.calculate_world_force(), np.array([0, 0, 0]))))


if __name__ == '__main__':
    unittest.main()
