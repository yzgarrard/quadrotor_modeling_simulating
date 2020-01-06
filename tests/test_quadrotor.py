import unittest
from src.Quadrotor import Quadrotor
import numpy as np
import math


class TestQuadrotor(unittest.TestCase):

    def test_world_gravity(self):
        """
        Sanity test. Should always be [0, 0, 9.81] regardless of quadrotor orientation
        :return:
        """
        testquadrotor = Quadrotor(mass=1,
                                  attitude=[math.pi, 0, 0])
        self.assertTrue(np.array_equal(testquadrotor.calculate_world_force(),
                                       np.array([0, 0, 9.81])))

    def test_hover_thrust(self):
        """
        Check that a given set of actuator thrusts counters gravity
        :return:
        """
        testquadrotor = Quadrotor(thrust=[0, 0, -12.57642])
        self.assertTrue(np.all(np.isclose(testquadrotor.calculate_world_force(), np.array([0, 0, 0]))))

    def test_nonplanar_force(self):
        """
        Check that when quadrotor is not aligned with an axis or plane, it generates force in the correct direction
        :return:
        """
        testquadrotor = Quadrotor(thrust=[0, 0, -20],
                                  attitude=[0.3, 0.3, 0.3])

        self.assertTrue(np.all(np.isclose(testquadrotor.calculate_world_force(),
                                          np.array([-7.14087943,  3.97779213, -5.67693615]))))


if __name__ == '__main__':
    unittest.main()
