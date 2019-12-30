import numpy as np
import math


class Quadrotor:
    """
    A quadrotor model to use.
    Uses FRD for body frame, NED for world frame.
    Uses X configuration with x-axis pointing forward, y-axis pointing right
    """
    _mass = None  # kilograms
    _g = np.array([0, 0, 9.81])  # I'm assuming this is on earth. Downward positive

    # arm lengths in meters. each row is [x,y,z] length of arm
    _arm_lengths = None

    _k_f = None  # constant of proportionality of force
    _k_w = None  # constant of proportionality of moment

    # rotor speeds, rad/s. Clockwise around the body-down axis is positive
    _omega = None

    _Fi = None   # Force vector of each actuator in body frame
    _Ft = None   # Total force vector of actuators in body frame

    _Mi = None   # Moment vector of each actuator in body frame
    _Mt = None   # Total moment vector of actuators in body frame

    # yaw, pitch, roll. Euler angles
    # clockwise around their respective axes is positive. Units are radians
    _attitude = np.array([None, None, None])

    # cos, sin, and tan of roll, pitch, and yaw. Radians
    _cr, _sr, _tr, _cp, _sp, _tp, _cy, _sy, _ty = \
        None, None, None, None, None, None, None, None, None

    _rotation_matrix = None  # ZYX rotation matrix to go from body frame to world frame

    _world_force = None  # force vector on the quadrotor in world frame

    def __init__(self,
                 mass: float,
                 arm_lengths: "list of floats",
                 k_f: float,
                 k_w: float,
                 omega: "list of floats",
                 roll: float,
                 pitch: float,
                 yaw: float):
        """
        :param mass: mass of quadrotor in kg
        :param arm_lengths: list floats of length 4. meters e.g. [0.225, 0.225, 0.225, 0.225]
        :param k_f: constant of proportionality of force. Motor constant?
        :param k_w: constant of proportionality of moment. Torque constant?
        :param omega: initial motor speeds, rad/s. e.g. [1200, -1200, 1200, -1200]
        :param roll: initial roll in radians
        :param pitch: initial pitch in radians
        :param yaw: initial yaw in radians
        """
        self._mass = mass
        self._arm_lengths = np.array([[arm_lengths[0], 0, 0],
                                      [0, arm_lengths[1], 0],
                                      [arm_lengths[2], 0, 0],
                                      [0, arm_lengths[3], 0]])
        self._k_f = k_f
        self._k_w = k_w
        self._omega = np.array(omega)
        self._attitude = np.array([yaw, pitch, roll])

        self._update_rotation_matrix()

    def _update_rotation_matrix(self):
        """
        Updates the rotation matrix based on current Euler angles. Uses ZYX (yaw pitch roll)
        """
        self._cy = math.cos(self._attitude[0])
        self._sy = math.sin(self._attitude[0])
        self._ty = math.tan(self._attitude[0])
        self._cp = math.cos(self._attitude[1])
        self._sp = math.sin(self._attitude[1])
        self._tp = math.tan(self._attitude[1])
        self._cr = math.cos(self._attitude[2])
        self._sr = math.sin(self._attitude[2])
        self._tr = math.tan(self._attitude[2])
        self._rotation_matrix = np.array([[self._cp * self._cy, self._sr * self._sp * self._cy - self._cr * self._sy,
                                           self._cr * self._sp * self._cy + self._sr * self._sy],
                                          [self._cp * self._sy, self._sr * self._sp * self._sy + self._cr * self._cy,
                                           self._cr * self._sp * self._sy - self._sr * self._cy],
                                          [-self._sp, self._sr * self._cp, self._cr * self._cp]])

    def calculate_world_force(self) -> np.ndarray:
        self._Fi = np.array(list(map(lambda w: [0, 0, self._k_f * pow(w, 2)], self._omega)))
        self._Ft = sum(self._Fi)
        # remember that upwards is negative
        # Rotates around yaw, pitch, then roll. Clockwise positive
        self._world_force = -np.dot(self._rotation_matrix, self._Ft) + self._mass * self._g
        return self._world_force


if __name__ == "__main__":
    testdrone = Quadrotor(mass=1,
                          arm_lengths=[0.225, 0.225, 0.225, 0.225],
                          k_f=0.000002,
                          k_w=0.000002,
                          omega=[1107.3617295, -1107.3617295, 1107.3617295, -1107.3617295],
                          roll=math.pi/2,
                          pitch=0,
                          yaw=0)
    print(testdrone.calculate_world_force())
