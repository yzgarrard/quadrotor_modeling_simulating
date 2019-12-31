import numpy as np
import math


class Quadrotor:
    """
    A quadrotor model to use.
    Uses FRD for body frame, NED for world frame.
    Uses X configuration with x-axis pointing forward, y-axis pointing right
    """
    _mass = None  # kilograms
    _inertia = None  # kg*m^2
    _g = np.array([0, 0, 9.81])  # I'm assuming this is on earth. Downward positive
    _arm_lengths = None     # arm lengths in meters. each row is [x,y,z] length of arm
    _Fi = None   # Force vector of each actuator in body frame
    _Ft = None   # Total force vector of actuators in body frame
    _Mi = None   # Moment vector of each actuator in body frame
    _Mt = None   # Total moment vector of actuators in body frame
    _attitude = np.array([None, None, None])  # roll, pitch, yaw. Euler angles. Clockwise positive. Radians
    _cr, _sr, _tr, = None, None, None   # cos, sin, and tan of roll with input in radians
    _cp, _sp, _tp, = None, None, None   # cos, sin, and tan of pitch with input in radians
    _cy, _sy, _ty = None, None, None    # cos, sin, and tan of yaw with input in radians
    _rotation_matrix = None  # ZYX rotation matrix to go from body frame to world frame
    _world_force = None  # force vector on the quadrotor in world frame
    _world_moment = None    # moment on the quadrotor in world frame
    state_space = [None, None, None,    # x y z         Position along x, y, and z axis
                   None, None, None,    # x' y' z'      Velocity along x, y, and z axis
                   None, None, None,    # phi, theta, psi   Roll, pitch, yaw angles
                   None, None, None]    # phi', theta', psi'    Roll, pitch, yaw rates

    def __init__(self,
                 mass: float = None,
                 arm_lengths: "list of floats" = None,
                 inertia: "list of floats" = None,
                 motor_thrusts: "list of floats" = None,
                 attitude: "list of floats" = None,
                 state_space: "list of floats" = None):
        """
        :param mass: mass of quadrotor in kg
        :param arm_lengths: list floats of length 4. meters e.g. [0.225, 0.225, 0.225, 0.225]
        :param inertia: I_xx, I_yy, I_zz in kg*m^2
        :param motor_thrusts: initial motor thrusts, Newtons. e.g. [-5, -5, -5, -5]
        :param attitude: roll, pitch, and yaw in radians. Clockwise positive
        :param state_space: initial state space of quadrotor. [x, y, z, x', y', z', r, p, y, r', p', y']
        """
        if mass is None:
            mass = 1.282
        if arm_lengths is None:
            arm_lengths = [0.275, 0.275, 0.275, 0.275]
        if inertia is None:
            inertia = [0.0219, 0.0109, 0.0306]
        if motor_thrusts is None:
            motor_thrusts = [0, 0, 0, 0]
        if attitude is None:
            attitude = [0, 0, 0]
        if state_space is None:
            state_space = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        self._mass = mass
        self._arm_lengths = np.array([[arm_lengths[0], 0, 0],
                                      [0, arm_lengths[1], 0],
                                      [arm_lengths[2], 0, 0],
                                      [0, arm_lengths[3], 0]])
        self._inertia = np.array(inertia)
        self._Fi = np.array([[0, 0, motor_thrusts[0]],
                             [0, 0, motor_thrusts[1]],
                             [0, 0, motor_thrusts[2]],
                             [0, 0, motor_thrusts[3]]])
        self._attitude = np.array(attitude)
        self.state_space = state_space

        self._update_rotation_matrix()

    def _update_rotation_matrix(self):
        """
        Updates the rotation matrix based on current Euler angles. Uses ZYX (yaw pitch roll) Tait-Bryan matrix
        """
        self._cr = math.cos(self._attitude[0])
        self._sr = math.sin(self._attitude[0])
        self._tr = math.tan(self._attitude[0])  # I update the tan of each angle just in case I need it later
        self._cp = math.cos(self._attitude[1])
        self._sp = math.sin(self._attitude[1])
        self._tp = math.tan(self._attitude[1])
        self._cy = math.cos(self._attitude[2])
        self._sy = math.sin(self._attitude[2])
        self._ty = math.tan(self._attitude[2])
        self._rotation_matrix = np.array([[self._cp * self._cy, self._sr * self._sp * self._cy - self._cr * self._sy,
                                           self._cr * self._sp * self._cy + self._sr * self._sy],
                                          [self._cp * self._sy, self._sr * self._sp * self._sy + self._cr * self._cy,
                                           self._cr * self._sp * self._sy - self._sr * self._cy],
                                          [-self._sp, self._sr * self._cp, self._cr * self._cp]])

    def calculate_world_force(self) -> np.ndarray:
        self._Ft = sum(self._Fi)
        # remember that upwards is negative
        # Rotates around yaw, pitch, then roll. Clockwise positive
        self._world_force = np.dot(self._rotation_matrix, self._Ft) + self._mass * self._g
        return self._world_force


if __name__ == "__main__":
    testdrone = Quadrotor(attitude=[0.3, 0.3, 0.3],
                          motor_thrusts=[-5, -5, -5, -5])
    print(testdrone.calculate_world_force())
