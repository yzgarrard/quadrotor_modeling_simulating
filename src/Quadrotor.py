import numpy as np
import math


class Quadrotor:
    """
    A quadrotor model to use.
    Uses FRD for body frame, NED for world frame.
    """
    _mass = None  # kilograms
    _inertia = None  # kg*m^2
    _g = np.array([0, 0, 9.81])  # I'm assuming this is on earth. Downward positive
    _arm_lengths = None  # arm lengths in meters. each row is [x,y,z] length of arm
    _thrust = None   # total thrust on the body. Newtons. [x, y, z]
    _attitude_rate = np.array([None, None, None])  # roll, pitch, and yaw rates. Clockwise positive. rad/s
    _attitude = np.array([None, None, None])  # roll, pitch, yaw. Euler angles. Clockwise positive. Radians
    _cr, _sr, _tr, = None, None, None  # cos, sin, and tan of roll with input in radians
    _cp, _sp, _tp, = None, None, None  # cos, sin, and tan of pitch with input in radians
    _cy, _sy, _ty = None, None, None  # cos, sin, and tan of yaw with input in radians
    _rotation_matrix = None  # ZYX rotation matrix to go from body frame to world frame
    _world_force = None  # force vector on the quadrotor in world frame
    _world_moment = None  # moment on the quadrotor in world frame
    _state_space = [None, None, None,  # x y z         Position along x, y, and z axis world frame
                   None, None, None,  # x' y' z'      Velocity along x, y, and z axis world frame
                   None, None, None,  # phi, theta, psi   Roll, pitch, yaw angles
                   None, None, None]  # phi', theta', psi'    Roll, pitch, yaw rates
    _timestep = None

    def __init__(self,
                 mass: float = None,
                 arm_lengths: "list of floats" = None,
                 inertia: "list of list of floats" = None,
                 attitude: "list of floats" = None,
                 attitude_rate: "list of floats" = None,
                 thrust: "list of floats" = None,
                 state_space: "list of floats" = None,
                 timestep: float = None):
        """
        :param mass: mass of quadrotor in kg
        :param arm_lengths: list floats of length 4. meters e.g.[[-0.13, 0.2252, 0],
                                                                 [0.0923, 0.2537, 0],
                                                                 [0.0923, -0.2537, 0],
                                                                 [-0.13, -0.2252, 0]]
        :param inertia: I_xx, I_yy, I_zz in kg*m^2
        :param thrust: initial thrust force on the drone
        :param attitude: roll, pitch, and yaw in radians. Clockwise positive
        :param attitude_rate: roll, pitch, and yaw rates in rad/s.
        :param state_space: initial state space of quadrotor. [x, y, z, x', y', z', r, p, y, r', p', y']:
        :param timestep: period between model updates. Seconds
        """
        if mass is None:
            mass = 1.282
        if arm_lengths is None:
            arm_lengths = [[-0.13, 0.2252, 0],
                           [0.0923, 0.2537, 0],
                           [0.0923, -0.2537, 0],
                           [-0.13, -0.2252, 0]]
        if inertia is None:
            inertia = [0.0219, 0.0109, 0.0306]
        if attitude is None:
            attitude = [0, 0, 0]
        if attitude_rate is None:
            attitude_rate = [0, 0, 0]
        if thrust is None:
            thrust = [0, 0, 0]
        if state_space is None:
            state_space = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if timestep is None:
            timestep = 0.001

        self._mass = mass
        self._arm_lengths = np.array(arm_lengths)
        self._inertia = np.array([[inertia[0], 0, 0],
                                  [0, inertia[1], 0],
                                  [0, 0, inertia[2]]])
        self._attitude = np.array(attitude)
        self._attitude_rate = attitude_rate
        self._thrust = thrust
        self._state_space = state_space
        self._timestep = timestep

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
        # remember that upwards is negative
        # Rotates around yaw, pitch, then roll. Clockwise positive
        self._world_force = np.dot(self._rotation_matrix, self._thrust) + self._mass * self._g
        return self._world_force

    def step_time(self):
        x, y, z, x_dot, y_dot, z_dot, roll, pitch, yaw, roll_dot, pitch_dot, yaw_dot = self._state_space

        x = x + x_dot * self._timestep
        x_dot = x_dot + self.calculate_world_force()[0] * self._timestep
        y = y + y_dot * self._timestep
        y_dot = y_dot + self.calculate_world_force()[1] * self._timestep
        z = z + z_dot * self._timestep
        z_dot = z_dot + self.calculate_world_force()[2] * self._timestep

        self._state_space = np.array([x, y, z, x_dot, y_dot, z_dot, roll, pitch, yaw, roll_dot, pitch_dot, yaw_dot])


if __name__ == "__main__":
    testdrone = Quadrotor(thrust=[0, 0, -20], attitude_rate=[0.01, 0.01, 0.01])
    print(testdrone._state_space)
