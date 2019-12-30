import matplotlib.pyplot as plt
import numpy as np
import math

g = np.array([0, 0, -9.81])  # Magnitude of acceleration due to gravity in world frame
m = 1  # Mass of quadrotor in kilograms
ri = np.array([[0.225, 0, 0],
               [0, 0.225, 0],
               [-0.225, 0, 0],
               [0, -0.225, 0]])  # Length of each arm in meters
kf = 0.000002  # Constant of proportionality of force
kw = 0.000002  # Constant of proportionality of moment
wi = (1400, -1400, 1400, -1400)  # Propeller angular velocities in body frame
Fi = np.array(list(map(lambda w: [0, 0, kf * pow(w, 2)], wi)))  # Force of each actuator in body frame
Wi = np.array(list(map(lambda w: [0, 0, np.sign(w) * kw * pow(w, 2)], wi)))  # Moment of each actuator in body frame

# Roll, pitch, and yaw are around the Forward-Right-Down axis. Clockwise is positive
roll = 0      # world frame angular position
pitch = 0     # world frame angular position
yaw = 0      # world frame angular position
attitude = np.array([roll, pitch, yaw])  # Radians


c1 = math.cos(attitude[0])
s1 = math.sin(attitude[0])
t1 = math.tan(attitude[0])
c2 = math.cos(attitude[1])
s2 = math.sin(attitude[1])
t2 = math.tan(attitude[1])
c3 = math.cos(attitude[2])
s3 = math.sin(attitude[2])
t3 = math.tan(attitude[2])

# rotmat = np.array([[c1 * c2, c1 * s2 * s3 - c3 * s1, s1 * s3 + c1 * c3 * s2],
#                    [c2 * s1, c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3],
#                    [-s2, c2 * s3, c2 * c3]]).transpose()     # ZYX

# Use this rotation matrix
rotmat = np.array([[c2*c3, -c2*s3, s2],
                   [c2*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1],
                   [s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2]]).transpose()   # XYZ

world_frame = np.dot(rotmat, attitude)

Ft = sum(Fi) + m * g
Mt = np.sum(np.array(list(map(lambda r, f: np.cross(f, r), ri, Fi))) + Wi, axis=0)

world_force = np.dot(rotmat, Ft)
world_acceleration = world_force / m

print("Body frame to world frame: " + str(attitude) + " --> " + str(world_frame))
print("Total force acting on body frame: " + str(Ft))
print("Total force acting on world frame: " + str(world_force))
print("Total moment acting on body frame: " + str(Mt))
print("Acceleration on world frame: " + str(world_acceleration))

# Data for plotting
t = np.arange(0.0, 5.0, 0.001)
s = world_acceleration[2] * np.power(t, 2)

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='time (s)', ylabel='height (m)',
       title='Height vs time')
ax.grid()

fig.savefig("test.png")
plt.show()
