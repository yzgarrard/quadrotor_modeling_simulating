import matplotlib.pyplot as plt
import numpy as np
from src.Quadrotor import Quadrotor

testdrone = Quadrotor(thrust=[0, 0, -20], attitude=[0.3, 0.3, 0.3])
print(testdrone.calculate_world_force())

# Data for plotting
t = np.arange(0.0, 5.0, 0.001)
x, y, z, x_dot, y_dot, z_dot = [], [], [], [], [], []
for i in t:
    testdrone.step_time()
    x.append(testdrone._state_space[0])
    y.append(testdrone._state_space[1])
    z.append(testdrone._state_space[2])
    x_dot.append(testdrone._state_space[3])
    y_dot.append(testdrone._state_space[4])
    z_dot.append(testdrone._state_space[5])


fig, ax = plt.subplots(2, 1)

ax[0].plot(t, x)
ax[0].plot(t, y)
ax[0].plot(t, z)
ax[0].legend(["x", "y", "z"])
ax[0].set(xlabel='time (s)', ylabel='position (m)', title='Position vs time')
ax[0].grid()

ax[1].plot(t, x_dot)
ax[1].plot(t, y_dot)
ax[1].plot(t, z_dot)
ax[1].legend(["x'", "y'", "z'"])
ax[1].set(xlabel='time (s)', ylabel='position (m)', title='Velocity vs time')
ax[1].grid()

# fig.savefig("test.png")
plt.show()
