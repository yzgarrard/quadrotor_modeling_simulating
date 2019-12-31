import matplotlib.pyplot as plt
import numpy as np
from src.Quadrotor import Quadrotor

test_quadrotor = Quadrotor()

# Data for plotting
t = np.arange(0.0, 5.0, 0.001)
s = 5 * t

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='time (s)', ylabel='height (m)',
       title='Height vs time')
ax.grid()

# fig.savefig("test.png")
plt.show()
