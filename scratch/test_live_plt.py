import matplotlib.pyplot as plt
import numpy as np
import time

# Turn interactive mode on
plt.ion()

# Set up initial plot
fig, ax = plt.subplots()
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
line, = ax.plot(x, y, '-b')
ax.set_ylim(-1.5, 1.5)

# # Continually update plot without blocking
for phase in np.linspace(0, 10*np.pi, 500):
    y = np.sin(x + phase)
    line.set_ydata(y)  # Update data
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)  # Pause briefly to yield thread execution

# Keep plot open at the end
plt.ioff()
plt.show()