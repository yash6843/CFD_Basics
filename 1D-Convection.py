import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# Declaring variables
l = 5            # Length of the domain
nx = 81          # Number of spatial grid points
dx = l / (nx - 1)  # Distance between each spatial grid point
nt = 200         # Number of time steps
c = 1            # Convection speed
disturbance_amplitude = 2  # Amplitude of the initial disturbance

sigma = 0.2
dt = sigma * dx / c  # Time step size

# Initial condition: u is 1 everywhere except between 0.5 and 1 where u is disturbance_amplitude
u = np.ones(nx)
u[int(0.5 / dx) : int(1 / dx + 1)] = disturbance_amplitude
initial_u = u.copy()
un = np.ones(nx)

# Setting up the figure and axis
fig, ax = plt.subplots()
x = np.linspace(0, l, nx)
(line1,) = ax.plot(x, initial_u, label="Initial")
(line2,) = ax.plot(x, u, label="Current")
ax.set_title("1D Linear Convection Equation")
ax.set_xlabel("Position")
ax.set_ylabel("Velocity")
ax.set_xlim(0, l)
ax.set_ylim(0, 2.5)
ax.legend()

# Update function for animation
def update(frame):
    global u, un
    un = u.copy()
    u[1:nx] = un[1:nx] - c * (dt / dx) * (un[1:nx] - un[0:nx-1])
    line2.set_ydata(u)
    ax.set_title(f'1D Linear Convection Equation (Frame {frame})')
    return (line2,)

# Creating the animation
ani = FuncAnimation(fig, update, frames=nt)
writer = PillowWriter(fps=20)
ani.save("./GIFs/1D_Convection.gif", writer=writer)
plt.show()