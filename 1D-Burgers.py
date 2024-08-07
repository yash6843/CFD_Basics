import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# Declaring variables
l = 5  # Total length of the domain
nx = 81  # Number of grid points
dx = l / (nx - 1)  # Distance between each grid point
nt = 200  # Number of time steps
nu = 0.3  # Diffusion coefficient / Kinematic viscosity
disturbance_amplitude = 2  # Amplitude of the initial disturbance

# Calculate time step size using the stability criterion
sigma = 0.2
dt = (sigma * dx**2) / nu  # Time step size

# Initialize the velocity field u
u = np.ones(nx)
u[int(0.5 / dx) : int(1 / dx + 1)] = disturbance_amplitude  # Initial disturbance
initial_u = u.copy()  # Copy of the initial condition for plotting
un = np.ones(nx)  # Array to store the updated velocity field

# Set up the figure and axis for plotting
fig, ax = plt.subplots()
x = np.linspace(0, l, nx)  # Spatial grid
(line1,) = ax.plot(x, initial_u, label="Initial")  # Plot initial condition
(line2,) = ax.plot(x, u, label="Current")  # Plot current velocity field
ax.set_title("1D Convection-Diffusion Equation")  # Plot title
ax.set_xlabel("Position")  # X-axis label
ax.set_ylabel("Velocity")  # Y-axis label
ax.set_xlim(0, l)  # Set X-axis limits
ax.set_ylim(0, 2.5)  # Set Y-axis limits
ax.legend()  # Add legend

# Update function for animation
def update(_):
    global u, un
    un = u.copy()  # Copy the current velocity field
    # Apply the finite difference method to update the velocity field
    u[1 :-1] = (

        un[1 : -1]
        + nu * (dt / (dx**2)) * (un[2:] - 2 * un[1 :-1] + un[0 : -2])  # Diffusion term
        - un[1 : - 1] * (dt / dx) * (un[1 : -1] - un[0 : -2])  # Convection term
    )
    line2.set_ydata(u)  # Update the plot with the new velocity field
    ax.set_title(f'1D Burgers Equation (Frame {_})')  # Update the plot title with the current frame number
    return (line2,)  # Return the updated line for the animation

# Create the animation
ani = FuncAnimation(fig, update, frames=nt)
writer = PillowWriter(fps=20)  # Set frames per second for the animation
ani.save("./outputs/1D_Burgers.gif", writer=writer)  # Save the animation as a GIF
plt.show()  # Show the plot
