import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.animation import FuncAnimation, PillowWriter

# Grid parameters
nx = 41  # Number of grid points in the x-direction
ny = 41  # Number of grid points in the y-direction
nt = 500  # Number of time steps
nit = 50  # Number of iterations for the pressure Poisson equation
c = 1  # Arbitrary constant (not used in this code)
dx = 2 / (nx - 1)  # Grid spacing in the x-direction
dy = 2 / (ny - 1)  # Grid spacing in the y-direction
x = np.linspace(0, 2, nx)  # x coordinates
y = np.linspace(0, 2, ny)  # y coordinates
X, Y = np.meshgrid(x, y)  # Meshgrid for plotting

# Physical parameters
rho = 1  # Density
nu = 0.1  # Kinematic viscosity
dt = 0.001  # Time step size

u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

def build_up_b(b, rho, dt, u, v, dx, dy):
    b[1:-1, 1:-1] = (rho * (1 / dt *
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) /
                             (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                            2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                                 (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                            ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))
    return b

def pressure_poisson(p, dx, dy, b):
    pn = np.empty_like(p)
    for _ in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) -
                         dx**2 * dy**2 / (2 * (dx**2 + dy**2)) *
                         b[1:-1, 1:-1])

        p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 2
        p[0, :] = p[1, :]    # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]    # dp/dx = 0 at x = 0
        p[-1, :] = 0         # p = 0 at y = 2
        
    return p

def cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros((ny, nx))
    
    # Create a figure for plotting
    fig, ax = plt.subplots(figsize=(11, 7), dpi=100)
    
    def update(frame):
        nonlocal u, v, p, un, vn, b
        
        un = u.copy()
        vn = v.copy()
        
        b = build_up_b(b, rho, dt, u, v, dx, dy)
        p = pressure_poisson(p, dx, dy, b)
        
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx *
                         (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                         (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx**2 *
                               (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                               dt / dy**2 *
                               (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx *
                         (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                         (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                         dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                         nu * (dt / dx**2 *
                               (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                               dt / dy**2 *
                               (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        u[0, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        u[-1, :] = 1  # set velocity on cavity lid equal to 1
        v[0, :] = 0
        v[-1, :] = 0
        v[:, 0] = 0
        v[:, -1] = 0

        # Clear the previous plot
        ax.clear()
        # Plot the pressure field as a contour
        ax.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
        ax.contour(X, Y, p, cmap=cm.viridis)
        # Plot the velocity field
        ax.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Cavity Flow at Time Step {frame}')
        
    ani = FuncAnimation(fig, update, frames=nt, repeat=False)
    writer = PillowWriter(fps=20)
    ani.save("./GIFs/Cavity_flow.gif", writer=writer)
    plt.show()

# Initialize the fields
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

# Run the cavity flow simulation and create a GIF
cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu)
