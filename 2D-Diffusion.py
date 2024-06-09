# Import necessary modules
import numpy as np 
from matplotlib import pyplot as plt, cm
from IPython.display import HTML
from matplotlib.animation import FuncAnimation, PillowWriter

# Declaring Variables
l = 2  # Total length of the domain
nx = 41  # Number of grid points
dx = l / (nx - 1)  # Distance between each grid point
nu = 0.3 # Diffusion coefficient / Kinematic viscosity

# Calculate time step size using the stability criterion
sigma = 0.2
dt = sigma*(dx**2)/ nu
tolerance = 1e-5 # Checking Convergence

# Generating Mesh / Array
x = np.linspace(0,2,nx)
y = np.linspace(0,2,nx)
# Initialization / Intial condition
u = np.ones((nx,nx))
un = np.ones((nx,nx))

u[int(0.5/dx):int(1/dx + 1),int(0.5/dx):int(1/dx + 1)] = 2

# FUNCTION TO UPDATE THE FIELD
def diffuse(u,nu,dt,dx):
    un = u.copy()
    # u[1:nx-1, 1:nx-1] = un[1:nx-1,1:nx-1] + (dt*nu)/(dx**2) * un[2:nx,1:nx-1] + un[1:nx-1,2:nx] + un[1:nx-1,0:nx-2] + un[0:nx-2,1:nx-1] - 4*un[1:nx-1,1:nx-1]
    # For better computing we use reverse indexing and avoid using nx for indexing
    
    u[1:-1,1:-1] = un[1:-1,1:-1] + (dt*nu)/(dx**2) * (un[2:,1:-1] + un[1:-1,2:] + un[1:-1,0:-2] + un[0:-2,1:-1] - 4*un[1:-1,1:-1])

    u[0,:] = 1
    u[-1,:] = 1
    u[:,0] = 1
    u[:,-1] = 1
    return u
    
# Post - processing 
fig , ax = plt.subplots(figsize=(8,6))
# Grid 
X , Y = np.meshgrid(x, y)
contour  = ax.contourf(X, Y, u, alpha=0.75, cmap=cm.viridis)
cbar = fig.colorbar(contour)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title(f'2D Contour Plot of the Diffused Field')

def update(frame):
    global u, un, tolerance, ani
    un = u.copy()
    u = diffuse(u,nu,dt,dx)
    ax.clear()
    contour = ax.contourf(X,Y,u, alpha=0.75,cmap=cm.viridis)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(f'2D Contour Plot of the Diffused Field (Frame {frame})')
    diff = np.linalg.norm(u-un)
    if diff < tolerance:
        ani.event_source.stop()
    return ax.collections 


ani = FuncAnimation(fig, update, frames=200,interval=100,blit=True)
writer = PillowWriter(fps=20)
ani.save("./outputs/2D_Diffusion.gif", writer=writer)
plt.show()