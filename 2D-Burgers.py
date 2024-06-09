# Importing necessary modules
import numpy as np 
from matplotlib import pyplot as plt, cm
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


# Generating mesh
x = np.linspace(0,l,nx)
y = np.linspace(0,l,nx)

u = np.ones((nx,nx))
un = np.ones((nx,nx))

v = np.ones((nx,nx))
vn = np.ones((nx,nx))

u[int(0.5/dx):int(1/dx + 1),int(0.5/dx):int(1/dx + 1)] = 2
v[int(0.5/dx):int(1/dx + 1),int(0.5/dx):int(1/dx + 1)] = 2


def condiffuse(u,v,nu,dt,dx):
    un = u.copy()
    vn = v.copy()
    # For better computing we use reverse indexing and avoid using nx for indexing
    
    u[1:-1,1:-1] = un[1:-1,1:-1] + (dt/dx**2) * (nu*(un[1:-1,2:] + un[1:-1,0:-2] + un[2:,1:-1] + un[0:-2,1:-1] - 4*un[1:-1,1:-1]) - dx*(un[1:-1,1:-1]*(un[2:,1:-1] - un[1:-1,1:-1]) + vn[1:-1,1:-1]*(un[1:-1,2:] - un[1:-1,1:-1])))
    v[1:-1,1:-1] = vn[1:-1,1:-1] + (dt/dx**2) * (nu*(vn[2:,1:-1] + vn[0:-2,1:-1] + vn[1:-1,2:] + vn[1:-1,0:-2] - 4*vn[1:-1,1:-1]) - dx*(un[1:-1,1:-1]*(vn[2:,1:-1] - vn[1:-1,1:-1]) + vn[1:-1,1:-1]*(vn[1:-1,2:] - vn[1:-1,1:-1])))
    
    # Boundary Conditions
    u[0,:] = 1
    u[-1,:] = 1
    u[:,0] = 1
    u[:,-1] = 1
    
    v[0,:] = 1
    v[-1,:] = 1
    v[:,0] = 1
    v[:,-1] = 1
    
    return u,v
    
# Post - processing 
vel = np.sqrt(u**2 + v**2)
fig , ax = plt.subplots(figsize=(8,6))
# Grid 
X , Y = np.meshgrid(x, y)
contour  = ax.contourf(X, Y, vel, alpha=0.75, cmap=cm.viridis)
cbar = fig.colorbar(contour)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title(f'2D Contour Plot of the Convection and Diffusion Field')

def update(frame):
    global u, un, v, vn, tolerance, ani
    
    un = u.copy()
    vn = v.copy()
    u, v = condiffuse(u,v,nu,dt,dx)
    ax.clear()
    vel = np.sqrt(u**2 + v**2)
    contour = ax.contourf(X,Y,vel, alpha=0.75,cmap=cm.viridis)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(f'2D Contour Plot of the Convected and Diffused Field (Frame {frame})')
    diff = np.linalg.norm(u-un) + np.linalg.norm(v-vn)
    if diff < tolerance:
        ani.event_source.stop()
    return ax.collections 


ani = FuncAnimation(fig, update, frames=200,interval=100,blit=True)
writer = PillowWriter(fps=20)
ani.save("./GIFs/2D_Burgers.gif", writer=writer)
plt.show()