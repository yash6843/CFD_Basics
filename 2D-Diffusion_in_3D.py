import numpy as np
from matplotlib import pyplot as plt , cm
from matplotlib.animation import FuncAnimation, PillowWriter

lx = 2
ly = 2
nx = 101
ny = 101
nu = 0.5
dx = lx/(nx-1)
dy = ly/(ny-1)
sigma=0.15
nt=500
dt=(sigma*(dx**2)/nu)
u = np.ones((nx,ny))
u[40:60,40:60] = 2
utemp = u.copy()
x = np.linspace(0, lx, nx)
y = np.linspace(0, ly, ny)
 
X, Y = np.meshgrid(x, y)


fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
 
ax.plot_surface(X, Y, u, cmap='cool', alpha=0.8)
 
ax.set_title('Diffusion of U', fontsize=14)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_zlabel('u', fontsize=12)

def update(frame):
    global u, dt, dx, nu
    
    un = u.copy()

    u[1:-1,1:-1] = un[1:-1,1:-1] + (dt*nu/dx**2)*(un[2:, 1:-1] + un[1:-1, 2:] - 4*un[1:-1, 1:-1] + un[0:-2, 1:-1] + un[1:-1, 0:-2])
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1

    ax.clear()
    ax.plot_surface(X, Y, utemp, cmap='cool', alpha=0.8)
    ax.plot_surface(X, Y, u, cmap='RdGy', alpha=0.8)
     
    ax.set_title(f'Diffusion of U | Frame {frame}', fontsize=14)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('u', fontsize=12)

    return ax.collections

ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)
writer = PillowWriter(fps=20)
ani.save("./outputs/2D_Diffusion_3D_output.gif", writer=writer)
plt.show()