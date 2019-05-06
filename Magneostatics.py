import numpy as np
from mayavi import mlab

mlab.figure(bgcolor = (1,1,1))

points_range = 50
phi = np.linspace(0, 2*np.pi, points_range)
theta = np.linspace(0, 2*np.pi, points_range)
phi, theta = np.meshgrid(phi, theta)

def draw_sphere(x0, y0, z0, r):
    x = (r* np.sin(phi) * np.cos(theta)) + x0
    y = (r*np.sin(phi) * np.sin(theta)) + y0
    z = (r*np.cos(phi)) + z0
    sphere = mlab.mesh(x, y, z)

def draw_vector(x0, y0, z0, u, v, w):
    mlab.quiver3d(x0, y0, z0, u, v, w, line_width = 1)

##a = np.array([[0, 0, 3, 1, 0, 0], [-1, 0, 0, 0, 0, 1]])
##x, y, z, u, v, w = zip(*a)
##mlab.quiver3d(x, y, z, u, v, w, line_width = 1, scale_factor=1)
##
##draw_sphere(1, 0, 3, .10)
##
##f = np.array([[0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 1]])
##x, y, z, u, v, w = zip(*f)
##mlab.quiver3d(x, y, z, u, v, w, line_width = 1, scale_factor=1)
##
##draw_sphere(1, 0, 1, .10)

n = 20
x0, y0 = 1, 1
zmin, zmax = -4, 4
lim = 10
x = np.linspace(-lim + x0, lim + y0, n)
y = np.linspace(-lim + x0, lim + y0, n)
z = np.linspace(zmin, zmax, n)
x, y, z = np.meshgrid(x, y, z)

#z = np.zeros((n, n))

draw_sphere(x0, y0, 0, .10)

def B(x, y, z):
    #Current and magnetic constant
    I, m_u = 1, 1
    #m_u = 1.25663706 * (10**(-6))                                   # m*kg/((s^2)*(A^2))
    #Distance from current 
    r =  np.sqrt((x - x0)**2 + (y - y0)**2)
    
    mag = I*m_u/((2*np.pi)*r)
    theta = np.arctan2(y - y0, x - x0)
    
    #Calculating the vector components
    bx = mag * (-np.sin(theta))
    by = mag * (np.cos(theta))
    bz =  z * 0
    return bx,by,bz


bx, by, bz = B(x, y, z)
mlab.quiver3d(x, y, z, bx, by, bz, line_width = 1, scale_factor= 4)

t = np.linspace(zmin, zmax, n)
mlab.plot3d(0*t + x0, 0*t + y0, t)

axes = np.linspace(-20, 20, 100)
x_axis = mlab.plot3d(0*axes, 0*axes, axes, color=(0,0,0), tube_radius = .02)
y_axis = mlab.plot3d(axes, 0*axes, 0*axes, color=(0,0,0), tube_radius = .02)
z_axis = mlab.plot3d(0*axes, axes, 0*axes, color=(0,0,0), tube_radius = .02)
##test_quiver3d()
#field_lines = mayavi.mlab.pipeline.streamline(magnitude,seedtype="line",integration_direction="both")
mlab.show()
