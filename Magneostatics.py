import numpy as np
from mayavi import mlab

#mlab.figure(bgcolor = (1,1,1))

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
    mlab.quiver3d(x0, y0, z0, u, v, w, line_width = 1, scale_factor= 10)

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

n = 10
##x0, y0 = 2, 2
x_c, y_c = 1, 1
zmin, zmax = -8, 8
lim = 10
x = np.linspace(-lim + x_c, lim + y_c, n)
y = np.linspace(-lim + x_c, lim + y_c, n)
z = np.linspace(zmin, zmax, n)
x, y, z = np.meshgrid(x, y, z)

#z = np.zeros((n, n))

#draw_sphere(x0, y0, 0, .10)

def B(I, x, y, z, x0, y0):                                                         # I is current, x0 and y0 are the current line position
    #Current and magnetic constant
    m_u = 1
    #m_u = 1.25663706 * (10**(-6))                                   # m*kg/((s^2)*(A^2))
    #Distance from current
    r = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z)**2)
    
    mag = I*m_u/((2*np.pi)*r)   
    theta = np.arctan2(y - y0, x - x0)
    
    #Calculating the vector components
    bx = mag * (-np.sin(theta))
    by = mag * (np.cos(theta))
    bz =  z * 0
    return bx,by,bz


bx0, by0, bz0 = B(1, x, y, z, 1, 1)
bx1, by1, bz1 = B(1, x, y, z, -1, -1)

bx = bx0 + bx1
by = by0 + by1
bz = bz0 + bz1
#draw_vector(x, y, z, bx0, by0, bz0)
draw_vector(x, y, z, bx, by, bz)

t = np.linspace(zmin, zmax, n)
mlab.plot3d(0*t + x_c, 0*t + y_c, t)
mlab.plot3d(0*t - 1, 0*t  - 1, t)

axes = np.linspace(-20, 20, 100)
x_axis = mlab.plot3d(0*axes, 0*axes, axes, color=(0,0,0), tube_radius = .02)
y_axis = mlab.plot3d(axes, 0*axes, 0*axes, color=(0,0,0), tube_radius = .02)
z_axis = mlab.plot3d(0*axes, axes, 0*axes, color=(0,0,0), tube_radius = .02)
##test_quiver3d()
#field_lines = mayavi.mlab.pipeline.streamline(magnitude,seedtype="line",integration_direction="both")
mlab.show()
