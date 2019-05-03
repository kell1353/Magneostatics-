import numpy as np
from mayavi import mlab

##"""Current"""
##I = 1 # Current in the positive Z direction
##"""Distance from current"""
###s = np.sqrt((x - 0)**2 + (y - 0)**2  + (z - z)**2)
##x = np.linspace(-np.pi, 0, 4)
##y = np.linspace(-3*np.pi/2, np.pi/2, 4)
##xx, yy = np.meshgrid(x,y)
##
##min_v = -2
##max_v = (-min_v)+1
##def test_quiver3d():
##    x, y, z = np.mgrid[min_v:max_v, min_v:max_v, min_v:max_v]
##    r = np.sqrt((x - 0)**2 + (y - 0)**2  + (z - z)**2)
##    
##    mag = I/(2*np.pi*(r + .0000000001))
##    
####    u = -y * mag * (np.sin(r)/(r + .0001))
####    v = x * mag * (np.sin(r)/(r + .0001))
##    u = -y * mag *10
##    v = x * mag *10
##    print(u)
##    w = np.zeros_like(z)
##    obj = mlab.quiver3d(x, y, z, u, v, w)
##    return obj
##
##
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
x = np.linspace(-10, 10, n)
y = np.linspace(-10, 10, n)
x, y = np.meshgrid(x, y)
z = np.zeros((n, n))

draw_sphere(x0, y0, 0, .10)

def B(x, y, z):
    I = 2
    mu = 2
    r =  np.sqrt((x)**2 + (y)**2)
    mag = ((mu/(2*np.pi))*(I/r))
    theta = np.arctan2(y, x)
    bx = mag * (-np.sin(theta))
    by = mag * (np.cos(theta))
    bz = mag * z
    return bx,by,bz

bx, by, bz = B(x, y, z)
mlab.quiver3d(x, y, z, bx, by, bz, line_width = 1, scale_factor=1)


t = np.linspace(-4, 4, 100)
mlab.plot3d(0*t, 0*t, t)
##
##test_quiver3d()
#field_lines = mayavi.mlab.pipeline.streamline(magnitude,seedtype="line",integration_direction="both")
mlab.show()
