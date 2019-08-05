import numpy as np
from mayavi import mlab
from sympy.solvers import solve
from sympy import Symbol
import math

#mlab.figure(bgcolor = (1,1,1))

""" Common Variables """
sqrt = np.sqrt
sin = np.sin
cos = np.cos
tan = np.tan
pi = np.pi

""" Draw the x, y, z axes """
axes = np.linspace(-20, 20, 100)
x_axis = mlab.plot3d(0*axes, 0*axes, axes, color=(0,0,0), tube_radius = .02)
y_axis = mlab.plot3d(axes, 0*axes, 0*axes, color=(0,0,0), tube_radius = .02)
z_axis = mlab.plot3d(0*axes, axes, 0*axes, color=(0,0,0), tube_radius = .02)

""" Draw 3D points at a given point """
points_range = 50
phi = np.linspace(0, 2*np.pi, points_range)
theta = np.linspace(0, 2*np.pi, points_range)
phi, theta = np.meshgrid(phi, theta)

def draw_sphere(x0, y0, z0, r):
    x = (r*sin(phi) * cos(theta)) + x0
    y = (r*sin(phi) * sin(theta)) + y0
    z = (r*cos(phi)) + z0
    sphere = mlab.mesh(x, y, z)

""" Draw a vector given the location and components """
def draw_vector(x0, y0, z0, u, v, w):
    mlab.quiver3d(x0, y0, z0, u, v, w, line_width = 1, scale_factor= 100)

""" Line of line segments """
#(x -x0/(x1-x0)) = (y - y0/(y1 - y0)) = (z - z0/(z1 - z0))
def L(x0, y0, z0, x1, y1, z1, n):                                        # n - number of  dl line segments
    t = np.linspace(0, 1, n)
    x = x0 + (x1 - x0)*t 
    y = y0 + (y1 - y0)*t
    z = z0 + (z1 - z0)*t

    wire = mlab.plot3d(x, y, z, color=(1,0,0), tube_radius = .02)

""" Calculate the distance of a vector to the closest point on a line """
#http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
def vectorMag(vx, vy, vz):
    return sqrt(vx**2 + vy**2 + vz**2)
def calcDist(x0, y0, z0, x1, y1, z1, xp, yp, zp): #(inital point on line, end point on line, comparison point)
    global dist, ux, uy, uz, bx0, by0, bz0, b0_mag
    ux, uy, uz = x1 - x0, y1 - y0, z1 - z0
    a = [ux, uy, uz]
    vx, vy, vz = xp - x0, yp - y0, zp - z0
    wx, wy, wz = xp - x1, yp - y1, zp - z1
##    print(ux, uy, uz)
##    print(vx, vy, vz)
##    print("\nPOINTS")
##    print(xp, yp, zp)

    bx0 = vy*wz - vz*wy
    by0 = vz*wx - vx*wz
    bz0 = vx*wy - vy*wx
##    print("\nCROSS PRODUCT")
##    print("\nBX0")
##    print(bx0)
##    print("\nBY0")
##    print(by0)
##    print("\nBZ0")
##    print(bz0)
    """Magnitude of Initial Vector"""
    uMag = vectorMag(ux, uy, uz) 
    """Magnitude of Resulting Vector"""
    b0_mag = vectorMag(bx0, by0, bz0)

    """Calculate distance in Meters (m)"""
    dist = b0_mag/uMag
##    print("\nDISTANCE") ######## Distance should only works for lines a radial distance within the endpoints
##    print(dist)
    return dist             


#######################################################################################
""" Start Calculating the magnetic field """
#######################################################################################

""" Create a grid of points to evaluate the magnetic field at """
# specify -lim for x,y,z and n = 1 to get single point
n, lim = 15, 20         # n must be even 
x = np.linspace(-lim, lim, n)       # n = 1 if I want only one plane
y = np.linspace(-lim, lim, n)       
z = np.linspace(-lim, lim, n)
x_grid, y_grid, z_grid = np.meshgrid(x, y, z)


""" Calculate the Magnetic Field Vectors in Teslas (T) or (kg/((s^2)(A))"""
def B(I, x, y, z, x0, y0, z0, xf, yf, zf):      # I is current, x0,y0,z0 initial line point xf,yf,zf end point of line
    global bx; global by; global bz
    """Current I is represented in Ampres (A)"""
    """Magnetic constant in Henry per Meter or (m*kg)/((s^2)(A^2))"""
    u_0 = 1 
    #u_0 = 1.25663706 * (10**(-6)) 
    
    """Nearest distance to the current carrying wire"""
    calcDist(x0, y0, z0, xf, yf, zf, x, y, z)

    """Normalize the vectors"""
    bx1 = bx0/b0_mag
    by1 = by0/b0_mag
    bz1 = bz0/b0_mag
    
    """Calculating the vector components for each point in the grid"""
    bx = I*u_0*bx1/(dist*(2*pi))
    by = I*u_0*by1/(dist*(2*pi))
    bz =  I*u_0*bz1/(dist*(2*pi))
##    print("\nBX")
##    print(bx)
##    print("\nBY")
##    print(by)
##    print("\nBZ")
##    print(bz)
    return bx,by,bz

""" Initial Point on the Wire """
x0, y0, z0 = 0, 0, 0
""" End Point on the Wire """
xf, yf, zf =  10, 0, 0

B(1, x_grid, y_grid, z_grid, x0, y0, z0, xf, yf, zf)
L(x0, y0, z0, xf, yf, zf, 10)

draw_vector(x_grid, y_grid, z_grid, bx, by, bz)
mlab.show()
