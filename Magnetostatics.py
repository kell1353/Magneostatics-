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

points_range = 50
phi = np.linspace(0, 2*np.pi, points_range)
theta = np.linspace(0, 2*np.pi, points_range)
phi, theta = np.meshgrid(phi, theta)

""" Draw 3D points at a given point """
def draw_sphere(x0, y0, z0, r):
    x = (r*sin(phi) * cos(theta)) + x0
    y = (r*sin(phi) * sin(theta)) + y0
    z = (r*cos(phi)) + z0
    sphere = mlab.mesh(x, y, z)

""" Draw a vector given the location and components """
def draw_vector(x0, y0, z0, u, v, w):
    mlab.quiver3d(x0, y0, z0, u, v, w, line_width = 1, scale_factor= 4)

""" Line of line segments """
#(x -x0/(x1-x0)) = (y - y0/(y1 - y0)) = (z - z0/(z1 - z0))
def L(x0, y0, z0, x1, y1, z1, n):                                        # n - number of  dl line segments
    t = np.linspace(0, 1, n)
    x = x0 + (x1 - x0)*t 
    y = y0 + (y1 - y0)*t
    z = z0 + (z1 - z0)*t

    wire = mlab.plot3d(x, y, z, color=(0.062, 0.168, 0.717), tube_radius = .02)


""" Calculate the distance of a vector to the closest point on a line """
#http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
def vectorMag(vx, vy, vz):
    return sqrt(vx**2 + vy**2 + vz**2)
def calcDist(x0, y0, z0, x1, y1, z1, xp, yp, zp): #(inital point on line, end point on line, comparison point)
    global dist, ux, uy, uz, bx0, by0, bz0
    ux, uy, uz = x1 - x0, y1 - y0, z1 - z0
    a = [ux, uy, uz]
    print(ux, uy, uz)
    vx, vy, vz = x0 - xp, y0 - yp, z0 - zp
    #print(vx, vy, vz)
    print("\nPOINTS")
    print(xp, yp, zp)

    print("\nCROSS PRODUCT")
    bx0 = uy*vz - uz*vy
    #print(bx0)
    by0 = uz*vx - ux*vz
    #print(by0)
    bz0 = ux*vy - uy*vx
    #print(bz0)

    print("\nDISTANCE")
    dist = vectorMag(bx0, by0, bz0)/vectorMag(ux, uy, uz)
    print(dist)
    return dist


""" Create a grid of points to evaluate the magnetic field at """
n, lim = 20, 10                                                                                # n must be even 
x = np.linspace(-10, 10, n)                                                        # n = 1 if I want only one plane
y = np.linspace(-lim, lim, n)
z = np.linspace(-lim, lim, n)
x_grid, y_grid, z_grid = np.meshgrid(x, y, z)


""" Calculate the magnetic field vector """
def B(I, x, y, z, x0, y0, z0, xf, yf, zf):                                      # I is current, x0,y0,z0 initial line point xf,yf,zf end point of line
    global bx; global by; global bz
    t = np.linspace(0, 1, 20)
    #Current and magnetic constant
    m_u = 1
    
    #m_u = 1.25663706 * (10**(-6))                                         # m*kg/((s^2)*(A^2))
    #Distance from current
    calcDist(x0, y0, z0, xf, yf, zf, x, y, z)
    vx, vy, vz = -(x0 - x), -(y0 - y), -(z0 - z)

    wx = uy*vz - uz*vy
    wy = uz*vx - ux*vz
    wz = ux*vy - uy*vx

    #Calculating the vector components
    print("\nMAGNITUDE")
    mag = I*m_u/(2*np.pi)
    print(mag)
    
    bx = mag * (wx/dist**2)
    print("\nBX")
    print(bx)
    by = mag * (wy/dist**2)
    print("\nBY")
    print(by)
    bz =  mag * (wz/dist**2)
    print("\nBZ")
    print(bz)
    return bx,by,bz

""" Initial Point on the Wire """
x0, y0, z0 = -1, -4, 0
""" End Point on the Wire """
xf, yf, zf =  1, 1, 1

B(1, x_grid, y_grid, z_grid, x0, y0, z0, xf, yf, zf)
L(x0, y0, z0, xf, yf, zf, 10)

draw_vector(x_grid, y_grid, z_grid, bx, by, bz)
mlab.show()
