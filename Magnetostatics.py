import numpy as np
from mayavi import mlab
import math

#mlab.figure(bgcolor = (1,1,1))

""" Common Variables """
sqrt = np.sqrt
sin = np.sin
cos = np.cos
tan = np.tan
pi = np.pi

"""Draw the x, y, z axes"""
axes = np.linspace(-20, 20, 100)
x_axis = mlab.plot3d(0*axes, 0*axes, axes, color=(0,0,0), tube_radius = .02)
y_axis = mlab.plot3d(axes, 0*axes, 0*axes, color=(0,0,0), tube_radius = .02)
z_axis = mlab.plot3d(0*axes, axes, 0*axes, color=(0,0,0), tube_radius = .02)


#######################################################################################
"""Graphing Definitions"""
#######################################################################################

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


"Draw a vector fields given the tail locations and components"
def draw_vector(x0, y0, z0, u, v, w):
    mlab.quiver3d(x0, y0, z0, u, v, w, line_width = 1, scale_factor= 1000000)


"Line of line segments, defined by two endpoints"
#(x -x0/(x1-x0)) = (y - y0/(y1 - y0)) = (z - z0/(z1 - z0))
"(x0,y0,z0), (x1,y1,z1) are the endpoints of the wire / n - number of  dl line segments"
def wL(x0, y0, z0, x1, y1, z1, n): 
    t = np.linspace(0, 1, n)
    xl = x0 + (x1 - x0)*t 
    yl = y0 + (y1 - y0)*t
    zl = z0 + (z1 - z0)*t
    wireLine = mlab.plot3d(xl, yl, zl, color=(1, 1, 1), tube_radius = .02)
    print(xl,yl,zl)
    

#https://sites.google.com/site/glennmurray/Home/rotation-matrices-and-formulas/rotation-about-an-arbitrary-axis-in-3-dimensions
"Circle of line segments around an arbitrary axis, defined by two endpoints"
"(x,y,z) is point to be rotated to create the circle, (x0,y0,z0), (x1,y1,z1) are the endpoints of the axis"
def wC(x, y, z, a, b, c, d, e, f, n):           # n - number of  dl line segments
    global xr, yr, zr
    t = np.linspace(0, 2*pi, n)
    "Direrction vector"
    u, v, w = d - a, e - b, f - c
    L = u**2 + v**2 + w**2

    xr = (((a*(v**2 + w**2) - u*(b*v + c*w - u*x - v*y - w*z))*(1 - cos(t)) + L*x*cos(t) + sqrt(L)*(-c*v + b*w - w*y + v*z)*sin(t)))/L
    yr = (((b*(u**2 + w**2) - v*(a*u + c*w - u*x - v*y - w*z))*(1 - cos(t)) + L*y*cos(t) + sqrt(L)*(c*u - a*w + w*x - u*z)*sin(t)))/L
    zr = (((c*(u**2 + v**2) - w*(a*u + b*v - u*x - v*y - w*z))*(1 - cos(t)) + L*z*cos(t) + sqrt(L)*(-b*u + a*v - v*x + u*y)*sin(t)))/L
    wireCircle = mlab.plot3d(xr, yr, zr, color=(1, 1, 1), tube_radius = .02)
    print(xr,yr,zr)
    
x0, y0, z0, x1, y1, z1, n = 0, 0, 0, 0, 0, 1, 6
wC(0, 7, 0, x0, y0, z0, x1, y1, z1, n)


#######################################################################################
"Magnetic Calculation Definitions"
#######################################################################################
"Calculate angle between point P and each of the endpoints"
def calcTheta(ux, uy, uz, vx, vy, vz):
    return np.arccos((ux*vx + uy*vy + uz*vz)/(sqrt((ux**2 + uy**2 + uz**2)*(vx**2 + vy**2 + vz**2))))


"Calculate the distance of a vector to the closest point on a line"
def vectorMag(vx, vy, vz):
    return sqrt(vx**2 + vy**2 + vz**2)
#http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
"(x0,y0,z0) - inital point of wire, (x1,y1,z1) - end point of wire, (xp, yp, zp) - comparison point"
def calcDist(x0, y0, z0, x1, y1, z1, xp, yp, zp): #(inital point on line, end point on line, comparison point)
    global dist, ux, uy, uz, vx, vy, vz, wx, wy, wz, bx0, by0, bz0, b0_mag
    "Calculate all of the Relevent Vectors"
    ux, uy, uz = x1 - x0, y1 - y0, z1 - z0
    vx, vy, vz = xp - x0, yp - y0, zp - z0
    wx, wy, wz = xp - x1, yp - y1, zp - z1
##    print(ux, uy, uz)
##    print(vx, vy, vz)
##    print("\nPOINTS")
##    print(xp, yp, zp)
    "Take the Cross Product of the two endpoint vectors to the sample point"
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
    "Magnitude of Initial Vector/Position Vector of Initial Point to P/Position Vector to Final Point to P"
    uMag = vectorMag(ux, uy, uz)
    vMag = vectorMag(vx, vy, vz)
    wMag = vectorMag(wx, wy, wz)
    "Magnitude of Resulting Vector from cross product"
    b0_mag = vectorMag(bx0, by0, bz0)

    "Calculate distance in Meters (m)"
    dist = b0_mag/uMag
##    print("\nDISTANCE") ######## Distance should only works for lines a radial distance within the endpoints
##    print(dist)    


"Calculate the Magnetic Field Vectors in Teslas (T) or (kg/((s^2)(A))"
def B(I, x, y, z, x0, y0, z0, xf, yf, zf):      # I is current, x0,y0,z0 initial line point xf,yf,zf end point of line
    global bx; global by; global bz
    "Current I is represented in Ampres (A)"
    "Permeability of Free Space in Henry per Meter or ((T*m)/A))"
    u_0 = 4*pi*(10**(-7))
    
    "Nearest distance to the current carrying wire"
    calcDist(x0, y0, z0, xf, yf, zf, x, y, z)

    "Calculate the angles from the endpoints to our grid points"
    theta1 = calcTheta(ux, uy, uz, vx, vy, vz)
    theta2 = calcTheta(ux, uy, uz, wx, wy, wz)
    ##print(theta1, theta2)

    "Normalize the vectors"
    bx1 = bx0/b0_mag
    by1 = by0/b0_mag
    bz1 = bz0/b0_mag
    
    "Calculating the vector components for each point in the grid"
    bx = I*u_0*bx1/(dist*(2*pi))
    by = I*u_0*by1/(dist*(2*pi))
    bz =  I*u_0*bz1/(dist*(2*pi))
##    print(sqrt(bx**2+by**2+bz**2))
##    print("\nBX")
##    print(bx)
##    print("\nBY")
##    print(by)
##    print("\nBZ")
##    print(bz)
    return bx,by,bz


#######################################################################################
"""Start Calculating the magnetic field"""
#######################################################################################

""" Initial Point on the Wire """
x0, y0, z0 = 0, 0, 0
""" End Point on the Wire """
xf, yf, zf =  10, 0, 0

""" Create a grid of points to evaluate the magnetic field at """
# specify -lim for x,y,z and n = 1 to get single point
n, lim = 20, 20        # n must be even 
x = np.linspace(-lim, lim, n)       # n = 1 if I want only one plane
y = np.linspace(-lim, lim, n)       
z = np.linspace(-lim, lim, n)
x_grid, y_grid, z_grid = np.meshgrid(x, y, z)

B(25, x_grid, y_grid, z_grid, x0, y0, z0, xf, yf, zf)
wL(x0, y0, z0, xf, yf, zf, 10)

"""Scale the vectors up so they can be seen graphically"""
draw_vector(x_grid, y_grid, z_grid, bx, by, bz)

"""Figure Functions"""
mlab.title("Magnetic Field", height = .9, size = .45)
mlab.view(distance = 200)

"""Vectorbar functions"""
vb = mlab.vectorbar(title = "Field Strength (T)", nb_labels = 5)
vb.scalar_bar_representation.position = [0.006644735064188584, 0.016157157980456187]
vb.scalar_bar_representation.position2 = [0.5567139298716236, 0.11830171009771967]
mlab.show()
