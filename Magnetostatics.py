import numpy as np
from mayavi import mlab
from sympy.solvers import solve
from sympy import Symbol
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#mlab.figure(bgcolor = (1,1,1))

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')

lim = 1
ax.axes.set_xlim([-lim,lim])
ax.axes.set_ylim([-lim,lim])
ax.axes.set_zlim([-lim,lim])

"""Axes"""
"x"
x1 = [-2, 2]
y1 = [0, 0]
z1 = [0, 0]

plt.plot(x1, y1, z1, linewidth=.75, color = 'black')  

"y"
x2 = [0, 0]
y2 = [-2, 2]
z2 = [0, 0]

plt.plot(x2, y2, z2, linewidth=.75, color = 'black')  

"z"
x3 = [0, 0]
y3 = [0, 0]
z3 = [-2, 2]

plt.plot(x3, y3, z3, linewidth=.75, color = 'black')

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

def draw_field(x0, y0, z0, u, v, w):
    plt.quiver(x0, y0, z0, u, v, w)

"""Line of line segments"""
#(x -x0/(x1-x0)) = (y - y0/(y1 - y0)) = (z - z0/(z1 - z0))
def L(x0, y0, z0, x1, y1, z1, n):                                        # n - number of  dl line segments
    t = np.linspace(0, 1, n)
    x = x0 + (x1 - x0)*t 
    y = y0 + (y1 - y0)*t
    z = z0 + (z1 - z0)*t

    wire = mlab.plot3d(x, y, z, color=(0.062, 0.168, 0.717), tube_radius = .02)


n = 4
##x0, y0 = 2, 2
lim = 4
x = np.linspace(-lim, lim, 1)
y = np.linspace(-lim, lim, n)
z = np.linspace(-lim, lim, n)
x_grid, y_grid, z_grid = np.meshgrid(x, y, z)



#http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
def vectorMag(vx, vy, vz):
    return np.sqrt(vx**2 + vy**2 + vz**2)
def calcDist(x0, y0, z0, x1, y1, z1, xp, yp, zp): #(inital point on line, end point on line, comarison point)
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



def B2(I, x, y, z, x0, y0, z0, xf, yf, zf):                                      # I is current, x0 and y0 are the current line position
    global bx; global by; global bz
    t = np.linspace(0, 1, 20)
##    xL = x0 + (xf - x0)*t 
##    yL = y0 + (yf - y0)*t
##    zL = z0 + (zf - z0)*t

##    plt.plot(xL, yL, zL, linewidth=.75, color = 'orange')
##    ax.scatter(xL, yL, zL)
    #Current and magnetic constant
    m_u = 1
    
    #m_u = 1.25663706 * (10**(-6))                                   # m*kg/((s^2)*(A^2))
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
B2(1, x_grid, y_grid, z_grid, -1, 0, 0, 1, 0, 0)
L(-1, 0, 0, 1, 0, 0, 10)

#draw_field(x_grid, y_grid, z_grid, bx, by, bz)
draw_vector(x_grid, y_grid, z_grid, bx, by, bz)

axes = np.linspace(-20, 20, 100)
x_axis = mlab.plot3d(0*axes, 0*axes, axes, color=(0,0,0), tube_radius = .02)
y_axis = mlab.plot3d(axes, 0*axes, 0*axes, color=(0,0,0), tube_radius = .02)
z_axis = mlab.plot3d(0*axes, axes, 0*axes, color=(0,0,0), tube_radius = .02)
mlab.show()

ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')
#plt.show()
