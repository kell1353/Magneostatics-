import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy.solvers import solve
from sympy import Symbol
import math

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')

lim = 1.5
ax.axes.set_xlim([-lim,lim])
ax.axes.set_ylim([-lim,lim])
ax.axes.set_zlim([-lim,lim])

sqrt = np.sqrt
sin = np.sin
cos = np.cos
tan = np.tan
pi = np.pi

"""Axes"""
"x"
x1 = [-1, 1]
y1 = [0, 0]
z1 = [0, 0]

plt.plot(x1, y1, z1, linewidth=.75, color = 'black')  

"y"
x2 = [0, 0]
y2 = [-1, 1]
z2 = [0, 0]

plt.plot(x2, y2, z2, linewidth=.75, color = 'black')  

"z"
x3 = [0, 0]
y3 = [0, 0]
z3 = [-1, 1]

plt.plot(x3, y3, z3, linewidth=.75, color = 'black')

"""Circular loop of line segments"""
##def p(x0, y0, z0 , r, n):                                        # r - radius, n - number of  dl line segments
##    t = np.linspace(0, 2*pi, n)
##    #x = (x0 + (r * np.cos(2*pi*t)))
##    theta = 45
##    x = x0 + r* np.cos(t)* (cos(np.deg2rad(-theta))-np.sin(t)*sin(np.deg2rad(-theta)))
##    y = y0 + r* np.cos(t)* (sin(np.deg2rad(-theta))-np.sin(t)*cos(np.deg2rad(-theta)))
##    #y = (y0 + (0 * t))
##    z = z0 + (r * np.sin(t))
##    plt.plot(x, y, z, linewidth=.75, color = 'black')
##    print(t)
##    print(x)
##    print(y)
##    ax.scatter(x, y, z)
##p(0, 0, 0, 1, 60)

def draw_field(x0, y0, z0, u, v, w):
    plt.quiver(x0, y0, z0, u, v, w)


def calcDist(x0, y0, z0, x1, y1, z1, xp, yp, zp): #(inital point on line, end point on line, comarison point)
    global dist
    t = Symbol('t')
    xl = x0 + (x1 - x0)*t 
    yl = y0 + (y1 - y0)*t
    zl = z0 + (z1 - z0)*t
    a = [x1 - x0, y1 - y0, z1 - z0]

    """Vector from point to arbitrary point on a line"""
    vx, vy, vz =  xl - xp, yl - yp, zl - zp
    b = [vx, vy, vz]

    """Solve for t"""
    sol = solve((a[0]*b[0] + a[1]*b[1] + a[2]*b[2]), t)

    """Calculate the closest point on line to our point"""
    xpf = xl.replace(t, float(sol[0]))
    ypf = yl.replace(t, float(sol[0]))
    zpf = zl.replace(t, float(sol[0]))
    pf = (xpf, ypf, zpf)

    """Solve for the final vector"""
    xf = vx.replace(t, float(sol[0]))
    yf = vy.replace(t, float(sol[0]))
    zf = vz.replace(t, float(sol[0]))
    vf = (xf, yf, zf)
    
    dist = math.sqrt(xf**2 + yf**2 + zf**2)
    return dist



#https://sites.google.com/site/glennmurray/Home/rotation-matrices-and-formulas/rotation-about-an-arbitrary-axis-in-3-dimensions
"""Rotation of a point around an arbitrary axis"""
def rotate(x, y, z, a, b, c, d, e, f, theta):
    global xr, yr, zr    
    """Direrction vector"""
    u = d - a
    v = e - b
    w = f - c

    L = u**2 + v**2 + w**2
    
    xr = (((a*(v**2 + w**2) - u*(b*v + c*w - u*x - v*y - w*z))*(1 - cos(theta)) + L*x*cos(theta) + sqrt(L)*(-c*v + b*w - w*y + v*z)*sin(theta)))/L
    yr = (((b*(u**2 + w**2) - v*(a*u + c*w - u*x - v*y - w*z))*(1 - cos(theta)) + L*y*cos(theta) + sqrt(L)*(c*u - a*w + w*x - u*z)*sin(theta)))/L
    zr = (((c*(u**2 + v**2) - w*(a*u + b*v - u*x - v*y - w*z))*(1 - cos(theta)) + L*z*cos(theta) + sqrt(L)*(-b*u + a*v - v*x + u*y)*sin(theta)))/L
    ax.scatter(x, y, z)
    ax.scatter(xr, yr, zr)
    return xr, yr, zr


"""Line of line segments"""
#(x -x0/(x1-x0)) = (y - y0/(y1 - y0)) = (z - z0/(z1 - z0))
def L(x0, y0, z0, x1, y1, z1, n):                                        # n - number of  dl line segments
    t = np.linspace(0, 1, n)
    x = x0 + (x1 - x0)*t 
    y = y0 + (y1 - y0)*t
    z = z0 + (z1 - z0)*t

    plt.plot(x, y, z, linewidth=.75, color = 'orange')
    ax.scatter(x, y, z)
    

"""Line of paralell line segments"""
def rL(x0, y0, z0, x1, y1, z1, n):                                      # n - number of  dl line segments
    t = np.linspace(0, 1, n)
    x = x0 + (x1 - x0)*t 
    y = y0 + (y1 - y0)*t
    z = z0 + (z1 - z0)*t

    #plt.plot(x, y, z, linewidth=.75, color = 'orange')
    #ax.scatter(x, y, z)
    vx, vy, vz = (xL1 - xL0), (yL1 - yL0), (zL1 - zL0)
    
    for i in range(0, 1):
        t = np.linspace(0, 2*pi, n)
        for angles in t:
            rotate(x[i], y[i], z[i], xL0, yL0, zL0, xL1, yL1, zL1, angles)
            calcDist(xL0, yL0, zL0, xL1, yL1, zL1, xr, yr, zr)
            
            ux, uy, uz = -xr, -yr, -zr
            #cross product of the two vectors
            bx = uy*vz - uz*vy
            by = uz*vx - ux*vz
            bz = ux*vy - uy*vx

            I, u_0 = 1, 1
            mag = I*u_0/((2*np.pi)*dist)
            scalingFactor = 2*r

            m = math.sqrt(bx**2 + by**2 + bz**2)
            draw_field(xr, yr, zr, (mag*bx)/(2*r), (mag*by)/(2*r), (mag*bz)/(2*r))


##L(0, 0, 0, 1, 1, 0, 3)
xL0, yL0, zL0, xL1, yL1, zL1 = 0, 0, 0, 1, 1, 0
L(xL0, yL0, zL0, xL1, yL1, zL1, 4)

#rL(xL0-r, yL0+r, zL0, xL1-r, yL1+r, zL1, 9)
for r in np.linspace(.5, 1, 5):
    rL(xL0-r, yL0+r, zL0, xL1-r, yL1+r, zL1, 9)


ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')
plt.show()
