import numpy as np
from mayavi import mlab
import math
from sympy.solvers import solve
from sympy import Symbol

#mlab.figure(bgcolor = (1,1,1))

points_range = 50
phi = np.linspace(0, 2*np.pi, points_range)
theta = np.linspace(0, 2*np.pi, points_range)
phi, theta = np.meshgrid(phi, theta)

"""Common Variables"""
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

def draw_sphere(x0, y0, z0, r):
    x = (r* np.sin(phi) * np.cos(theta)) + x0
    y = (r*np.sin(phi) * np.sin(theta)) + y0
    z = (r*np.cos(phi)) + z0
    sphere = mlab.mesh(x, y, z)

def draw_field(x0, y0, z0, u, v, w):
    mlab.quiver3d(x0, y0, z0, u, v, w, line_width = 1, scale_factor= 1)


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
    """Take the dot product of a and b, then solve for t"""
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
##    ax.scatter(x, y, z)
##    ax.scatter(xr, yr, zr)
    return xr, yr, zr


"""Line of line segments"""
#(x -x0/(x1-x0)) = (y - y0/(y1 - y0)) = (z - z0/(z1 - z0))
def L(x0, y0, z0, x1, y1, z1, n):                                        # n - number of  dl line segments
    t = np.linspace(0, 1, n)
    x = x0 + (x1 - x0)*t 
    y = y0 + (y1 - y0)*t
    z = z0 + (z1 - z0)*t

    wire = mlab.plot3d(x, y, z, color=(0.062, 0.168, 0.717), tube_radius = .02)
    #ax.scatter(x, y, z)


""" Calculate the magnetic field vector """
def B(I, x0, y0, z0, x1, y1, z1, n):                                      # n - number of  dl line segments
    t = np.linspace(0, 1, n)
    x = x0 + (x1 - x0)*t 
    y = y0 + (y1 - y0)*t
    z = z0 + (z1 - z0)*t

    #plt.plot(x, y, z, linewidth=.75, color = 'orange')
    #ax.scatter(x, y, z)
    vx, vy, vz = (xL1 - xL0), (yL1 - yL0), (zL1 - zL0)
    
    for i in range(0, n): # determines the number of points on the wire that are calculated off of
        t = np.linspace(0, 2*pi, n)
        for angles in t:
            rotate(x[i], y[i], z[i], xL0, yL0, zL0, xL1, yL1, zL1, angles)
            calcDist(xL0, yL0, zL0, xL1, yL1, zL1, xr, yr, zr)
            
            ux, uy, uz = -xr, -yr, -zr
            #cross product of the two vectors
            bx = uy*vz - uz*vy
            by = uz*vx - ux*vz
            bz = ux*vy - uy*vx

            u_0 = 1
            mag = I*u_0/((2*np.pi)*dist)
            scalingFactor = 2*r

            draw_field(xr, yr, zr, (mag*bx)/(2*r), (mag*by)/(2*r), (mag*bz)/(2*r))



############################## End of Definitions ##############################
xL0, yL0, zL0, xL1, yL1, zL1 = 0, 0, 0, 1, 1, 0
L(xL0, yL0, zL0, xL1, yL1, zL1, 10)

#rL(xL0-r, yL0+r, zL0, xL1-r, yL1+r, zL1, 9)
for r in np.linspace(.5, 1, 10):
    B(1, xL0-r, yL0+r, zL0, xL1-r, yL1+r, zL1, 5)




##x0, y0, z0 = 1, 0, 1
##r = 1
##phi = np.linspace(0, np.pi, points_range)
##theta = np.linspace(0, 2*np.pi, points_range)
##u = ((r*np.cos(theta)) + x0)
##v = (0*np.cos(phi)) + y0
##w = (r*np.sin(theta)) + z0
##mlab.plot3d(u, v, w)
##
##
##""" Draw Wire """
##x_c, z_c = 1, 1
##n = 12
##zmin, zmax = -12, 12
##ymin, ymax = -12, 12
##t = np.linspace(zmin, zmax, n)
##mlab.plot3d(0*t + x_c, t, 0*t + z_c)
##
##
##
##
####x0, y0 = 2, 2
##lim = 4
####x = np.linspace(-lim + x_c, lim + y_c, n)
####y = np.linspace(-lim + x_c, lim + y_c, n)
####z = np.linspace(zmin, zmax, n + 60)
##
##x = np.linspace(-lim + x_c, lim + z_c, n)
##y = np.linspace(ymin, ymax, n + 60)
##z = np.linspace(-lim + x_c, lim + z_c, n)
##x, y, z = np.meshgrid(x, y, z)
##
##
#### bx0, by0, bz0 = B(1, x, y, z, 1, 1, 0)
##bx0, by0, bz0 = B(1, x, y, z, 1, 0, 1)
####bx1, by1, bz1 = B(1, x, y, z, -1, -1, 0)
##
###r = np.maximum(r,6)
##
##
####bx = bx0 + bx1
####by = by0 + by1
####bz = bz0 + bz1
##draw_field(x, y, z, bx0, by0, bz0)
###mlab.flow(x, y, z, bx0, by0, bz0)
###draw_field(x, y, z, bx, by, bz)
##
##
###mlab.plot3d(0*t - 1, 0*t  - 1, t)
###field_lines = mayavi.mlab.pipeline.streamline(magnitude,seedtype="line",integration_direction="both")
mlab.show()
