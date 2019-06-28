import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

"""Line of line segments"""
def L(x0, y0, z0, x1, y1, z1, n):                                        # n - number of  dl line segments
    t = np.linspace(0, 1, n)
    x = x0 + (x1 - x0)*t
    y = y0 + (y1 - y0)*t
    z = z0 + (z1 - z0)*t
    plt.plot(x, y, z, linewidth=.75, color = 'orange')
    ax.scatter(x, y, z)
L(0, 0, 0, 1, 1, 0, 3)
L(0, 0, 0, 1, 1, 1, 3)

"""Rotation of a point around an arbitrary axis"""
def rotate(x, y, z, a, b, c, d, e, f, theta):
    """Direrction vector"""
    u = d - a
    v = e - b
    w = f - c

    L = u**2 + v**2 + w**2
    
    x1 = (((a*(v**2 + w**2) - u*(b*v + c*w - u*x - v*y - w*z))*(1 - cos(theta)) + L*x*cos(theta) + sqrt(L)*(-c*v + b*w - w*y + v*z)*sin(theta)))/L
    y1 = (((b*(u**2 + w**2) - v*(a*u + c*w - u*x - v*y - w*z))*(1 - cos(theta)) + L*y*cos(theta) + sqrt(L)*(c*u - a*w + w*x - u*z)*sin(theta)))/L
    z1 = (((c*(u**2 + v**2) - w*(a*u + b*v - u*x - v*y - w*z))*(1 - cos(theta)) + L*z*cos(theta) + sqrt(L)*(-b*u + a*v - v*x + u*y)*sin(theta)))/L
    ax.scatter(x, y, z)
    ax.scatter(x1, y1, z1)

##    print(x1)
##    print(y1)
##    print(z1)
##
##    print(L)
##    print(sqrt(L))
t = np.linspace(0, 2*pi, 25)
#rotate(1, 0, 0, 0, -1, 0, 0, 1, 0, pi/4)
#rotate(0, 1, 0, 0, 0, 0, 1/sqrt(3), 1/sqrt(3), 1/sqrt(3), 4.1887902047863905)

for angles in t:
    rotate(0, 1, 0, 0, 0, 0, 1, 1, 1, angles)
    #rotate(0, 2, 0, 0, 0, 0, 1, 1, 1, angles)


"""Original Line"""
#p0 = (-1, -1, -1)
#p1 = (1, 1, 1)

# line equation is :
#x = x0 + (x1 - (x0))
#y = y0 + (y1 - (y0))
#z = z0 + (z1 - (z0))
x0, x1 = 0, 0
y0, y1 = -1, 1
z0, z1 = 0, 0

t = np.linspace(0, 1, 10)

x = x0 + (x1 - (x0))*t
y = y0 + (y1 - (y0))*t
z = z0 + (z1 - (z0))*t

plt.plot(x, y, z, linewidth=.75)  

"""Wire that lies straight on the xy plane"""
def plot_vec(x0, y0, z0, x1, y1, z1):
    x = [x0, x1]
    y = [y0, y1]
    z = [z0, z1]

    plt.plot(x, y, z, linewidth=.75)  
    ax.scatter(x, y, z)
    
    phi = 45
    theta = np.arctan2(x[1]-x[0], z[1]-z[0])
    print(theta)
    I = 1
    u_0 = 1
    r = np.sqrt((x[1]-x[0])**2 + (z[1]-z[0])**2)
    #mag = 4*I*u_0/((2*np.pi)*r)
    mag = .5

    x = [x1, x1 + (mag)*(cos(theta))*cos()]
    print(x)
    y = [y1, y1 + (mag)*(-cos(np.deg2rad(90)))]
    print(y)
    z = [z1, z1 + (mag)*(-sin(theta))]

    plt.plot(x, y, z, linewidth=.75, color = 'black')  
    ax.scatter(x, y, z)

##plot_vec(0 + x0, 0, 0 + z0, 1, 0, 1)
##plot_vec(0 + x0, 0, 0 + z0, -1, 0, 1)
##plot_vec(0 + x0, 0, 0 + z0, -1, 0, 0)
##plot_vec(0 + x0, 0, 0 + z0, 0, 0, 1)
##plot_vec(0 + x0, .5, 0 + z0, -1, .5, .5)

##plot_vec(0, 0, 0, 0, 0, 1)
##plot_vec(0, 0, 0, 1, -1, 0)
##plot_vec(0, 0, 0, 1, -1, 1)
##plot_vec(.5, .5, 0, 1.5, -.5, 0)

ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')
plt.show()
