import numpy as np
from mayavi import mlab
import math

#mlab.figure(bgcolor = (1,1,1))

""" Common Variables """
sqrt = np.sqrt
sin, cos, tan = np.sin, np.cos, np.tan
pi = np.pi

#######################################################################################
"Graphing Definitions"
#######################################################################################

"Draw 3D points at a given point"
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
    mlab.quiver3d(x0, y0, z0, u, v, w, line_width = 2, scale_factor= 1000000)

"Line of line segments, defined by two endpoints"
#(x -x0/(x1-x0)) = (y - y0/(y1 - y0)) = (z - z0/(z1 - z0))
"(x0,y0,z0), (x1,y1,z1) are the endpoints of the wire / n - number of  dl line segments"
def wL(n, x0, y0, z0, x1, y1, z1):
    t = np.linspace(0, 1, n)
    xl = x0 + (x1 - x0)*t 
    yl = y0 + (y1 - y0)*t
    zl = z0 + (z1 - z0)*t
    wireLine = mlab.plot3d(xl, yl, zl, color=(1, 1, 1), tube_radius = .35)

#https://sites.google.com/site/glennmurray/Home/rotation-matrices-and-formulas/rotation-about-an-arbitrary-axis-in-3-dimensions
"Circle of line segments around an arbitrary axis, defined by two endpoints"
"(x,y,z) is point to be rotated to create the circle, (x0,y0,z0), (x1,y1,z1) are the endpoints of the axis of rotation"
def wC(n, t, x0, y0, z0, x1, y1, z1, x, y, z):           # n - number of  dl line segments
    global u; global v; global w; global xr; global yr; global zr; global L
    #t = np.linspace(0, 2*pi, n)
    "Direrction vectors"
    u, v, w = x1 - x0, y1 - y0, z1 - z0
    L = u**2 + v**2 + w**2
    "Calculate the points that are found from rotataing"
    xr = (((x0*(v**2 + w**2) - u*(y0*v + z0*w - u*x - v*y - w*z))*(1 - cos(t)) + L*x*cos(t) + sqrt(L)*(-z0*v + y0*w - w*y + v*z)*sin(t)))/L
    yr = (((y0*(u**2 + w**2) - v*(x0*u + z0*w - u*x - v*y - w*z))*(1 - cos(t)) + L*y*cos(t) + sqrt(L)*(z0*u - x0*w + w*x - u*z)*sin(t)))/L
    zr = (((z0*(u**2 + v**2) - w*(x0*u + y0*v - u*x - v*y - w*z))*(1 - cos(t)) + L*z*cos(t) + sqrt(L)*(-y0*u + x0*v - v*x + u*y)*sin(t)))/L
    #wireCircle = mlab.plot3d(xr, yr, zr, color=(1, 1, 1), tube_radius = .35)


#######################################################################################
"Magnetic Calculation Definitions"
#######################################################################################

"Calculate angle between position vectors"
def calcTheta(ux, uy, uz, vx, vy, vz):
    return np.arccos((ux*vx + uy*vy + uz*vz)/(sqrt((ux**2 + uy**2 + uz**2)*(vx**2 + vy**2 + vz**2))))

"Calculate the distance of a vector to the closest point on a line"
def vectorMag(vx, vy, vz):
    return sqrt(vx**2 + vy**2 + vz**2)

#http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
"(x0,y0,z0) - inital point of wire, (x1,y1,z1) - end point of wire, (xp, yp, zp) - comparison point(s)"
def calcValues(x0, y0, z0, x1, y1, z1, xp, yp, zp): #(inital point on line, end point on line, comparison point)
    global dist, theta1, theta2, bx0, by0, bz0, b0_mag
    "Calculate all of the Relevent Vectors"
    ux, uy, uz = x1 - x0, y1 - y0, z1 - z0
    vx, vy, vz = xp - x0, yp - y0, zp - z0
    wx, wy, wz = xp - x1, yp - y1, zp - z1
    "Take the Cross Product of the two endpoint vectors to the sample point"
    bx0 = vy*wz - vz*wy
    by0 = vz*wx - vx*wz
    bz0 = vx*wy - vy*wx
    "Magnitude of Initial Vector/Position Vector of Initial Point to P/Position Vector to Final Point to P"
    uMag = vectorMag(ux, uy, uz)
    vMag = vectorMag(vx, vy, vz)
    wMag = vectorMag(wx, wy, wz)
    "Magnitude of Resulting Vector from cross product"
    b0_mag = vectorMag(bx0, by0, bz0)
    "Calculate the angles between the vectors"
##    theta1 = calcTheta(ux, uy, uz, vx, vy, vz)
##    theta2 = calcTheta(ux, uy, uz, wx, wy, wz)
    "Calculate distance in Meters (m)"
    dist = b0_mag/uMag
##    print("\nDISTANCE") ######## Distance should only works for lines a radial distance within the endpoints


"Calulate the magnetic field at a given point"
def calcField_wire(xp, yp, zp):
    Bx, By, Bz = 0, 0, 0
    for i in range(0, len(currents)):
        # Variables
        I_w =  currents[i][0]
        x0_w, y0_w, z0_w = currents[i][1], currents[i][2], currents[i][3]
        x1_w, y1_w, z1_w = currents[i][4], currents[i][5], currents[i][6]
        # Calculations
        bx, by, bz = B_wire(I_w, x0_w, y0_w, z0_w, x1_w, y1_w, z1_w, xp, yp, zp)
        Bx += bx; By += by; Bz += bz
    mag = vectorMag(Bx, By, Bz)
    print("\nThe magnetic field at that point is: [" + str(round(Bx, 9)) + ", " + str(round(By, 9)) + ", " + str(round(Bz, 9)) + "].")
    print("The magnitude of the field is: " + str(round(mag, 9)) + ".")

    
"Calculate the Magnetic Field Vectors of a Wire in Teslas (T) or (kg/((s^2)(A))"
def B_wire(I, x0, y0, z0, x1, y1, z1, xp, yp, zp):      # I is current, x0,y0,z0 initial line point x1,y1,z1 end point of line
    global bx; global by; global bz
    "Current I is represented in Ampres (A)"
    "Permeability of Free Space in Henry per Meter or ((T*m)/A))"
    u_0 = 4*pi*(10**(-7))
    "Nearest distance to the current carrying wire"
    calcValues(x0, y0, z0, x1, y1, z1, xp, yp, zp)
    "Normalize the vectors and error check for divide by zero"
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        bx1 = np.nan_to_num(bx0/b0_mag)
        by1 = np.nan_to_num(by0/b0_mag)
        bz1 = np.nan_to_num(bz0/b0_mag)
    "Calculating the vector components for each point in the grid"
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        bx = np.nan_to_num(I*u_0*bx1/(dist*(2*pi)))
        by = np.nan_to_num(I*u_0*by1/(dist*(2*pi)))
        bz =  np.nan_to_num(I*u_0*bz1/(dist*(2*pi)))
    return bx,by,bz


"Calculate the Magnetic Field Vectors for a Loop in Teslas (T) or (kg/((s^2)(A))"
"I is current, (x0,y0,z0) - inital point and (x1,y1,z1) - end point of axis of rotation, (xp, yp, zp) - point to rotate around the axis"
def B_loop(n, x0, y0, z0, x1, y1, z1, xp, yp, zp):      
    global Bx; global By; global Bz
    t = np.linspace(0, 2*pi, n)
    xrList, yrList, zrList = [], [], []
    count, rp = 0, n         # Set n=1 to specify one point
    Bx, By, Bz = np.zeros((rp, rp, rp)), np.zeros((rp, rp, rp)), np.zeros((rp, rp, rp))         
    for i in t:
        wC(n, i, x0, y0, z0, x1, y1, z1, xp, yp, zp)
        xrList.append(xr), yrList.append(yr), zrList.append(zr)
        if i > 0:
            xr0, yr0, zr0 = float(xrList[count-1]), float(yrList[count-1]), float(zrList[count-1])
            bx, by, bz = B_wire(25, xr0, yr0, zr0, xr, yr, zr, x_grid, y_grid, z_grid)  
            Bx += bx; By += by; Bz += bz
            wL(10, xr0, yr0, zr0, xr, yr, zr)
        count +=  1


#######################################################################################
"End of Definitions"
#######################################################################################

"Create a grid of points to evaluate the magnetic field at"
# specify -lim for x,y,z and n = 1 to get single point
n, lim = 25, 20       # For current loop n < 10
x = np.linspace(-lim, lim, n)       # n = 1 if I want only one plane
y = np.linspace(-lim, lim, n)       
z = np.linspace(-lim, lim, n)
x_grid, y_grid, z_grid = np.meshgrid(x, y, z)


prompt = input("Would like to calculate the magnetic field around a wire(W) or loop(L)?: ")

"Start Calculating the magnetic fields "
#######################################################################################
# Calculating the magnetic field of a sufficiently large wire
#######################################################################################
if prompt == "W":               
    rx, ry, rz = n, n, n
    Bx, By, Bz = np.zeros((rx, rx, rx)), np.zeros((ry, ry, ry)), np.zeros((rz, rz, rz))
    "Current and axis for multiple Infinite Wire(s)"
    currents = [(25, -20, 0, 0, 20, 0, 0)]#, (25, 0, -20, 0, 0, 20, 0)] #(-20, -20, -20, 20, 20, 20)
    for i in range(0, len(currents)):
        for current in currents:
            # Variables
            I_w =  currents[i][0]
            x0_w, y0_w, z0_w = currents[i][1], currents[i][2], currents[i][3]
            x1_w, y1_w, z1_w = currents[i][4], currents[i][5], currents[i][6]
            # Calculations
            bx, by, bz = B_wire(I_w, x0_w, y0_w, z0_w, x1_w, y1_w, z1_w, x_grid, y_grid, z_grid)
            wL(10, x0_w, y0_w, z0_w, x1_w, y1_w, z1_w)
            Bx += bx; By += by; Bz += bz

    "Scale and plot the vectors up so they can be seen graphically "
    draw_vector(x_grid, y_grid, z_grid, Bx, By, Bz)

    print("If you would like to know exact vector of the magnetic field at a given point")
    print("use the definition calcField_wire(x, y, z).")

####################################################################################### 
# Calculating the magnetic field of a Current around Loop
#######################################################################################
elif prompt == "L":   
    "Endpoints of vector to rotate circle around"
    x0, y0, z0, x1, y1, z1 = 0, 0, 0, 0, 0, 1
    "Calculate field and graph the wire"
    B_loop(n, x0, y0, z0, x1, y1, z1, 0, 10, 0)
    "Scale and plot the vectors up so they can be seen graphically "
    draw_vector(x_grid, y_grid, z_grid, Bx, By, Bz)

"Current around Solenoid"

#######################################################################################
"""Draw the x, y, z axes"""
axes = np.linspace(-20, 20, 100)
x_axis = mlab.plot3d(0*axes, 0*axes, axes, color=(0,0,0), tube_radius = .02)
y_axis = mlab.plot3d(axes, 0*axes, 0*axes, color=(0,0,0), tube_radius = .02)
z_axis = mlab.plot3d(0*axes, axes, 0*axes, color=(0,0,0), tube_radius = .02)

"Figure Functions"
mlab.title("Magnetic Field", height = .9, size = .45)
mlab.view(distance = 200)

"Vectorbar functions"
vb = mlab.vectorbar(title = "Field Strength (T)", nb_labels = 5)
vb.scalar_bar_representation.position = [0.006644735064188584, 0.016157157980456187]
vb.scalar_bar_representation.position2 = [0.5567139298716236, 0.11830171009771967]
mlab.show()
