import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111, projection='3d')
ax.axes.axes.set_aspect('equal')

"""Axes"""
"x"
x = [-1, 1]
y = [0, 0]
z = [0, 0]

plt.plot(x, y, z, linewidth=.75, color = 'black')  

"y"
x = [0, 0]
y = [-1, 1]
z = [0, 0]

plt.plot(x, y, z, linewidth=.75, color = 'black')  

"z"
x = [0, 0]
y = [0, 0]
z = [-1, 1]

plt.plot(x, y, z, linewidth=.75, color = 'black')  


"""Original Line"""
#p0 = (-1, -1, -1)
#p1 = (1, 1, 1)

# line equation is :
#x = x0 + (x1 - (x0))
#y = y0 + (y1 - (y0))
#z = z0 + (z1 - (z0))

##x = [-1, 1]
##y = [-1, 1]
##z = [-1, 1]

#for t in [0, 1]
t = np.linspace(0, 1, 10)
x = -1 + 2*t
y = -1 + 2*t
z = -1 + 2*t

plt.plot(x, y, z, linewidth=.75)  



"""Side Lines"""

##x = [0, -np.sqrt(2)/2]
##y = [0, np.sqrt(2)/2]
##z = [0, 0]

x = [0, -.5]
y = [0, .5]
z = [0, 0]

plt.plot(x, y, z, linewidth=.75)  
ax.scatter(x, y, z)

I, u_0 = 1, 1
r = np.sqrt((-.5 - 0)**2 + (.5 - 0)**2 + (0 - 0)**2)
##mag = I*u_0/((2*np.pi)*r)
mag = 1

x1 = r*np.cos(np.sqrt(2)/2)
y1 = r*np.sin(np.sqrt(2)/2)

x = [-.5, r*-.707]
y = [.5, r*.707]
z = [0, mag]

plt.plot(x, y, z, linewidth=.75)  
ax.scatter(x, y, z)




x = [.5, .75]
y = [.5, .25]
z = [.5, .5]

plt.plot(x, y, z, linewidth=.75)  
ax.scatter(x, y, z)

x = [-.75, -1]
y = [-.75, -.5]
z = [-.75, -.75]

plt.plot(x, y, z, linewidth=.75)  
ax.scatter(x, y, z)




ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')
plt.show()
