import numpy as np
import matplotlib.pyplot as plt
import math

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

P=np.array([2,-4])
u=np.array([0,6])
c=-u
F=-1
r1=c@c.T-F
r2=np.linalg.norm(c-P)

len = 100
theta = np.linspace(0,2*np.pi,len)
x_circ1 = np.zeros((2,len))
x_circ1[0,:] = r1*np.cos(theta)
x_circ1[1,:] = r1*np.sin(theta)
x_circ1 = (x_circ1.T + c).T

len = 100
theta = np.linspace(0,2*np.pi,len)
x_circ2 = np.zeros((2,len))
x_circ2[0,:] = r2*np.cos(theta)
x_circ2[1,:] = r2*np.sin(theta)
x_circ2 = (x_circ2.T + P).T


y1 = np.linspace(-30,30,len)
y2 = np.power(y1,2)

y = np.vstack((y2/8,y1))


plt.plot(x_circ1[0,:],x_circ1[1,:],label='$Required circle$')
plt.plot(x_circ2[0,:],x_circ2[1,:],label='$Given circle$')
plt.plot(y[0,:],y[1,:],label='$Given parabola$')




plt.plot(c[0], c[1], 'o')
plt.text(c[0] * (1 + 0.1), c[1] * (1 - 0.1) , 'c')


plt.plot(P[0], P[1], 'o')
plt.text(P[0] * (1 + 0.1), P[1] * (1 - 0.1) , 'P')


ax.plot()
plt.xlabel('$x$');plt.ylabel('$y$')
plt.axis('equal')
plt.legend(loc='best')

plt.grid()
plt.show()

