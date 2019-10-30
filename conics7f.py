import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import linalg as LA

A_1=np.array([4,3])
K_1=12
A_2=np.array([3,4])
K_2=12
A=np.vstack((A_1,A_2))
K=np.array((K_1,K_2))
K=K.T
C=K@LA.inv(A)
print(C)


fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

#Circle parameters
a_1=np.array([0,1])
a_2=np.array([1,0])
V=np.vstack((a_1,a_2))

u = -(C.T@V)/2
u=u.T
c=-u
F =0

#defining centre and radius of Circle C1
c=-u
r=np.sqrt(c.T@c-F)

#Generating points on the circle C1
len = 100
theta = np.linspace(0,2*np.pi,len)
x_circ = np.zeros((2,len))
x_circ[0,:] = r*np.cos(theta)
x_circ[1,:] = r*np.sin(theta)
x_circ = (x_circ.T + c).T

plt.plot(x_circ[0,:],x_circ[1,:],label='$Locus$')

plt.plot(C[0], C[1], 'o')
plt.text(C[0] * (1 + 0.1), C[1] * (1 - 0.1) , 'C')
plt.plot(c[0], c[1], 'o')
plt.text(c[0] * (1 + 0.1), c[1] * (1 - 0.1) , 'O')

ax.plot()
plt.xlabel('$x$');plt.ylabel('$y$')

plt.legend(loc='best')

plt.grid()
plt.show()
