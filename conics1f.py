import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import linalg as LA
from coeffs import *

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

len = 50
o=np.vstack([0,0])
y_11 = np.linspace(-5,5,len)
y_21= np.power(y_11,2)

y_1 = np.vstack((y_11,y_21/3))


y_22 = np.linspace(-5,5,len)
y_21 = np.power(y_22,2)

y_2= np.vstack((y_21/3,y_22))


V1=np.array(([0,0],[0,1]))
P1=np.array([3/4,-3/2])
P2=np.array([-3/2,3/4])
u1=np.array([-1.5,0])
n= u1 + V1.T@P1
m=omat@n
l = line_dir_pt(m,P1,-5,18)
 
plt.plot(y_1[0,:],y_1[1,:],label='Parabola_1')
plt.plot(y_2[0,:],y_2[1,:],label='Parabola_2')
plt.plot(o[0], o[1], 'o')
plt.text(o[0] * (1 + 0.5), o[1] * (1 - 0.1) , 'O')

plt.plot(P1[0], P1[1], 'o')
plt.text(P1[0] * (1 + 0.5), P1[1] * (1 - 0.1) , 'P1')

plt.plot(P2[0], P2[1], 'o')
plt.text(P2[0] * (1 + 0.5), P2[1] * (1 - 0.1) , 'P2')

plt.plot(l[0,:],l[1,:],label='Tangent')

ax.plot()
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
plt.show()
