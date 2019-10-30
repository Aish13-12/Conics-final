import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import linalg as LA
from coeffs import*

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
len = 100
theta = np.linspace(-5,5,len)

V1=np.array([1,0])
V2=np.array([0,-1/3])
V = np.vstack((V1,V2))
print(V)
a=2**(1/2)
b=3**(1/2)
P1=np.array([a,b])
print(P1)
#P=P.reshape(2,1)
#V=V.reshape(2,2)
u=np.array([0,0])

F = 1
eigval,eigvec = LA.eig(V)
print(eigval)
print(eigvec)

D = np.diag(eigval)
P = eigvec
#print("D=\n",D)
#print("P=\n",P)

n= u + V.T@P1
m=omat@n
x_AB=line_dir_pt(m,P1,-3,6)




y11 = np.linspace(1,3,len)
y21 = np.sqrt((1-D[0,0]*np.power(y11,2))/(D[1,1]))
y31 = -1*np.sqrt((1-D[0,0]*np.power(y11,2))/(D[1,1]))
y1 = np.hstack((np.vstack((y11,y21)),np.vstack((y11,y31))))

y12 = np.linspace(-3,-1,len)
y22 = np.sqrt((1-D[0,0]*np.power(y12,2))/(D[1,1]))
y32 = -1*np.sqrt((1-D[0,0]*np.power(y12,2))/(D[1,1]))
y2 = np.hstack((np.vstack((y12,y22)),np.vstack((y12,y32))))




#Plotting standard hyperbola
plt.plot(y1[0,:len],y1[1,:len],color='b',label='Hyperbola')
plt.plot(y1[0,len+1:],y1[1,len+1:],color='b')

plt.plot(y2[0,:len],y2[1,:len],color='b')
plt.plot(y2[0,len+1:],y2[1,len+1:],color='b')

plt.plot(P1[0], P1[1], 'o')
plt.text(P1[0] * (1 + 0.5), P1[1] * (1 - 0.1) , 'P1')

plt.plot(x_AB[0,:],x_AB[1,:],label='$Tangent$')
ax.plot()
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid()
plt.axis('equal')

plt.show()
