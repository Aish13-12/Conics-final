import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import linalg as LA
from coeffs import*


a=9
b=-18
c=5
e_1=(-b+(b**2-4*a*c)**(1/2))*(1/(2*a))
print(e_1)
e_2=(-b-(b**2-4*a*c)**(1/2))*(1/(2*a))
print(e_2)

S=np.array([5,0])
l=(S@S.T-9)/np.linalg.norm(S)
l=abs(l)
print(l)

if e_1>1:
	a=(l*e_1)/(e_1**2-1)
	print(a)
	b=a*((e_1**2-1)**(1/2))
	print(b)
	p=a**2-b**2
	print(p)

if e_2>1:
	a=(l*e_2)/(e_2**2-1)
	print(a)
	b=a*((e_2**2-1)**(1/2))
	p=a**2-b**2
	print(p)
	
V=np.array(([16,0],[0,-9]))
u=np.array([0,0])
F=-144
eigval,eigvec = LA.eig(V)
D = np.diag(eigval)


y11 = np.linspace(3,4,len)
y21 = np.sqrt((1-D[0,0]*np.power(y11,2))/(D[1,1]))
y31 = -1*np.sqrt((1-D[0,0]*np.power(y11,2))/(D[1,1]))
y1 = np.hstack((np.vstack((y11,y21)),np.vstack((y11,y31))))

y12 = np.linspace(-4,-3,len)
y22 = np.sqrt((1-D[0,0]*np.power(y12,2))/(D[1,1]))
y32 = -1*np.sqrt((1-D[0,0]*np.power(y12,2))/(D[1,1]))
y2 = np.hstack((np.vstack((y12,y22)),np.vstack((y12,y32))))


plt.plot(y1[0,:len],y1[1,:len],color='b',label='Hyperbola')
plt.plot(y1[0,len+1:],y1[1,len+1:],color='b')

plt.plot(y2[0,:len],y2[1,:len],color='b')
plt.plot(y2[0,len+1:],y2[1,len+1:],color='b')

plt.axis('equal')

plt.xlabel('$x$');plt.ylabel('$y$')

plt.legend(loc='best')

plt.grid()
plt.show()

