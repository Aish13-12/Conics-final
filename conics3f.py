import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from coeffs import*

fig=plt.figure()
ax=fig.add_subplot(111,aspect='equal')

V=np.array(([25,0],[0,9]))
u=np.array([0,0])
P=np.array([1.5,2.5*(3**(1/2))])
F=-225
m,n=LA.eig(V)
m=m/-F
a=1/np.sqrt(m[0])
print(a)
b=1/np.sqrt(m[1])
print(b)
s=P.T@V+u.T
#t=LA.norm(s)
#print(t)

if a>b:
	e=np.sqrt(1-b**2/a**2)
	f1=np.array([a*e,0])
	f2=np.array([-a*e,0])
	print(e)
	
if a<b:
	e=np.sqrt(1-a**2/b**2)
	f1=np.array([0,b*e])
	f2=np.array([0,-b*e])
	print(e)
	
l1=abs(s@f1.T)
l2=abs(s@f2.T)
d=l1*l2/(LA.norm(s))**2
print(d)		

len=100
theta=np.linspace(0,2*np.pi,len)
y = np.zeros((2,len))
y[0,:] = a*np.cos(theta)

y[1,:] = b*np.sin(theta)

n= u + V.T@P
m=omat@n
t=line_dir_pt(m,P,-0.25,0.25)
l1=line_dir_pt(n,f1,-0.1,0.1)
l2=line_dir_pt(n,f2,-0.2,0.2)

plt.plot(t[0,:],t[1,:],label='Tangent')
plt.plot(y[0,:],y[1,:],label='Given Ellipse')
plt.plot(l1[0,:],l1[1,:],label='Parabola_1')
plt.plot(l2[0,:],l2[1,:],label='Parabola_1')

plt.plot(u[0], u[1], 'o')
plt.text(u[0] * (1 + 0.5), u[1] * (1 - 0.1) , 'C')


plt.plot(P[0], P[1], 'o')
plt.text(P[0] * (1 + 0.5), P[1] * (1 - 0.1) , 'O')

plt.plot(f1[0], f1[1], 'o')
plt.text(f1[0] * (1 + 0.5), f1[1] * (1 - 0.1) , 'F\N{SUBSCRIPT ONE}')
plt.plot(f2[0], f2[1], 'o')
plt.text(f2[0] * (1 + 0.5), f2[1] * (1 - 0.1) , 'F\N{SUBSCRIPT TWO}')

ax.plot()
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
plt.show()

