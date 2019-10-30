import numpy as np
import matplotlib.pyplot as plt
import math
from coeffs import*

e=3/5
d=6
a=6/(2*e)
b=a*((1-e**2)**0.5)
A=np.array([a,0])
B=np.array([0,b])
C=np.array([-a,0])
D=np.array([0,-b])
O=np.array([0,0])

if a>b:
  F1=np.array([d/2,0])
  F2=-F1
  
if a<b:
  F1=np.array([0,d/2])
  F2=-F1
  


x_AB=B-A
x_AC=C-A
x_DB=B-D
x_DC=C-D
p=0.5*np.linalg.norm((np.cross(x_AB,x_AC)))
q=0.5*np.linalg.norm((np.cross(x_DB,x_DC)))
Area=p+q
print(Area)


len=100
theta=np.linspace(0,2*np.pi,len)
y = np.zeros((2,len))
y[0,:] = a*np.cos(theta)

y[1,:] = b*np.sin(theta)

x_AB=line_gen(B,A)
x_BC=line_gen(C,B)
x_CD=line_gen(D,C)
x_DA=line_gen(A,D)
F1F2=line_gen(F1,F2)

plt.plot(y[0,:],y[1,:],label='Ellipse at origin')
plt.plot(x_AB[0,:],x_AB[1,:])
plt.plot(x_BC[0,:],x_BC[1,:])
plt.plot(x_CD[0,:],x_CD[1,:])
plt.plot(x_DA[0,:],x_DA[1,:])
plt.plot(F1F2[0,:],F1F2[1,:])



plt.plot(A[0], A[1], 'o')
plt.text(A[0] * (1 - 0.1), A[1] * (1 + 0.1) , 'A')

plt.plot(B[0], B[1], 'o')
plt.text(B[0] * (1 - 0.1), B[1] * (1 + 0.1) , 'B')

plt.plot(C[0], C[1], 'o')
plt.text(C[0] * (1 - 0.1), C[1] * (1 + 0.1) , 'C')

plt.plot(D[0], D[1], 'o')
plt.text(D[0] * (1 - 0.1), D[1] * (1 + 0.1) , 'D')

plt.plot(O[0], O[1], 'o')
plt.text(O[0] * (1 - 0.1), O[1] * (1 + 0.1) , 'O')

plt.plot(F1[0], F1[1], 'o')
plt.text(F1[0] * (1 - 0.1), F1[1] * (1 + 0.1) , 'F\N{SUBSCRIPT ONE}')

plt.plot(F2[0], F2[1], 'o')
plt.text(F2[0] * (1 - 0.1), F2[1] * (1 + 0.1) , 'F\N{SUBSCRIPT TWO}')



plt.axis('equal')

plt.xlabel('$x$');plt.ylabel('$y$')

plt.legend(loc='best')

plt.grid()
plt.show()

