
# NumPy: a library for vector and  matrix calculations
import numpy as np

#nD-array
#1D-array = vector
#2D-array = 2D-matrix = a collection of 1D-arrays
#3D-array = 3D-matrix = a collection of 2D-arrays
#4D-array = 4D-matrix = a collection of 3D-arrays

#1D-array = vector
v=np.array([4,5,1,0,-2])  #v=[v[0],v[1],v[2],v[3],v[4]]
#NOTE: indexing from 0 !!
print(v)

v.shape #(5,)
v[0] #element 0 = first element
v[-1] #last element
v[0:3] #elements 0,1,2 NOTE: element v[3] is NOT included !
print(v[0:3])
v[1:-1] #elements 1,2,...,second to last
v[1:] #elements 1,2,...,last
v[-3:] #elements a[2],a[3],a[4], third to last, second to last ,...,last
ind=[1,2,4]
v[ind] #elements 1,2,4

#indexing with true/false-condotions
v=np.array([4,5,1,0,-2])  #v=[v[0],v[1],v[2],v[3],v[4]]
print(v>3) #[true,true,false,false,false]
print(v[v>3]) #elements which are > 3
v[(v>3)|(v<0)] #elements which are >3 or <0

y=np.array([0,1,0,1,1])
y==1
v[y==1] #elements 1,3,4

#&  and
#|  or
#!= not equal

np.max(v) #max-element
np.argmax(v) #corresponding index
np.min(v) #min-element
np.argmin(v) #corresponding index
np.sum(v) #sum of elements
np.sort(v) #smallest -> largest
-np.sort(-v) #largest -> smallest
np.argsort(v) #indices, smallest->largest
np.argsort(-v) #indices, largest->smallest

#constant 1D-arrays
n=10
np.zeros(n)
np.ones(n)

#1D-arrays with fixed step
v=np.arange(0,10,2) #=[0,2,4,6,8], (start,end,step), end IS NOT included
print(v,'\n')
w=np.linspace(0,1,11) #=[0,0.1,0.2,...,0.9,1] (start,end,number of elements), end IS included
print(w)

#2D-array = matrix
A=np.array([[1,3,4,0],[-1,5,8,6],[5,3,0,1]])
#rows [1,3,4,0],[-1,5,8,6],[5,3,0,1]
A

m,n=A.shape
#m=number of rows
#n=number of columns
r=0
s=1
A[r,s] #element at row r, column s (column = sarake in Finnish)

A[r,:] #or A[r], row r, 1D-array !!
A[:,s] #column s, 1D-array !!

A[1:3,0:3] #rows 1 and 2, columns 0,1 and 2
A[[0,2],:] #rows 0 and 2
A[:,[0,2,3]] #columns 0,2,3

#indexing with true/false-conditions
A=np.array([[1,3,4,0,2],[-1,5,8,6,1],[5,3,0,1,-2],[0,4,2,8,6]])


A[A>0] #elements which are>0, 1D-array

y=np.array([1,0,1,1,1])
y==1
A[:,y==1] #columns 0,2,3,4

z=np.array([0,1,0,1])
A[z==1,:] #rows 1,3

#and: &
#or:  |
#not equal:  !=

#constant matrices
m=3
n=4
print(np.zeros((m,n)),'\n') #(m,n)-matrix of zeros
print(np.ones((m,n)))  #(m,n)-matrix of ones

#creating an array using bigger pieces

a=np.array([1,2,3])
b=np.array([4,5,6])

abh=np.hstack((a,b)) #stack horizontally
print(abh,'\n')

abv=np.vstack((a,b)) #stack vertically
print(abv,'\n')

#a and b to rows of a matrix abr
abr=np.zeros((2,3))
abr[0,:]=a
abr[1,:]=b
print(abr,'\n')

#a and b as columns of a matrix abs
abc=np.zeros((3,2))
abc[:,0]=a
abc[:,1]=b
print(abc,'\n')

A=np.array([[1,2],
            [3,4]]) #(2,2)

B=np.array([[5,1,2],
            [6,7,3]]) #(2,3)


C=np.array([[6,1],
            [8,0],
            [3,2]]) #(3,2)

ABh=np.hstack((A,B)) #stack horizontally
print(ABh,'\n')
ACv=np.vstack((A,C)) #stack vertically
print(ACv,'\n')


#or
ABh=np.zeros((2,5))
ABh[:,:2]=A #columns 0 and 1
ABh[:,2:]=B #columns 2...

ACv=np.zeros((5,2))
ACv[:2,:]=A #rows 0 and 1
ACv[2:,:]=C #rows 2...

#max, min, sum
A=np.array([[1,3,4,0,2],
            [-1,5,8,6,1],
            [5,3,0,1,-2],
            [0,4,2,8,6]])

np.max(A) #max of the elements
np.min(A) #min of the elements
np.max(A,axis=0) #maxs of the columns, 1D-array, (n,)
np.max(A,axis=0,keepdims=True) #,maxs of the columns, row vector, (1,n)
np.max(A,axis=1) #maxs of the rows, 1D-array, (m,)
np.max(A,axis=1, keepdims=True) #maxs of the rows, column vector, (m,1)

np.sum(A)  #sum of elements
np.sum(A,axis=0) #sums of columns
np.sum(A,axis=0,keepdims=True)
np.sum(A,axis=1) #sums of rows
np.sum(A,axis=1,keepdims=True)

#elementwise operations
a=np.array([1,2,3])
b=np.array([4,5,6])

5*a
a/5
a+5
5/a
a**2 #to power 2

#for arrays of same shape
a+b
a-b
a*b
a/b

#transpose, rows <-> columns
A=np.array([[1,2,3],
            [4,5,6]])
print(A.T,'\n')

np.array([1,2,3]).T #no effect on 1D-arrays

#matrix multiplication @
A=np.array([[1,2],
            [4,7]])

B=np.array([[3,5],
            [-4,2]])

print(A@B,'\n')

#inverse matrix
Ainv=np.linalg.inv(A)
print(Ainv,'\n')

A@Ainv # = unit matrix

#Matplotlib: plotting library
import matplotlib.pyplot as plt

# If x and y are 1D-arrays (or Python-lists [a,b,c,...]) of equal length,
# then plt.plot(x,y)connects the 2D-points [x[0],y[0]] , [x[1],y[1]], [x[2],y[2]], … with line
# i.e  x is the horizontal coordinate, y is vertical
# plt.plot(y) -> horizontal coordinates 0,1,2,..., vertical y[0],y[1],y[2],...

n=11
x=np.linspace(-1,1,n)
y1=np.sin(np.pi*x)
y2=x**2

plt.figure(figsize=(8,5)) #(width,height)
plt.plot(x,y1,'r',linewidth=2,label='y1')
plt.plot(x,y2,'g-o',markersize=10,label='y2')
plt.plot([-1,0,1],[0.5,-0.4,1.4],'k',label='a line')
plt.plot(x[0],y1[0],'ms',label='a point')
plt.title('main title',fontsize=12)
plt.xlabel('title for horizontal axis',fontsize=14)
plt.ylabel('title for vertical axis')
plt.grid() #background grid
plt.legend(fontsize=12) #shows the labels
plt.xlim(-1.1,1.1) #limits for horizontal axis
plt.xticks(np.linspace(-1,1,11)) #grid-lines in horizontal axis
plt.ylim(-1,2) #limits for vertical axis
plt.show() #shows the picture

#%% bar-plot

x=[1,2,3,4]
y=[5,2,6,3]
plt.figure(figsize=(7,5))
plt.bar(x,y,facecolor='c',edgecolor='k',lw=1,zorder=2) #zorder=2->grid-viivat tolppien takana
plt.grid()
plt.xticks(x)
plt.show()

#%% load iris.txt to colab
#luetaan tiedosto iris.txt (150,5)-matriisiksi data
data=np.loadtxt('iris.txt')
m,n=data.shape
print(m)
print(n)
X=data[:,0:4] #sarakkeet 0-3, koordinaatit x0,x1,x2,x3
y=data[:,4] #sarake 4, luokka 1,2 tai 3

#picture of the data points using coordinates (= columns of  X) i and j
i=0
j=1

plt.figure(figsize=(5,5))
plt.plot(X[y==0,i],X[y==0,j],'r.',label='y=0') #points from class y=0
plt.plot(X[y==1,i],X[y==1,j],'g.',label='y=1') #points from class y=1
plt.plot(X[y==2,i],X[y==2,j],'b.',label='y=2') #points from class y=2
plt.grid()
plt.legend()
plt.xlabel('$x_i$')
plt.ylabel('$x_j$',rotation=0)
plt.show()

#values of column k
plt.figure(figsize=(8,5))
k=0
plt.plot(X[:,k],'b.',markersize=12)
plt.grid()
plt.title('column ' + str(k))
plt.xlabel('element number',fontsize=14)
plt.show()

#distribution of values in column k (histogram)
plt.figure(figsize=(8,5))
#bins = number of intervals
plt.hist(X[:,k],bins=30,edgecolor='k',zorder=2)
plt.grid()
plt.xlabel('value',fontsize=14)
plt.ylabel('values on interval',fontsize=14)
plt.title('column ' + str(k))
plt.show()