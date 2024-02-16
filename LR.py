import numpy as np
import sys
x=np.array([
    [1, 18, 62],
    [1, 34, 85],
    [1, 53, 64],
    [1, 30, 82],
    [1, 12, 56],
    [1, 55, 67],
    [1, 70, 70],
    [1, 45, 99],
    [1, 16, 110],
    [1, 66, 40],
    [1, 26, 70]
])

y=np.array([
    [0],
    [1],
    [0],
    [1],
    [1],
    [0],
    [1],
    [1],
    [1],
    [0],
    [0]
])

def sigmod(x):
    return 1/(1+np.exp(-x))

w=np.array([
    [0.1],
    [-2.1],
    [3.1]
])


k=0.0001
inter=5000
cost=[0]
cost1=[0]
for i in range(1,inter+1):
    pred=sigmod(np.dot(x,w))
    w=w-k*np.dot(x.T,pred-y)
    if i%(inter//50)==0:
        cost=0
        Y=sigmod(np.dot(x[j],w))
        cost1 = -sum((y*np.log(Y)+(1-y)*np.log(1-Y)))
        for j in range(len(y)):
            Y=sigmod(np.dot(x[j],w))
            cost+=-(y[j]*np.log(Y)+(1-y[j])*np.log(1-Y))
    sys.stdout.write("\rLoad data: {}/{}. Cost : {} ".format(i,inter,cost[0]))
        
# print(sigmod(np.dot([1,13,90],w)))
# print(sigmod(np.dot([1,24,60],w)))
# print(sigmod(np.dot([1,55,130],w)))
# print(sigmod(np.dot([1,33,50],w)))
# print(sigmod(np.dot([1,10,100],w)))
# print(sigmod(np.dot([1,26,50],w)))
# print(sigmod(np.dot([1,90,130],w)))
# print(sigmod(np.dot([1,11,120],w)))


