import numpy as np
import sys
# x=np.array([
#     [1, 1, 1, 1, 1, 1, 1],
#     [1, 2, 0, 1, 1, 0, 2],
#     [1, 3, 1, 3, 3, 1, 3],
#     [1, 0, 1, 0, 0, 1, 0],
#     [1, 2, 3, 4, 4, 3, 2],
#     [1, 1, 0, 1, 8, 1, 5],
#     [1, 1, 8, 1, 1, 6, 6],
#     [1, 9, 8, 1, 0, 1, 0],
#     [1, 9, 0, 7, 1, 2, 8],
#     [1, 5, 3, 8, 2, 9, 0]
# ])
x=np.array([
    [1, 2],
    [1, 4],
    [1, 6],
    [1, 8],
    [1, 10],
    [1, 9],
    [1, 7],
    [1, 5],
    [1, 3],
    [1, 1]
])
y=np.array([
    [1],
    [1],
    [1],
    [1],
    [1],
    [0],
    [0],
    [0],
    [0],
    [0]
])

def sigmod(x):
    return 1/(1+np.exp(-x))

# w=np.array([
#     [4],
#     [0.001],
#     [0.001],
#     [0.001],
#     [-0.001],
#     [-0.001],
#     [-0.001]
# ])

w=np.array([
    [4],
    [1]
])


k=0.0001
inter=200000
cost=[0]
for i in range(1,inter+1):
    pred=sigmod(np.dot(x,w))
    w=w-k*np.dot(x.T,pred-y)
    if i%(inter//50)==0:
        cost=0
        for j in range(len(y)):
            Y=sigmod(np.dot(x[j],w))
            cost+=-(y[j]*np.log(Y)+(1-y[j])*np.log(1-Y))
    sys.stdout.write("\rLoad data: {}/{}. Cost : {}".format(i,inter,cost[0]))

print("\nResult :",sigmod(np.dot([1, 2],w)))


