import numpy as np

def SM(z):
    z=(z+1000)/500
    ez=np.exp(z)
    res=[[sum(ez[0])]]
    for i in range(1,len(ez)):
        res=np.vstack((res,[sum(ez[i])]))
    return ez/res

x=np.array([
    [18, 62],
    [34, 85],
    [53, 64],
    [30, 82],
    [12, 56],
    [55, 67],
    [70, 70],
    [45, 99],
    [16, 110],
    [66, 40],
    [26, 70]
])

y=np.array([
    [1, 0],
    [0, 1],
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0],
    [0, 1],
    [0, 1],
    [0, 1],
    [1, 0],
    [1, 0]
])

w=np.array([
    [10, 3],
    [-70, 10]
])

bias=np.array([
    [2.3, -3]
])
#x: 11 x 2
#w: 2 x 2
#bias: 1x2
#z: 11 x 2
k=0.005
for i in range(50000):
    z=x@w+bias
    # print(z)
    pre=SM(z)
    # print(pre)
    w=w-k*(x.T@(pre-y))
    bias=bias-k*(sum(pre-y))
    if i%5000==0:
        cost=0
        for m in range(11):
            for n in range(2):
                cost-=y[m][n]*np.log(pre[m][n])
        print(cost/len(y))
# zzz=[[1,13,90]]@w
print(SM([[13,90]]@w+bias))
print(SM([[24,60]]@w+bias))
print(SM([[55,130]]@w+bias))
print(SM([[33,50]]@w+bias))
print(SM([[10,100]]@w+bias))
print(SM([[26,50]]@w+bias))
print(SM([[90,130]]@w+bias))
print(SM([[11,120]]@w+bias))