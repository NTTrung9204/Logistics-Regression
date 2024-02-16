import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
import warnings 
warnings.filterwarnings( "ignore" )

class LinearRegression:
    def __init__(self, X, Y, epsilon = 0.001, loop = 100000):
        self.X = X
        self.Y = Y
        self.epsilon = epsilon
        self.loop = loop
        self.ineration = []
        self.cost = []
        self.length = len(Y)

    def Cost(self, y_predict):
        return -sum((self.Y*np.log(y_predict)+(1-self.Y)*np.log(1-y_predict)))

    def predict(self, w, bias, x):
        return self.sigmod(x @ (w.T) + bias)
    
    def sigmod(self, x):
        return 1/(1+np.exp(-x))
    
    def fit(self):
        w = np.random.rand(1, len(self.X[0]))
        v = 0
        bias = np.random.rand(1)
        for i in range(self.loop):
            sys.stdout.write("\rIneration: {}/{}".format(i+1, self.loop))
            y_predict = self.predict(w - 0.9 * v, bias, self.X)
            v = 0.9 * v + self.epsilon * (((self.X.T) @ (y_predict - self.Y)).T)
            bias=sum(bias - self.epsilon/self.length*(y_predict - self.Y))/self.length
            w = w - v
            self.ineration.append(i)
            self.cost.append(self.Cost(y_predict)[0])
        
        return w[0], bias[0]
    
    def showCost(self):
        print(self.cost[-1])
        plt.plot(self.ineration, self.cost)
        plt.show()


data = pd.read_csv('DataSet/diabetes.csv')

x = data.iloc[:, :8].values
y = data.iloc[:, -1].values.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split( x, y, test_size = 1/3, random_state = 0 ) 

LR = LinearRegression(X_train, Y_train, 0.0000001, 5000)
w, bias = LR.fit()
print()
LR.showCost()

total = 0
list = [1 if LR.predict(w, bias, X_test[i]) > 0.5 else 0 for i in range(len(X_test))]
for i in range(len(X_test)):
    if list[i] == Y_test[i] : total+=1

print(total/len(X_test))
print(list)
print(x[159])
print(LR.predict(w, bias, x[159]))
print(X_test)
# my_tree = LogisticRegression()
# pred = my_tree.fit(X_train, Y_train)

# total = 0
# list = pred.predict(X_test)
# print(list)
# for i in range(len(X_test)):
#     if list[i] == Y_test[i] : total+=1

# print(total/len(X_test))
# print(list)
# print(X_test[-1])
# print(X_test[-2])
# print(X_test[-3])