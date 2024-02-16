from sklearn import tree
import sys
print(sys.path)

my_tree = tree.DecisionTreeClassifier()

data = [
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
]

result = [[0], [1], [0], [1], [1], [0], [1], [1], [1], [0], [0]]

pred = my_tree.fit(data, result)

#out = pred.predict([[19]])

print(pred.predict([[24, 70]]))
print(pred.predict([[84, 70]]))
print(pred.predict([[90, 90]]))