from sklearn.linear_model import LogisticRegression

my_tree = LogisticRegression()

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

print(my_tree)

#out = pred.predict([[19]])

# print(pred.predict([[24, 70]]))
# print(pred.predict([[84, 70]]))
# print(pred.predict([[90, 90]]))
# print(pred.predict([[27, 30]]))
# print(pred.predict([[34, 80]]))
# print(pred.predict([[60, 60]]))
# print(pred.predict([[1, 1]]))
# print(pred.predict([[90, 90]]))
# print(pred.predict([[1, 90]]))
k=[[90, 1]]
print(pred.predict(k))

print(type(data),type(result),type(k))