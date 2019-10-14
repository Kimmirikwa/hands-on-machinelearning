from sklearn import datasets

iris = datasets.load_iris()

# get the training data
X = iris['data'][:, (2, 3)]  # petal length and petal width
y = iris['target']