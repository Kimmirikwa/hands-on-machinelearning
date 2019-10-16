from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

mnist = fetch_openml('mnist_784')

X = mnist['data']
y = mnist['target']

X_train_validation, X_test, y_train_validation, y_test = train_test_split(X, y, test_size=10000, random_state=42)
X_train, X_validation, y_train, y_validation = train_test_split(X_train_validation, y, test_size=10000, random_state=42)