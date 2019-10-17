import time
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mnist = fetch_openml('mnist_784')

X = mnist['data']
y = mnist['target']

X_train = X[:60000]
X_test = X[60000:]
y_train = y[:60000]
y_test = y[60000:]

log_clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42)
start = time.time()
log_clf.fit(X_train, y_train)
end = time.time()

print("Training took: ", end - start)  # 23.43

y_pred = log_clf.predict(X_test)
print("the accuracy without PCA: ", accuracy_score(y_test, y_pred))  # 0.9255