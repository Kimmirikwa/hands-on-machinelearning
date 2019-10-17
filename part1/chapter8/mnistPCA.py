import time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import IncrementalPCA
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

# pca to ensure the selected dimensions retain 95% of the original variance
n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
	inc_pca.partial_fit(X_batch)

X_train_reduced = []
for X_batch in np.array_split(X_train, n_batches):
	X_train_reduced.extend(inc_pca.transform(X_batch))
log_clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42)
start = time.time()
log_clf.fit(X_train_reduced, y_train)
end = time.time()

print("Training with PCA took: ", end - start)

# we need to transform X_test to be used in making predictions
X_test_reduced = inc_pca.transform(X_test)
y_pred = log_clf.predict(X_test_reduced)
print("the accuracy with PCA: ", accuracy_score(y_test, y_pred))  # 0.9206
