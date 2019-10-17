import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

mnist = fetch_openml("mnist_784")

X = mnist['data']
y = mnist['target']

# randomly selecting 10000 digits to speed up dimension reduction
m = 10000
random_indices = np.random.permutation(60000)[:m]

X = X[random_indices]
y = y[random_indices]

tsne = TSNE(n_components=2, random_state=42)
print("starting transformation>>>>>>>>>")
X_reduced = tsne.fit_transform(X)