import time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

def plot_digits(X, y, min_distance=0.05, images=None, figsize=(13, 10)):
    # Let's scale the input features so that they range from 0 to 1
    X_normalized = MinMaxScaler().fit_transform(X)
    # Now we create the list of coordinates of the digits plotted so far.
    # We pretend that one is already plotted far away at the start, to
    # avoid `if` statements in the loop below
    neighbors = np.array([[10., 10.]])
    # The rest should be self-explanatory
    plt.figure(figsize=figsize)
    cmap = cm.get_cmap("jet")
    digits = np.unique(y)
    for digit in digits:
    	plt.scatter(X_normalized[y == digit, 0], X_normalized[y == digit, 1], c=[cmap(int(digit) / 9)])
    plt.axis("off")
    ax = plt.gcf().gca()  # get current axes in current figure
    for index, image_coord in enumerate(X_normalized):
        closest_distance = np.linalg.norm(np.array(neighbors) - image_coord, axis=1).min()
        if closest_distance > min_distance:
            neighbors = np.r_[neighbors, [image_coord]]
            if images is None:
                plt.text(image_coord[0], image_coord[1], str(int(y[index])),
                         color=cmap(int(y[index]) / 9), fontdict={"weight": "bold", "size": 16})
            else:
                image = images[index].reshape(28, 28)
                imagebox = AnnotationBbox(OffsetImage(image, cmap="binary"), image_coord)
                ax.add_artist(imagebox)

def plot_2_dims(transformerClass, X, y):
	transformer = transformerClass(n_components=2, random_state=42)
	t0 = time.time()
	X_reduced = transformer.fit_transform(X)
	t1 = time.time()
	print("{} took {:.1f}s.".format(transformer.__class__.__name__, t1 - t0))
	plot_digits(X_reduced, y, images=X, figsize=(35, 25))
	plt.show()

mnist = fetch_openml("mnist_784")

X = mnist['data']
y = mnist['target']

# randomly selecting 10000 digits to speed up dimension reduction
m = 10000
random_indices = np.random.permutation(60000)[:m]

X = X[random_indices]
y = y[random_indices]

plot_2_dims(PCA, X[:10000], y[:10000])
plot_2_dims(LocallyLinearEmbedding, X[:2000], y[:2000])