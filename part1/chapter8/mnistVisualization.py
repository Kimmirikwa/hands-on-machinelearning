import time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding, MDS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
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

def plot_2_dims(transformer, X, y, is_pipeline=False, fit_with_y=False):
	t0 = time.time()
	if fit_with_y:
		X_reduced = transformer.fit_transform(X, y)
	else:
		X_reduced = transformer.fit_transform(X)
	t1 = time.time()
	reducer_name = "+".join([type(step[1]).__name__ for step in transformer.get_params()['steps']]) if is_pipeline else type(transformer).__name__
	print("{} took {:.1f}s (on {} MNIST images)".format(reducer_name, t1 - t0, len(X)))
	plot_digits(X_reduced, y, images=X, figsize=(35, 25))
	plt.show()

mnist = fetch_openml("mnist_784")

X = mnist['data']
y = mnist['target']

random_indices = np.random.permutation(60000)

X = X[random_indices]
y = y[random_indices]

plot_2_dims(PCA(n_components=2, random_state=42), X[:2000], y[:2000])  # 0.1s

plot_2_dims(LocallyLinearEmbedding(n_components=2, random_state=42), X[:2000], y[:2000])  # 12.6s

plot_2_dims(MDS(n_components=2, random_state=42), X[:2000], y[:2000])  # 365.3s

plot_2_dims(LinearDiscriminantAnalysis(n_components=2), X[:2000], y[:2000], fit_with_y=True)  # 1.6s

plot_2_dims(TSNE(n_components=2, random_state=42), X[:2000], y[:2000])  # 45.2s

pca_lle = Pipeline([
	("pca", PCA(n_components=2, random_state=42)),
	("lle", LocallyLinearEmbedding(n_components=2, random_state=42))])
plot_2_dims(pca_lle, X[:2000], y[:2000], is_pipeline=True)  # 0.9s

pca_mds = Pipeline([
	("pca", PCA(n_components=2, random_state=42)),
	("mds", MDS(n_components=2, random_state=42))])
plot_2_dims(pca_mds, X[:2000], y[:2000], is_pipeline=True)  # 174.2s

pca_lda = Pipeline([
	("pca", PCA(n_components=2, random_state=42)),
	("lda", LinearDiscriminantAnalysis(n_components=2))])
plot_2_dims(pca_lda, X[:2000], y[:2000], is_pipeline=True, fit_with_y=True)  # 0.1s

pca_tsne = Pipeline([
	("pca", PCA(n_components=2, random_state=42)),
	("tsne", TSNE(n_components=2, random_state=42))])
plot_2_dims(pca_tsne, X[:2000], y[:2000], is_pipeline=True)  # 17.6s