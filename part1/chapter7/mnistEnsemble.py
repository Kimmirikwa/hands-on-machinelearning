import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

mnist = fetch_openml('mnist_784')

X = mnist['data']
y = mnist['target']

X_train_validation, X_test, y_train_validation, y_test = train_test_split(X, y, test_size=10000, random_state=42)
X_train, X_validation, y_train, y_validation = train_test_split(X_train_validation, y, test_size=10000, random_state=42)

# the classifiers
forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)
extra_trees_clf = ExtraTreesClassifier(n_estimators=10, random_state=42)
svc_clf = LinearSVC(random_state=42)
mlp_clf = MLPClassifier(random_state=42)

estimators = [
	forest_clf,
	extra_trees_clf,
	svc_clf,
	mlp_clf
]

named_estimators = [
	("forest_clf", forest_clf),
	("extra_trees_clf", extra_trees_clf),
	("svc_clf", svc_clf),
	("mlp_clf", mlp_clf)
]

voting_clf = VotingClassifier(named_estimators)
voting_clf.fit(X_train, y_train)
voting_clf.score(X_validation, y_validation)

# by default the voting classifier uses hard voting, to change to soft voting, we simply change the scoring
# no need to train the model again
voting_clf.voting = "soft"
voting_clf.score(X_test, y_test)

# stacking ensemble
X_validation_predictions = np.empty((len(X_validation), len(estimators)), dtype=np.float32)

for index, estimator in enumerate(estimators):
	X_validation_predictions[:, index] = estimator.predict(X_validation)

random_forest_blender = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42)
random_forest_blender.fit(X_validation_predictions, y_validation)  # train on the output of the previous predictions
print(random_forest_blender.oob_score_)

# test on testing set
X_test_predictions = np.empty((len(X_test), len(estimators)), dtype=np.float32)

for index, estimator in enumerate(estimators):
	X_test_predictions[:, index] = estimator.predict(X_test)

y_pred = random_forest_blender.predict(X_test_predictions)
print(accuracy_score(y_test, y_pred))