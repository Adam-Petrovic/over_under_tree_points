
from decision_tree import DecisionTree
import numpy as np
from collections import Counter


class RandomForest:
    """
    Our Random Forest Class. Random Forest works by taking a subset of features and repeating Decision Trees over and
    over again to determine the best model.

    Instance Attributes:
     - n_trees: Number of trees in our 'Forest'
     - max_depth: The maximum depth of our Forest
     - min_samples: The mimnumum samples to make the next split
     - n_features: The number of features to consider
     - trees: Keeps track of the trees
    """
    def __init__(self, n_trees=20, max_depth=10, min_samples_split=2, n_feature=None) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_feature
        self.trees = []

    def fit(self, X, y) -> None:
        """
        Fits our dataset X with our target Y (trains our model)
        :param X: Dataset
        :param y: Targets
        """
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                n_features=self.n_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y) -> tuple[list, list]:
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X) -> np.ndarray:
        """
        Handles our predictions
        :param X:
        :return:
        """
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions
