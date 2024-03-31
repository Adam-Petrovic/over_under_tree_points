"""
Random Forest Tree:
Works by taking a subset of parameters and making many DecisionTrees
"""
from collections import Counter
import numpy as np
from decision_tree import DecisionTree


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

    n_trees: int
    max_depth: int
    min_samples_split: int
    n_features: int
    trees: list

    def __init__(self, n_trees: int = 10, max_depth: int = 10,
                 min_samples_split: int = 2, n_feature: int = None) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_feature
        self.trees = []

    def fit(self, dataset: np.ndarray, targets: np.ndarray) -> None:
        """
        Fits our dataset X with our target Y (trains our model)
        :param dataset: Dataset
        :param targets: Targets
        """
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                n_features=self.n_features)
            dataset_sample, target_sample = self._bootstrap_samples(dataset, targets)
            tree.fit(dataset_sample, target_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, dataset: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_samples = dataset.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return dataset[idxs], targets[idxs]

    def _most_common_label(self, target: np.ndarray) -> int:
        counter = Counter(target)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, dataset: list[np.ndarray]) -> np.ndarray:
        """
        Handles our predictions
        :param dataset:
        :return:
        """
        predictions = np.array([tree.predict(dataset) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions


if __name__ == "__main__":
    import python_ta

    python_ta.check_all(config={
        'max-line-length': 120
    })
