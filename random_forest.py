"""
Over Under Tree Points: random_forest.py

Module Description
==================
This module deals with the Random Forest Implementation. It works by taking subsets of the parameters passed in from
train.py, and making many DecisionTree based on these to prevent over fitting.

In order to implement this code, I have followed this Tutorial, as discussed with Prof. Sharmin (she says it's okay):
    - https://www.youtube.com/watch?v=kFwe2ZZU7yw

Copyright and Usage Information
===============================

This file is provided solely for the personal and use of Adam.
All forms of distribution of this code, whether as given or with any changes, are
expressly prohibited, unless permitted by Adam.
For more information on copyright on this material,
please message Adam at adam.petrovic2005@gmail.com

This file is Copyright (c) 2024 Adam Petrovic
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

     Representation Invariants:
      - n_trees > 0
      - max_depth > 0
      - n_features > 0
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
         - dataset: Dataset
         - targets: Targets

        Preconditions:
         - len(dataset) > 0
         - len(targets) > 0
         - len(dataset) == len(targets)
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
        """
        Boostraps our samples to then use for our Decision Trees
        Parameters:
         - dataset: Dataset
         - targets: Targets

        Preconditions:
         - len(dataset) > 0
         - len(targets) > 0
         - len(dataset) == len(targets)
        """
        n_samples = dataset.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return dataset[idxs], targets[idxs]

    def _most_common_target(self, targets: np.ndarray) -> int:
        """
        Returns the most commmon target amongst the games in the Node (1 or 0)
        Parameters:
         - targets: Targets

        Preconditions:
         - len(targets) > 0
        """
        counter = Counter(targets)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, betting_games: list) -> np.ndarray:
        """
        Takes our dataset of games to predict, then runs these to the multiple self.trees.
        It then majority votes the outcome from these trees

        Parameter:
          - betting_games: The list of games we want to figure out if they're over or under our bet score

        Preconditions:
         - len(betting_games) > 0
        """
        predictions = np.array([tree.predict(betting_games) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_target(pred) for pred in tree_preds])
        return predictions


if __name__ == "__main__":
    import python_ta

    python_ta.check_all(config={
        'extra-imports': ['numpy', 'collections', 'decision_tree'],  # the names (strs) of imported modules
        'allowed-io': [],  # the names (strs) of functions that call print/open/input
        'max-line-length': 120
    })
