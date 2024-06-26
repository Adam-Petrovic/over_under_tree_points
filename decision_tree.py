"""Over Under Tree Points: decision_tree.py

Module Description
==================
This module deals decision trees, or what RandomForest makes a lot of copies of.

In order to implement this code, I have followed this Tutorial, as discussed with Prof. Sharmin (she says it's okay):
    - https://www.youtube.com/watch?v=NxEHSAfFlK8

Copyright and Usage Information
===============================

This file is provided solely for the personal and use of Adam.
All forms of distribution of this code, whether as given or with any changes, are
expressly prohibited, unless permitted by either Adam.
For more information on copyright on this material,
please message Adam at adam.petrovic2005@gmail.com

This file is Copyright (c) 2024 Adam Petrovic
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np


@dataclass
class _Split:
    left: Optional[Node]
    right: Optional[Node]


class Node:
    """
    Instance Attributes:
    - feature: whichever feature this is from/for
    - threshold: the splitting criteria (if the feature is determined to be scored less than the threshold, go left.
     Else, right)
    - split: represents the right and left branches of the Node
    - value: Is None if it represents a split, else it is a leaf Node which stores the sorted data
    """

    feature: int
    threshold: float
    split: _Split
    value: Optional[Any]

    def __init__(self, feature: int = None, threshold: float = None, split: _Split = None,
                 value: Any = None) -> None:
        self.feature = feature
        self.threshold = threshold
        self.split = split
        self.value = value

    def is_leaf_node(self) -> bool:
        """
        Returns True if self is a leaf
        """
        return self.value is not None


class DecisionTree:
    """
    The Class which represents the Decision Tree, Populated of Nodes
    Instance Attributes:
    - min_samples_split: The minimum samples we will split, regardless of overfitting
    - max_depth: The maximum height of our tree
    - n_features: The number of features we take into consideration for splitting
    - root: The base split for our data

    Representation Invariants:
     - min_samples_split > 1
     - max_depth > 0
     - n_features > 0
    """

    min_samples_split: int
    max_depth: int
    n_features: int
    root: Optional[Node]

    def __init__(self, min_samples_split: int = 2, max_depth: int = 100, n_features: int = None) -> None:
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, dataset: np.ndarray, targets: np.ndarray) -> None:
        """
        Begins growing our trees based on our dataset and our target

        Parameters:
         - dataset: Dataset
         - targets: Targets

        Preconditions:
         - len(dataset) == len(targets)
         - len(dataset) > 0
         - len(targets) > 0
        """
        self.n_features = dataset.shape[1] if not self.n_features else min(dataset.shape[1], self.n_features)
        self.root = self._grow_tree(dataset, targets)

    def _grow_tree(self, dataset: np.ndarray, targets: np.ndarray, depth: int = 0) -> Node:
        """
        Recursive method to grow our tree- it stops once our stopping criteria is met.

        Parameters:
         - dataset: Dataset
         - targets: Targets
         - depth: The depth of this Node
         - return: Returns either a leaf (the game) or the next splitting node

        Preconditions:
         - len(dataset) == len(targets)
         - len(dataset) > 0
         - len(targets) > 0
         - self.max_depth + 1 > depth >= 0
        """
        n_samples, n_feats = dataset.shape
        n_labels = len(np.unique(targets))

        # check the stopping criteria
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_target(targets)
            return Node(feature=None, threshold=None, split=_Split(left=None, right=None), value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # find the best split
        best_feature, best_thresh = self._best_split(dataset, targets, feat_idxs)

        # create child nodes
        left_idxs, right_idxs = self._split(dataset[0:, best_feature], best_thresh)
        left = self._grow_tree(dataset[left_idxs, 0:], targets[left_idxs], depth + 1)
        right = self._grow_tree(dataset[right_idxs, 0:], targets[right_idxs], depth + 1)
        return Node(feature=best_feature, threshold=best_thresh, split=_Split(left=left, right=right), value=None)

    def _best_split(self, dataset: np.ndarray, target: np.ndarray, feat_idxs: list[int]) -> tuple[int, float]:
        """
        Calculates the best split based on the best information gain and returns our index spliting_index
        and the Node's split_threshold.


        Parameters:
         - dataset: Dataset
         - target: Targets
         - feat_idxs: Indexes of our features

        Preconditions:
         - len(dataset) == len(targets)
         - len(dataset) > 0
         - len(targets) > 0
         - feat_idxs > 0
        """
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            dataset_column = dataset[0:, feat_idx]
            thresholds = np.unique(dataset_column)

            for thr in thresholds:
                # calculate the information gain
                gain = self._information_gain(target, dataset_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _information_gain(self, targets: np.ndarray, dataset_column: np.ndarray, threshold: float) -> float:
        """
        Calculates the information gain for this split, and returns it.

        Parameters:
         - targets: Targets
         - dataset_column: Column in our data X based on our feature index
         - threshold: The splitting criteria

         Preconditions:
          - len(targets) > 0
        """
        # parent entropy
        parent_entropy = self._entropy(targets)

        # create children
        left_idxs, right_idxs = self._split(dataset_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # calculate the weighted avg. entropy of children
        n = len(targets)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(targets[left_idxs]), self._entropy(targets[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, dataset_column: np.ndarray, split_thresh: float) -> tuple:
        """
        Splits our node, and then it returns the indexes to split.

        Parameters:
         - dataset_column: Our column in our dataset (X)
         - split_thresh: Determines whether to go left or right

         Preconditions:
          - len(dataset_column) > 0
        """
        left_idxs = np.argwhere(dataset_column <= split_thresh).flatten()
        right_idxs = np.argwhere(dataset_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, targets: np.ndarray) -> float:
        """
        Standard Entropy Formula for calculating splits

        Parameters:
         - targets:

        Preconditions:
         - len(targets) > 0
        """
        hist = np.bincount(targets)
        ps = hist / len(targets)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _most_common_target(self, targets: np.ndarray) -> Any:
        """
        Counts and returns 1 or 0 depending on which appears most

        Parameters:
         - targets: Targets

         Preconditions:
          - len(targets) > 0
         """
        counter = Counter(targets)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, games: np.ndarray) -> np.ndarray:
        """
        Our main predictor function, which will go down our tree similar to a BinarySearchTree to find where our
        game lies

        Parameters:
         - games: Games we want to predict over/under for

         Preconditions:
          - len(games) > 0
        """
        return np.array([self._traverse_tree(game, self.root) for game in games])

    def _traverse_tree(self, game_entry: list, node: Node) -> Any:
        """
        How we find where our game lies in our DecisionTree, implemented recursively- Similar to BinarySearchTree
        traversal were if the feature is ranked less than the threshold, then the threshold, then it goes left. Else,
        it goes right.

        Returns the target/label where the game belongs.

        Parameters:
         - game_entry: Our game in question
         - node: Our node

         Preconditions:
          - len(game_entry) == 24
        """
        if node.is_leaf_node():
            return node.value

        if game_entry[node.feature] <= node.threshold:
            return self._traverse_tree(game_entry, node.split.left)
        return self._traverse_tree(game_entry, node.split.right)


if __name__ == "__main__":
    import python_ta

    python_ta.check_all(config={
        'extra-imports': ['__future__', 'typing', 'numpy', 'collections', '',
                          'dataclasses'],  # the names (strs) of imported modules
        'allowed-io': [],  # the names (strs) of functions that call print/open/input
        'max-line-length': 120
    })
