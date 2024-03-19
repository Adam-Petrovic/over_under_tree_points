from decision_tree import DecisionTree
import numpy as np
from collections import Counter


class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_feature=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_feature
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                n_features=self.n_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions


print(
    'PHI	PHI @ NYK	02/05/2023	L	240	97	30	74	40.5	7	25	28.0	30	36	83.3	8	36	44	23	4	4	12	23	-11 SAC	SAC @ NOP	02/05/2023	L	240	104	35	82	42.7	11	42	26.2	23	27	85.2	11	27	38	23	5	5	15	22	-32 NOP	NOP vs. SAC	02/05/2023	W	240	136	48	85	56.5	14	26	53.8	26	34	76.5	12	36	48	26	11	3	11	23	32 MIN	MIN vs. DEN	02/05/2023	W	240	128	48	87	55.2	10	27	37.0	22	30	73.3	5	29	34	33	7	3	11	23	30'.split(
        ' '))
