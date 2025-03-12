import numpy as np
from collections import Counter


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right 
        self.value = value
    
    def is_leaf(self):
        return self.value is not None
    


class DecisionTree:
    def __init__(self, min_samples_split=5, max_depth=10):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self.form_tree(X, y, 0)

    def form_tree(self, X, y, depth):
        n_samples = X.shape[0]
        n_labels = len(np.unique(y))

        
        # base case checking
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            return Node(value=self.calculate_leaf_value(y))
        
        # best split
        best_feature, best_threshold = self.best_split(X, y)
        # true or false returned
        # left_indices = X[:, best_feature] < best_threshold
        left_indices, right_indices = self._split(X[:, best_feature], best_threshold)
        

        X_left, y_left = X[left_indices, :], y[left_indices]
        X_right, y_right = X[right_indices, :], y[right_indices]

        # recursive call --> better use split function
        left = self.form_tree(X_left, y_left, depth + 1)
        right = self.form_tree(X_right, y_right, depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)




    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root) for x in X])


    def traverse_tree(self, x, node):
        if(node.is_leaf()):
            return node.value
        if(x[node.feature] < node.threshold):
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)
        
    def calculate_leaf_value(self, y):
        counter = Counter(y)
        # returns the most common "1" elements in the list, as a dictionary
        most_common = counter.most_common(1)[0][0]
        return most_common # return the key value which is our label
    
    def best_split(self, X, y):
        max_gain = -1
        feature_idx, best_threshold = None, None

        for idx in range(X.shape[1]):
            feature = X[:, idx]
            thresholds = np.unique(feature)

            if len(thresholds) != 1:
                thresholds = (thresholds[1:] + thresholds[:-1]) / 2
            
            for threshold in thresholds:
                gain = self.IG_calculation(y, feature, threshold)
                if gain > max_gain:
                    max_gain = gain
                    feature_idx = idx
                    best_threshold = threshold
        return feature_idx, best_threshold


    def IG_calculation(self, y, X_feature, threshold):
        parent_entropy = self.entropy(y)
        
        left_indices, right_indices = self._split(X_feature, threshold)
        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0
        n_samples = len(y)
        n_left, n_right = len(left_indices), len(right_indices)
        left_entropy = self.entropy(y[left_indices])
        right_entropy = self.entropy(y[right_indices])

        ig = parent_entropy - (n_left / n_samples) * left_entropy - (n_right / n_samples) * right_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column < split_thresh).flatten()
        right_idxs = np.argwhere(X_column >= split_thresh).flatten()

        return left_idxs, right_idxs

    def entropy(self, y):
        histogram = np.bincount(y)
        p = histogram / len(y)
        return -np.sum([p_i * np.log2(p_i) for p_i in p if p_i > 0])
    

    def accuracy(self, y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)
    

