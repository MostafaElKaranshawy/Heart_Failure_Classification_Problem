import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier

class BaggingClassifier:
    def __init__(self, n_estimators=10, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        np.random.seed = 42
        self.classifiers = []
    
    def sampling(self, x_train, y_train):
        n_samples = x_train.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return x_train[indices], y_train[indices]
    
    def fit(self, x_train, y_train):
        self.classifiers = []
        for _ in range(self.n_estimators):
            x_sample, y_sample = self.sampling(x_train, y_train)
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(x_sample, y_sample)
            self.classifiers.append(tree)
    
    def test(self, x_test, y_test):
        n = x_test.shape[0]
        predictions = np.array([classifier.predict(x_test) for classifier in self.classifiers])
        majority_votes = [Counter(predictions[:, i]).most_common(1)[0][0] for i in range(n)]
        
        acc = 0.0
        for i in range (0, n):
            if majority_votes[i] == y_test[i]:
                acc += 1
        acc = acc / n
        return acc
    
    def predict(self, X):
        predictions = np.array([classifier.predict(X) for classifier in self.classifiers])
        majority_votes = [Counter(predictions[:, i]).most_common(1)[0][0] for i in range(X.shape[0])]
        return np.array(majority_votes)
