import numpy as np

class DecisionStump:
    def __init__(self):
        self.alpha = None          # The weight of this classifier
        self.threshold = None      # The threshold value for the split
        self.feature_col = None    # The feature to split on

        # Determines the direction of the inequality
        self.is_positive_class_less_than_or_equal = True

    def predict(self, x):
        if self.is_positive_class_less_than_or_equal:
            return np.where(x[:, self.feature_col] <= self.threshold, 1, -1)

        return np.where(x[:, self.feature_col] > self.threshold, 1, -1)
    

class AdaBoost:
    def __init__(self, num_iterations=50):
        self.num_iterations = num_iterations
        self.stumps = []

    def fit(self, x_train, y_train):
        n_samples, n_features = x_train.shape
        weights = np.ones(n_samples) / n_samples
        new_y_train = np.where(y_train == 0, -1, 1)

        for _ in range(self.num_iterations):
            stump = DecisionStump()
            min_error = float('inf')
            best_classified_samples = None

            # loop through each feature to choose the best split
            for feature in range(n_features):
                unique_values = np.unique(x_train[:, feature])

                # we use thresholds as midpoints of the unique values
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2

                # try each threshold from the thresholds of the feature
                for threshold in thresholds:
                    # work first as the left of <= is positive class
                    classified_samples = np.where(x_train[:, feature] <= threshold, 1, -1)
                    error = np.sum(weights[new_y_train != classified_samples])
                    if error < min_error:
                        min_error = error
                        stump.feature_col = feature
                        stump.threshold = threshold
                        best_classified_samples = classified_samples.copy()
                        stump.is_positive_class_less_than_or_equal = True

                    # now consider that the right of <= is the positive class
                    classified_samples = np.where(x_train[:, feature] > threshold, 1, -1)
                    error = np.sum(weights[new_y_train != classified_samples])
                    if error < min_error:
                        min_error = error
                        stump.feature_col = feature
                        stump.threshold = threshold
                        best_classified_samples = classified_samples.copy()
                        stump.is_positive_class_less_than_or_equal = False


            # min_error = max(min_error, 1e-10) # to avoid division by zero

            # calculating the weight of the stump
            stump.alpha = 0.5 * np.log((1 - min_error) / (min_error + 1e-10) )

            # updating and normalizing the weights
            weights *= np.exp(-stump.alpha * new_y_train * best_classified_samples)
            weights /= np.sum(weights)

            # adding the stump to the collection of stumps
            self.stumps.append(stump)

    def predict(self, x_test):
        n_samples = x_test.shape[0]
        predicted_ys = np.zeros(n_samples)

        for stump in self.stumps:
            predicted_ys += stump.alpha * stump.predict(x_test)

        return np.where(np.sign(predicted_ys) == -1, 0, 1)