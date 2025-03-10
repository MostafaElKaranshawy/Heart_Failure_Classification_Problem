# Heart Failure Classification Problem

## Data Preprocessing

- **Fixed Random Seed** used to generate the same random subset of the data each time the algorithm runs.
- **Train/Validation/Test Split** Split the dataset into 70% training, 10% validation, and 20% testing
- **Preprocesssing** ensures that the data is in numerical format
  - removing NAs
  - normalizing
  - Categorical feature must be encoded
    - 0, 1 for the binary categories and so on.....
- **Validation set** is the test set but in the training not evaluating -> used to tuning hyperparameters -> select the best model, ex-> in KNN select the best K and so on...

## Algorithms

- **While training** use training set only, but if the classifier has any hyperparameters, then use the validation set.
- **After training** use the test set to evaluate the model

### Decision Tree

- Implementation from scratch.
- use **`IG`** to split at each node.
- **Hint**:
  - You can structure your tree as a recursive set of nodes.
  - Each leaf will output a class prediction.
  - Consider implementing a maximum depth or minimum samples split to prevent overfitting.
  - Train your decision tree on the training set.

### Bagging Ensemble

- Use **decision tree** to as the learner for the bagging -> means that we will repeat DT Algorithm over the randomly generated dataset
  - **Randomly generated dataset**: is the dataset created by repeating each sample **$m$** times (number of the classifiers) then shuffle the data -> all the samples have the same weight in the large dataset.
  - Use training dataset for Training.

### AdaBoost Ensemble

- Use Decision Stamp as the weak learner -> (single node DT to split).
- Reweight data.
- can use Decision tree with maximum depth set to 1 or a few levels.
- Train the AdaBoost ensemble on the training data.
- You should decide on a reasonable number of boosting rounds (for instance, 50 rounds) or experiment with the number of weak learners using the validation set. -> **Number of weak learners is a hyperparameter so we use validation set to choose it.**

## Evaluation

- Compute the accuracy and F-Score for each model.
- Plot the **confusion matrices** and find the most confusing classes After training the models.
- evaluate each of them on the test set and record the performance metrics.
