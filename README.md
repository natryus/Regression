# Machine Learning Regression Project README

This repository contains a machine learning project that focuses on predicting the "target" variable based on other features present in the dataset.

## Requirements

- Python 3.11.4
- pandas==2.0.3
- numpy==1.25.1
- joblib==1.3.2
- matplotlib==3.7.2
- scikit-learn==1.3.0

## Regression Models

### MLPRegressor (Multi-Layer Perceptron Regressor)

- `hidden_layer_sizes`: Specifies the number of neurons in each hidden layer.
- `max_iter`: Determines the maximum number of iterations for training.
- `random_state`: Ensures reproducibility by setting the random seed.

### RandomForestRegressor (Random Forest Regressor)

- `max_depth`: Sets the maximum depth of individual decision trees.
- `n_estimators`: Specifies the number of decision trees in the random forest.
- `random_state`: Ensures reproducibility by setting the random seed.

### KNeighborsRegressor (K-Nearest Neighbors Regressor)

- `n_neighbors`: Sets the number of nearest neighbors to consider.
- `algorithm`: Specifies the algorithm used to compute nearest neighbors.
- `metric`: Defines the distance metric used to measure similarity.
- `leaf_size`: Determines the size of the leaf nodes in the KD tree (if applicable).

### SVR (Support Vector Regressor)

- `degree`: Specifies the polynomial degree for polynomial kernel functions.
- `kernel`: Determines the type of kernel used (e.g., linear, radial basis function).
- `C`: Sets the regularization parameter for controlling the trade-off between fitting the data and preventing overfitting.
- `gamma`: Specifies the kernel coefficient (auto or scale for default).

Each model follows a similar workflow, including data preparation, training, evaluation, and metric calculation. The choice of model depends on the specific requirements of the regression task and the characteristics of the dataset.
