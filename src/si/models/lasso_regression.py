import numpy as np
from base.model import Model
from data.dataset import Dataset
from si.metrics.mse import mse

class LassoRegression(Model):

    """
O LassoRegression é um modelo linear que utiliza a regularização L1. 
Este modelo resolve o problema de regressão linear usando Coordinate Descent
    
"""

    def __init__(self, l1_penalty: float, scale:bool = True, max_iter: int = 1000, patience: int=5, **kwargs):


        """
 parameters:
- l1_penalty - L1 regularization parameter
- scale - wheter to scale the data or not
atributes:
-theta: np.array
-theta_zero: float
-mean: np.array
-std: np.array
"""

        super().__init__(**kwargs)
        self.l1_penalty = l1_penalty
        self.max_inter = max_iter
        self.patience = patience
        self.scale = scale

        # parametros a estimar
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}


    def _fit(self, dataset: Dataset) -> 'LassoRegression':

        """
        Fit the model to the dataset using coordinate descent
parametros: 
dataset:Dataset
Return:
Self: LassoRegression

"""
        if self.scale:
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            X = (dataset.X - self.mean) /self.std
        else:
            X = dataset.X

        y = dataset.y
        m, n = X.shape

        self.theta = np.zeros(n)
        self.theta_zero = 0

        # Coordinate Descent
        for iteration in range(self.max_iter):
            # Predicted y with current coefficients
            y_pred = np.dot(X, self.theta) + self.theta_zero

            # Update intercept (theta_zero)
            self.theta_zero = np.mean(y - np.dot(X, self.theta))

            # Update each coefficient using coordinate descent
            for j in range(n):
                # Compute residual without the contribution of feature j
                residual = y - (np.dot(X, self.theta) - X[:, j] * self.theta[j]) - self.theta_zero
                rho = np.dot(X[:, j], residual) / m

                # Soft-thresholding for Lasso (L1 regularization)
                if rho < -self.l1_penalty / 2:
                    self.theta[j] = rho + self.l1_penalty / 2
                elif rho > self.l1_penalty / 2:
                    self.theta[j] = rho - self.l1_penalty / 2
                else:
                    self.theta[j] = 0

            # Calculate the cost for early stopping
            cost = self.cost(dataset)
            self.cost_history[iteration] = cost

            # Early stopping based on improvement in cost
            if iteration > 0 and self.cost_history[iteration] >= self.cost_history[iteration - 1]:
                patience_count += 1
                if patience_count >= self.patience:
                    break
            else:
                patience_count = 0

        return self

    def _predict(self, dataset: Dataset) -> np.array:

        """
    Predict the output for the dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict.

        Returns
        -------
        predictions: np.array
            The predicted values of y.

"""

        X = (dataset.X - self.mean) / self.std if self.scale else dataset.X
        return np.dot(X, self.theta) + self.theta_zero

    def _score(self, dataset: Dataset) -> float:
        """
        Compute the Mean Squared Error (MSE) of the model on the dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the MSE on.

        Returns
        -------
        mse: float
            The Mean Squared Error of the model.
        """
        predictions = self._predict(dataset)
        return mse(dataset.y, predictions)

    def cost(self, dataset: Dataset) -> float:
        """
        Compute the cost function with L1 regularization.

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the cost function.

        Returns
        -------
        cost: float
            The cost function with L1 regularization.
        """
        y_pred = self._predict(dataset)
        return (np.sum((y_pred - dataset.y) ** 2) / (2 * len(dataset.y))) + self.l1_penalty * np.sum(np.abs(self.theta))


# Test the LassoRegression model
if __name__ == '__main__':
    from si.data.dataset import Dataset

    # Create a simple linear dataset
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    dataset_ = Dataset(X=X, y=y)

    # Initialize and fit the Lasso Regression model
    model = LassoRegression(l1_penalty=1.0, max_iter=1000, patience=5, scale=True)
    model._fit(dataset_)

    # Compute the score
    score = model._score(dataset_)

    # Compute the cost
    cost = model.cost(dataset_)

    # Predict a new sample
    y_pred = model._predict(Dataset(X=np.array([[3, 5]])))