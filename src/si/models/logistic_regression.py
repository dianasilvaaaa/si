import numpy as np
from si.metrics.accuracy import accuracy
from si.statistics.sigmoid_function import sigmoid_function


class LogisticRegression(Model):

    def __init__(
            self, l2_penalty: float = 1, alpha: float = 0.001,
            max_iter: int = 1000, patience: int = 5, scale: bool = True, **kwargs):
        
        """
        Parameters
        ----------
        l2_penalty: float
            - The L2 regularization parameter. By default, it's set to 1.
        alpha: float
            - The learning rate. By default, it's set to 0.001.
        max_iter: int
            - The maximum number of iterations. By default, it's set to 1000.
        patience: int
            - The number of iterations without improvement before stopping the training .By default, it's set to 5.
        scale: bool
            - Whether to scale the dataset or not. By default, is set to True.        
        """
        # parameters
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale

        # attributes
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}

    def _fit(self, dataset: Dataset) -> 'LogisticRegression':
        """
        Fit the model to the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: RidgeRegression
            The fitted model
        """
        if self.scale:
            # compute mean and std
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            # scale the dataset
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0

        i = 0
        early_stopping = 0
        # gradient descent
        while i < self.max_iter and early_stopping < self.patience:
            # predicted y
            y_pred = sigmoid_function(np.dot(X, self.theta) + self.theta_zero)

            # computing and updating the gradient with the learning rate
            gradient = (self.alpha / m) * np.dot(y_pred - dataset.y, X)

            # computing the penalty
            penalization_term = self.theta * (1 - self.alpha * (self.l2_penalty / m))

            # updating the model parameters
            self.theta = penalization_term - gradient
            self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

            # compute the cost
            self.cost_history[i] = self.cost(dataset)
            if i > 0 and self.cost_history[i] > self.cost_history[i - 1]:
                early_stopping += 1
            else:
                early_stopping = 0
            i += 1

        return self

    def _predict(self, dataset: Dataset) -> np.array:
        """
        Predict the output of the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of

        Returns
        -------
        predictions: np.array
            The predictions of the dataset
        """
        
        X = (dataset.X - self.mean) / self.std if self.scale else dataset.X

        # predict the output using the logistic regression model
        prediction = sigmoid_function(np.dot(X, self.theta) + self.theta_zero)

        # converts the probabilities to binary values
        ## converting to 0
        indices_0 = np.where(prediction < 0.5)[0]
        prediction[indices_0] = 0

        ## converting to 1
        
        indices_1 = np.where(prediction >= 0.5)[0]
        prediction[indices_1] = 1

        return prediction

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Compute the Mean Square Error of the model on the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the MSE on

        Returns
        -------
        mse: float
            The Mean Square Error of the model
        """
        return accuracy(dataset.y, predictions)
    

    def cost(self, dataset: Dataset) -> float:
        """
        Compute the cost function (J function) of the model on the dataset using L2 regularization

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the cost function on

        Returns
        -------
        cost: float
            The cost function of the model
        """

        # feature predition
        y_pred = self.predict(dataset)


        # compute the cost function with L2 regularization
        regularization_term = self.l2_penalty / (2 * dataset.shape()[0]) * np.sum(self.theta ** 2)
        error_term = -1/dataset.shape()[0] * np.sum(dataset.y * np.log(y_pred) + (1 - dataset.y) * np.log(1 - y_pred))

