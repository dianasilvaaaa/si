import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class LassoRegression(Model):
    """
    Regressão Lasso com regularização L1 para ajustar coeficientes.
    """

    def __init__(self, l1_penalty: float = 1, max_iter: int = 1000, patience: int = 5, scale: bool = True, **kwargs):
        """
        Inicializa o modelo Lasso Regression.

        Parâmetros:
        ----------
        l1_penalty : float
            Parâmetro de regularização L1.
        max_iter : int
            Número máximo de iterações.
        patience : int
            Número de iterações sem melhora antes de interromper.
        scale : bool
            Indica se os dados devem ser normalizados.
        """
        
        super().__init__(**kwargs)
        self.l1_penalty = l1_penalty
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale

        # Atributos do modelo
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}

    def _fit(self, dataset: Dataset) -> 'LassoRegression':
        """
        Ajusta o modelo aos dados fornecidos.

        Parâmetros:
        ----------
        dataset : Dataset
            Conjunto de dados de entrada.

        Retorna:
        -------
        self : LassoRegression
            O modelo ajustado.
        """
        # Normaliza os dados, se necessário
        if self.scale:
            self.mean, self.std = dataset.X.mean(axis=0), dataset.X.std(axis=0)
            X = (dataset.X - self.mean) / (self.std + 1e-8)  # Evita divisão por zero
        else:
            X = dataset.X

        m, n = X.shape

        # Inicializa os parâmetros
        self.theta = np.zeros(n)
        self.theta_zero = 0

        early_stopping_counter = 0
        for iteration in range(self.max_iter):
            y_pred = np.dot(X, self.theta) + self.theta_zero

            # Atualiza o intercepto
            self.theta_zero = np.mean(dataset.y - y_pred)

            # Atualiza os coeficientes
            for j in range(n):
                X_feature = X[:, j]
                residual = dataset.y - (y_pred - X_feature * self.theta[j])
                gradient = np.dot(X_feature, residual) / m
                self.theta[j] = self._apply_soft_threshold(gradient, self.l1_penalty) / (np.dot(X_feature, X_feature) / m)

            # Calcula o custo atual
            cost = self.cost(dataset)
            self.cost_history[iteration] = cost

            # Verifica se houve melhora no custo
            if iteration > 0 and cost >= self.cost_history[iteration - 1]:
                early_stopping_counter += 1
                if early_stopping_counter >= self.patience:
                    break
            else:
                early_stopping_counter = 0

        return self

    def cost(self, dataset: Dataset) -> float:
        """
        Calcula o custo do modelo.

        Parâmetros:
        ----------
        dataset : Dataset
            Dados de entrada para cálculo do custo.

        Retorna:
        -------
        cost : float
            Valor do custo (erro quadrático médio + regularização L1).
        """
        y_pred = self.predict(dataset)
        mse_error = np.mean((dataset.y - y_pred) ** 2)
        l1_term = self.l1_penalty * np.sum(np.abs(self.theta))
        return mse_error / 2 + l1_term

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Faz previsões para novos dados.

        Parâmetros:
        ----------
        dataset : Dataset
            Dados de entrada.

        Retorna:
        -------
        predictions : np.ndarray
            Previsões geradas pelo modelo.
        """
        X = (dataset.X - self.mean) / (self.std + 1e-8) if self.scale else dataset.X
        return np.dot(X, self.theta) + self.theta_zero

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Avalia o modelo com base no erro médio quadrático (MSE).

        Parâmetros:
        ----------
        dataset : Dataset
            Conjunto de dados para avaliação.
        predictions : np.ndarray
            Previsões do modelo.

        Retorna:
        -------
        mse : float
            Erro médio quadrático.
        """

        return mse(dataset.y, predictions)

    def _apply_soft_threshold(self, value: float, penalty: float) -> float:
        """
        Aplica a técnica de soft thresholding para regularização L1.

        Parâmetros:
        ----------
        value : float
            Gradiente calculado para um coeficiente.
        penalty : float
            Penalização L1.

        Retorna:
        -------
        thresholded_value : float
            Valor ajustado após a regularização.
        """
        if abs(value) > penalty:
            return np.sign(value) * (abs(value) - penalty)
        return 0.0
