import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse

class LassoRegressor(Model):
    """
    Modelo de Regressão Lasso utilizando regularização L1.
    Este método reduz coeficientes menos importantes, tornando o modelo mais simples e interpretável.
    """

    def __init__(self, l1_weight: float = 1.0, iterations: int = 1000, tolerance: int = 5, normalize: bool = True, **kwargs):
        """
        Inicializa o modelo de regressão Lasso.

        Parâmetros:
        -----------
        l1_weight : float
            O peso da penalização L1 no modelo.
        iterations : int
            Número máximo de iterações para o ajuste.
        tolerance : int
            Quantidade de iterações sem melhoria antes de parar.
        normalize : bool
            Indica se os dados devem ser normalizados.
        """
        
        super().__init__(**kwargs)
        self.l1_weight = l1_weight
        self.iterations = iterations
        self.tolerance = tolerance
        self.normalize = normalize

        # Atributos do modelo
        self.coefficients = None
        self.intercept = None
        self.data_mean = None
        self.data_std = None
        self.history = {}

    def fit(self, dataset: Dataset) -> 'LassoRegressor':
        """
        Ajusta o modelo aos dados utilizando um método iterativo.

        Parâmetros:
        -----------
        dataset : Dataset
            O conjunto de dados de entrada.

        Retorna:
        --------
        self : LassoRegressor
            O modelo ajustado.
        """
        # Normalização dos dados
        if self.normalize:
            self.data_mean = np.mean(dataset.X, axis=0)
            self.data_std = np.std(dataset.X, axis=0)
            X = (dataset.X - self.data_mean) / self.data_std
        else:
            X = dataset.X

        y = dataset.y
        num_samples, num_features = X.shape

        # Inicialização dos parâmetros
        self.coefficients = np.zeros(num_features)
        self.intercept = 0

        no_improve_counter = 0  # Para controle de parada antecipada

        for iteration in range(self.iterations):
            # Previsões atuais
            y_pred = np.dot(X, self.coefficients) + self.intercept

            # Atualiza o intercepto (termo livre)
            self.intercept = np.mean(y - np.dot(X, self.coefficients))

            # Atualiza cada coeficiente separadamente
            for feature_idx in range(num_features):
                # Remove a contribuição da feature atual
                residual = y - (np.dot(X, self.coefficients) - X[:, feature_idx] * self.coefficients[feature_idx]) - self.intercept
                gradient = np.dot(X[:, feature_idx], residual) / num_samples

                # Aplica o método de thresholding para L1
                self.coefficients[feature_idx] = self._apply_threshold(gradient, self.l1_weight) / np.sum(X[:, feature_idx]**2)

            # Calcula o custo para verificar a melhoria
            current_cost = self.compute_cost(dataset)
            self.history[iteration] = current_cost

            if iteration > 0 and self.history[iteration] >= self.history[iteration - 1]:
                no_improve_counter += 1
                if no_improve_counter >= self.tolerance:
                    break
            else:
                no_improve_counter = 0

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Realiza previsões para novos dados.

        Parâmetros:
        -----------
        dataset : Dataset
            Dados de entrada para prever os valores.

        Retorna:
        --------
        predictions : np.ndarray
            Valores previstos pelo modelo.
        """
        X = (dataset.X - self.data_mean) / self.data_std if self.normalize else dataset.X
        return np.dot(X, self.coefficients) + self.intercept

    def compute_cost(self, dataset: Dataset) -> float:
        """
        Calcula a função de custo, incluindo o termo de regularização L1.

        Parâmetros:
        -----------
        dataset : Dataset
            Dados de entrada para o cálculo do custo.

        Retorna:
        --------
        cost : float
            Valor da função de custo.
        """
        y_pred = self.predict(dataset)
        return (1 / (2 * len(dataset.y))) * np.sum((dataset.y - y_pred) ** 2) + self.l1_weight * np.sum(np.abs(self.coefficients))

    def score(self, dataset: Dataset) -> float:
        """
        Avalia o modelo utilizando o erro médio quadrático (MSE).

        Parâmetros:
        -----------
        dataset : Dataset
            Dados de entrada para avaliação.

        Retorna:
        --------
        mse : float
            Erro médio quadrático do modelo.
        """
        predictions = self.predict(dataset)
        return mse(dataset.y, predictions)

    def _apply_threshold(self, value: float, penalty: float) -> float:
        """
        Aplica a técnica de soft thresholding para regularização L1.

        Parâmetros:
        -----------
        value : float
            Valor residual para o coeficiente.
        penalty : float
            Penalização L1.

        Retorna:
        --------
        thresholded_value : float
            Valor após aplicação da regularização.
        """
        if value > penalty:
            return value - penalty
        elif value < -penalty:
            return value + penalty
        else:
            return 0.0