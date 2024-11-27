from unittest import TestCase

import numpy as np
from datasets import DATASETS_PATH

import os
from si.io.csv_file import read_csv

from si.model_selection.split import train_test_split
from si.models.lasso_regression import LassoRegression


class LassoRegressionTest(TestCase):
    """
    Unidade de teste para a implementação da regressão Lasso.
    """

    def setUp(self):
        """
        Configura o ambiente de teste carregando dados e dividindo em treino e teste.
        """
        dataset_path = os.path.join("datasets", "cpu", "cpu.csv")
        self.data = read_csv(filename=dataset_path, features=True, label=True)
        self.train_data, self.test_data = train_test_split(self.data, test_size=0.2)

    def test_model_training(self):
        """
        Verifica se o modelo ajusta corretamente os parâmetros durante o treinamento.
        """
        model = LassoRegression(l1_penalty=0.5, max_iter=300, scale=True)
        model._fit(self.train_data)

        # Verifica se os coeficientes e o intercepto foram definidos
        self.assertIsInstance(model.theta, np.ndarray, "Os coeficientes não foram inicializados corretamente.")
        self.assertTrue(np.isfinite(model.theta_zero), "O intercepto não foi inicializado corretamente.")

    def test_model_predictions(self):
        """
        Avalia se o modelo gera previsões do mesmo tamanho que os dados de teste.
        """
        model = LassoRegression(l1_penalty=0.8, max_iter=200, scale=False)
        model._fit(self.train_data)

        predictions = model._predict(self.test_data)

        # Valida o tamanho das previsões
        self.assertEqual(predictions.size, self.test_data.X.shape[0], "As previsões não correspondem ao tamanho esperado.")

    def test_model_evaluation(self):
        """
        Testa a capacidade do modelo de calcular o erro médio quadrático (MSE).
        """
        model = LassoRegression(l1_penalty=1.0, max_iter=500, scale=True)
        model._fit(self.train_data)

        predictions = model._predict(self.test_data)
        mse_value = model._score(self.test_data, predictions)

        # Calcula o MSE esperado dinamicamente
        expected_mse = round(np.mean((self.test_data.y - predictions)**2), 2)
        self.assertAlmostEqual(mse_value, expected_mse, places=2, msg="O MSE calculado não corresponde ao esperado.")
