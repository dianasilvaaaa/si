from unittest import TestCase

import numpy as np
from datasets import DATASETS_PATH

import os
from si.io.csv_file import read_csv

from si.model_selection.split import train_test_split
from si.models.lasso_regression import LassoRegression

class TestLassoRegressor(TestCase):
    """
    Teste unitário para o modelo de regressão Lasso
    """

    def setUp(self):
        """
        Configuração inicial dos testes.
        """
        # Caminho do arquivo CSV
        self.csv_file = os.path.join("datasets", "cpu", "cpu.csv")

        # Carrega o conjunto de dados
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

        # Divide o conjunto em treino e teste
        self.train_data, self.test_data = train_test_split(self.dataset, test_size=0.2)

    def test_fit(self):
        """
        Testa se o modelo é ajustado corretamente nos dados de treino.
        """
        lasso = LassoRegression(l1_weight=1.0, iterations=500, normalize=True)
        lasso.fit(self.train_data)

        # Verifica se os coeficientes e o intercepto foram inicializados
        self.assertIsNotNone(lasso.coefficients)
        self.assertIsNotNone(lasso.intercept)

    def test_predict(self):
        """
        Testa as previsões do modelo após o ajuste.
        """
        lasso = LassoRegression(l1_weight=1.0, iterations=500, normalize=True)
        lasso.fit(self.train_data)

        # Realiza previsões no conjunto de teste
        predictions = lasso.predict(self.test_data)

        # Verifica se as previsões têm o mesmo tamanho que o conjunto de teste
        self.assertEqual(predictions.shape[0], self.test_data.shape()[0])

    def test_score(self):
        """
        Testa a avaliação do modelo utilizando o erro médio quadrático (MSE).
        """
        lasso = LassoRegression(l1_weight=1.0, iterations=500, normalize=True)
        lasso.fit(self.train_data)

        # Calcula o MSE no conjunto de teste
        mse_value = lasso.score(self.test_data)

        # Verifica se o MSE é aproximadamente o esperado (ajustado ao dataset usado)
        self.assertAlmostEqual(mse_value, 5777.56, places=2)
