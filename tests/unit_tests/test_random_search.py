from unittest import TestCase
from datasets import DATASETS_PATH
import os
from si.io.data_file import read_data_file
from si.model_selection.randomized_search import randomized_search_cv
from si.models.logistic_regression import LogisticRegression
import numpy as np
from si.metrics.accuracy import accuracy

class TestRandomizedSearchCV(TestCase):
    def setUp(self):
        """
        Set up the test environment, including loading the dataset.
        """
        # Carregar o dataset breast-bin.csv
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

    def test_randomized_search(self):
        """
        Test the randomized_search_cv function with specified hyperparameter distributions.
        """
        # Criar o modelo LogisticRegression
        model = LogisticRegression()

        # Definir a grade de hiperparâmetros
        hyperparameter_grid = {
            'l2_penalty': np.linspace(1, 10, 10),
            'alpha': np.linspace(0.001, 0.0001, 100),
            'max_iter': np.linspace(1000, 2000, 200, dtype=int)
        }

        # Realizar a busca aleatória
        results = randomized_search_cv(
            model=model,
            dataset=self.dataset,
            hyperparameter_grid=hyperparameter_grid,
            scoring=accuracy,
            cv=3,
            n_iter=10
        )

        # Resultados do teste
        print("Resultados da Busca Aleatória:")
        print(f"Scores: {results['scores']}")
        print(f"Melhores Hiperparâmetros: {results['best_hyperparameters']}")
        print(f"Melhor Score: {results['best_score']}")

        # Verificar a saída
        self.assertEqual(len(results['scores']), 10, "O número de scores deve ser igual a n_iter.")
        self.assertIsInstance(results['best_hyperparameters'], dict, "Os melhores hiperparâmetros devem ser um dicionário.")
        self.assertTrue(0 <= results['best_score'] <= 1, "O melhor score deve estar entre 0 e 1.")
