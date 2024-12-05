from unittest import TestCase
import os
import numpy as np
from si.io.data_file import read_data_file
from si.model_selection.randomized_search import randomized_search_cv
from si.models.logistic_regression import LogisticRegression
from si.metrics.accuracy import accuracy
from datasets import DATASETS_PATH
from si.model_selection.cross_validate import k_fold_cross_validation

class TestRandomizedSearchCV(TestCase):
    def setUp(self):
        """
        Set up the test environment, including loading the dataset.
        """
        # Carregar o dataset breast-bin.csv
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

    def test_randomized_search_cv(self):
        """
        Test the randomized_search_cv function with specified hyperparameter distributions.
        """
        # Criar o modelo LogisticRegression
        model = LogisticRegression()

        # Definir a grade de hiperparâmetros
        hyperparameter_grid = {
            'l2_penalty': np.linspace(1, 10, 10),
            'alpha': np.linspace(0.001, 0.0001, 100),
            'max_iter': np.linspace(1000, 2000, 200, dtype=int),
        }

        # Realizar a busca aleatória
        results = randomized_search_cv(
            model=model,
            dataset=self.dataset,
            hyperparameter_grid=hyperparameter_grid,
            cv=3,
            n_iter=10,
            scoring=accuracy
        )

        # Logs para inspeção dos resultados
        print("Resultados da Busca Aleatória:")
        print(f"Scores: {results['scores']}")
        print(f"Melhores Hiperparâmetros: {results['best_hyperparameters']}")
        print(f"Melhor Score: {results['best_score']}")

        # Verificações de Validação
        # Verificar o número de scores gerados
        self.assertEqual(
            len(results['scores']), 10, "O número de scores deve ser igual ao número de iterações (n_iter)."
        )

        # Verificar que os melhores hiperparâmetros são um dicionário
        self.assertIsInstance(
            results['best_hyperparameters'], dict, "Os melhores hiperparâmetros devem ser um dicionário."
        )

        # Verificar que os melhores hiperparâmetros contêm todas as chaves esperadas
        self.assertSetEqual(
            set(results['best_hyperparameters'].keys()),
            {'l2_penalty', 'alpha', 'max_iter'},
            "Os melhores hiperparâmetros devem conter 'l2_penalty', 'alpha' e 'max_iter'.",
        )

        # Verificar que o melhor score está dentro dos limites esperados
        self.assertTrue(
            0 <= results['best_score'] <= 1, "O melhor score deve estar entre 0 e 1."
        )

        # Validar o formato das combinações de hiperparâmetros
        for combination in results['hyperparameters']:
            self.assertIsInstance(combination, dict, "Cada combinação de hiperparâmetros deve ser um dicionário.")

        # Validar os scores individuais
        for score in results['scores']:
            self.assertTrue(
                0 <= score <= 1, "Cada score deve estar entre 0 e 1."
            )