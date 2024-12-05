from unittest import TestCase
from datasets import DATASETS_PATH
import os
from si.io.data_file import read_data_file
from si.model_selection.randomized_search import randomized_search_cv_v2  
from si.models.logistic_regression import LogisticRegression
import numpy as np

class TestRandomGridSearchCV(TestCase):

    def setUp(self):
        # Caminho para o arquivo CSV
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        # Carregar o conjunto de dados
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

    def test_random_grid_search_k_fold_cross_validation(self):
        # Criar o modelo de regressão logística
        model = LogisticRegression()

        # Definir o grid de parâmetros (l2_penalty, alpha e max_iter)
        parameter_grid = {
            'l2_penalty': np.linspace(1, 10, 10),  # Valores de l2_penalty de 1 a 10
            'alpha': np.linspace(0.001, 0.0001, 100),  # Valores de alpha entre 0.001 e 0.0001
            'max_iter': np.linspace(1000, 2000, 200)  # Valores de max_iter entre 1000 e 2000
        }

        # Realizar a procura aleatória de hiperparâmetros com validação cruzada de 3 divisões e 10 combinações aleatórias
        results_ = randomized_search_cv_v2(model=model,
                                           dataset=self.dataset,
                                           hyperparameter_grid=parameter_grid,
                                           scoring=None,  # Definir o scoring se necessário (por exemplo, 'accuracy')
                                           cv=3,
                                           n_iter=10)

        # Verificar se foram gerados 10 resultados
        self.assertEqual(len(results_["scores"]), 10)

        # Verificar se os melhores hiperparâmetros têm 3 parâmetros
        best_hyperparameters = results_['best_hyperparameters']
        self.assertEqual(len(best_hyperparameters), 3)

        # Verificar se o melhor score está dentro de um intervalo esperado (por exemplo, perto de 0.97)
        best_score = results_['best_score']
        self.assertEqual(np.round(best_score, 2), 0.97)
