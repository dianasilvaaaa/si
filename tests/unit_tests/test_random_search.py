from unittest import TestCase
from datasets import DATASETS_PATH
import os
from si.io.data_file import read_data_file
from si.model_selection.randomized_search import randomized_search_cv_v2  # Adaptando para o nome do método atualizado
from si.models.logistic_regression import LogisticRegression
import numpy as np

class TestRandomGridSearchCV(TestCase):

    def setUp(self):
        # Ruta al archivo CSV
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        # Cargar el conjunto de datos
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

    def test_random_grid_search_k_fold_cross_validation(self):
        # Crear el modelo de regresión logística
        model = LogisticRegression()

        # Definir el grid de parámetros (l2_penalty, alpha y max_iter)
        parameter_grid = {
            'l2_penalty': np.linspace(1, 10, 10),  # Valores de l2_penalty de 1 a 10
            'alpha': np.linspace(0.001, 0.0001, 100),  # Valores de alpha entre 0.001 y 0.0001
            'max_iter': np.linspace(1000, 2000, 200)  # Valores de max_iter entre 1000 y 2000
        }

        # Realizar la búsqueda aleatoria de hiperparámetros con validación cruzada de 3 pliegues y 10 combinaciones aleatorias
        results_ = randomized_search_cv_v2(model=model,
                                           dataset=self.dataset,
                                           hyperparameter_grid=parameter_grid,
                                           scoring=None,  # Definir el scoring si es necesario (por ejemplo, 'accuracy')
                                           cv=3,
                                           n_iter=10)

        # Verificar que se generaron 10 resultados
        self.assertEqual(len(results_["scores"]), 10)

        # Verificar que los mejores hiperparámetros tengan 3 parámetros
        best_hyperparameters = results_['best_hyperparameters']
        self.assertEqual(len(best_hyperparameters), 3)

        # Verificar que el mejor puntaje esté dentro de un rango esperado (por ejemplo, cerca de 0.97)
        best_score = results_['best_score']
        self.assertEqual(np.round(best_score, 2), 0.97)
