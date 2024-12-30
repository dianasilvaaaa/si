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
        #Este método implementa o teste da funcionalidade de busca aleatória com validação cruzada.

        # Criar o modelo de regressão logística
        model = LogisticRegression() #Cria uma instância do modelo de regressão logística que será testado

        # Definir o grid de parâmetros (l2_penalty, alpha e max_iter)
        parameter_grid = {
            'l2_penalty': np.linspace(1, 10, 10),  # l2_penalty: Valores entre 1 e 10, divididos em 10 intervalos iguais
            'alpha': np.linspace(0.001, 0.0001, 100),  # alpha: Valores entre 0.001 e 0.0001, divididos em 100 intervalos iguais.
            'max_iter': np.linspace(1000, 2000, 200)  # max_iter: Valores entre 1000 e 2000, divididos em 200 intervalos iguais.
        }

        # Realizar a procura aleatória de hiperparâmetros com validação cruzada de 3 divisões e 10 combinações aleatórias
        results_ = randomized_search_cv_v2(model=model, #model: O modelo configurado
                                           dataset=self.dataset, #dataset: Dataset carregado.
                                           hyperparameter_grid=parameter_grid, #hyperparameter_grid: Grid de hiperparâmetros definido anteriormente
                                           scoring=None,  # Definir o scoring se necessário (por exemplo, 'accuracy')
                                           cv=3, #cv=3: Realiza validação cruzada com 3 folds
                                           n_iter=10) #n_iter=10: Testa 10 combinações aleatórias de hiperparâmetros

        # Verificar se foram gerados 10 resultados:
        #scores: Lista de pontuações para as 10 combinações testadas
        #best_hyperparameters: A melhor combinação de hiperparâmetros
        #best_score: A melhor pontuação obtida. 

        # O retorno (results_) é um dicionário com os resultados
        self.assertEqual(len(results_["scores"]), 10) #Garante que o número de pontuações geradas seja igual a n_iter (10).

        # Verificar se os melhores hiperparâmetros têm 3 parâmetros
        best_hyperparameters = results_['best_hyperparameters'] #best_hyperparameters: A melhor combinação de hiperparâmetros. 
        self.assertEqual(len(best_hyperparameters), 3)  #Verifica que a melhor configuração de hiperparâmetros contém exatamente 3 parâmetros (l2_penalty, alpha, max_iter).


        # Verificar se o melhor score está dentro de um intervalo esperado 
        best_score = results_['best_score'] 
        #Garante que o melhor score obtido esteja próximo de 0.97.
        self.assertEqual(np.round(best_score, 2), 0.97)
