from unittest import TestCase
import numpy as np
from datasets import DATASETS_PATH
import os
from si.metrics.rmse import rmse
from si.models.knn_regressor import KNNRegressor
from si.io.csv_file import read_csv
from si.models.knn_classifier import KNNClassifier
from si.model_selection.split import train_test_split


class TestKNNRegressor(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        knn = KNNRegressor(k=3) #Cria um modelo KNN para regressão com k=3.
        #Inicializa um modelo KNN para regressão com k=3 (usa os 3 vizinhos mais próximos para prever os valores).
        knn.fit(self.dataset) #Ajusta o modelo ao dataset

        self.assertTrue(np.all(self.dataset.features == knn.dataset.features)) #Verifica se o modelo armazenou corretamente:features: As características do dataset.
        self.assertTrue(np.all(self.dataset.y == knn.dataset.y)) ##Verifica se o modelo armazenou corretamente:y: Os valores reais. 

    def test_predict(self):
        knn = KNNRegressor(k=3) #Cria um modelo KNN para regressão com k=3.
        train_dataset, test_dataset = train_test_split(self.dataset) #Divide o dataset em treino (ajustar) e teste (avaliar).

        knn.fit(train_dataset) #Ajusta o modelo ao conjunto de treino.
        predictions = knn.predict(test_dataset) #faz previsões para os dados do conjunto de teste

        self.assertEqual(len(predictions), len(test_dataset.y)) #Confirma que o número de previsões geradas é igual ao número de amostras no conjunto de teste.

    def test_score(self):
        knn = KNNRegressor(k=3) #Cria um modelo KNN para regressão com k=3.
        train_dataset, test_dataset = train_test_split(self.dataset) #Divide o dataset em treino e teste.

        knn.fit(train_dataset) #Ajusta o modelo ao conjunto de treino.
        predictions = knn.predict(test_dataset) #Faz previsões para o conjunto de teste.

        score = knn.score(test_dataset) #calcula A pontuação (score) do modelo. Para regressão, é geralmente o RMSE
        expect_score = rmse(test_dataset.y, predictions) #calcula o valor esperado do RMSE diretamente entre os valores reais e as previsões.
        self.assertAlmostEqual(score, expect_score) #Usa assertAlmostEqual para confirmar que a pontuação do modelo (calculada internamente) é praticamente igual ao RMSE calculado manualmente.
