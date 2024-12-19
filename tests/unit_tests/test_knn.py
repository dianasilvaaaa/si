from unittest import TestCase
import numpy as np
from datasets import DATASETS_PATH
import os
from si.metrics.rmse import rmse
from si.models.knn_regressor import KNNRegressor
from si.io.csv_file import read_csv
from si.models.knn_classifier import KNNClassifier
from si.model_selection.split import train_test_split

class TestKNN(TestCase): #Define os testes para o KNNClassifier.

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        knn = KNNClassifier(k=3) #Cria um modelo KNN com k=3.

        knn.fit(self.dataset) #Ajusta o modelo (fit) ao dataset Iris.

        self.assertTrue(np.all(self.dataset.features == knn.dataset.features)) #Verifica se o modelo armazenou corretamente. (features): As características do dataset.
        self.assertTrue(np.all(self.dataset.y == knn.dataset.y)) #Verifica se o modelo armazenou corretamente. y: Os rótulos verdadeiros.

    def test_predict(self):
        knn = KNNClassifier(k=1) #Cria um modelo KNN com k=1.

        train_dataset, test_dataset = train_test_split(self.dataset) #Divide o dataset Iris em treino e teste.

        knn.fit(train_dataset) #Ajusta o modelo ao conjunto de treino.
        predictions = knn.predict(test_dataset) #Faz previsões para o conjunto de teste
        self.assertEqual(predictions.shape[0], test_dataset.y.shape[0]) #verifica se: Se o número de previsões é igual ao número de rótulos no conjunto de teste.
        self.assertTrue(np.all(predictions == test_dataset.y)) #verifica se: Se as previsões coincidem com os rótulos verdadeiros.

    def test_score(self):
        knn = KNNClassifier(k=3) #Cria um modelo KNN com k=3.

        train_dataset, test_dataset = train_test_split(self.dataset) #Divide o dataset Iris em treino e teste.

        knn.fit(train_dataset) #Ajusta o modelo ao conjunto de treino.
        score = knn.score(test_dataset)
        #Calcula a pontuação (score), que deve ser 1 (100% de acurácia).
        self.assertEqual(score, 1)

############################################
class TestKNNRegressor(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        knn = KNNRegressor(k=3) #Cria um modelo KNN para regressão com k=3.
        knn.fit(self.dataset) #Ajusta o modelo ao dataset

        self.assertTrue(np.all(self.dataset.features == knn.dataset.features)) #Verifica se o modelo armazenou corretamente:features: As características do dataset.
        self.assertTrue(np.all(self.dataset.y == knn.dataset.y)) ##Verifica se o modelo armazenou corretamente:y: Os valores reais. 

    def test_predict(self):
        knn = KNNRegressor(k=3) #Cria um modelo KNN para regressão com k=3.
        train_dataset, test_dataset = train_test_split(self.dataset) #Divide o dataset em treino e teste.

        knn.fit(train_dataset) #Ajusta o modelo ao conjunto de treino.
        predictions = knn.predict(test_dataset) #faz previsões para o conjunto de teste

        self.assertEqual(len(predictions), len(test_dataset.y)) #Se o número de previsões é igual ao número de rótulos no conjunto de teste.

    def test_score(self):
        knn = KNNRegressor(k=3) #Cria um modelo KNN para regressão com k=3.
        train_dataset, test_dataset = train_test_split(self.dataset) #Divide o dataset em treino e teste.

        knn.fit(train_dataset) #Ajusta o modelo ao conjunto de treino.
        predictions = knn.predict(test_dataset) #Faz previsões para o conjunto de teste.

        score = knn.score(test_dataset) #calcula A pontuação (score) do modelo.
        expect_score = rmse(test_dataset.y, predictions) #calcula o valor esperado do RMSE.
        self.assertAlmostEqual(score, expect_score) #Verifica se a pontuação do modelo é aproximadamente igual ao RMSE calculado.
