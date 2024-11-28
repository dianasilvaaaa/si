import numpy as np
from unittest import TestCase
from datasets import DATASETS_PATH
import os
from si.io.data_file import read_data_file
from si.model_selection.split import stratified_train_test_split
from si.models.random_forest_classifier import RandomForestClassifier
from si.metrics.accuracy import accuracy


class TestRandomForestClassifier(TestCase):

    def setUp(self):
        # Configuração inicial para carregar o dataset e dividi-lo em treino e teste
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")
        self.train_dataset, self.test_dataset = stratified_train_test_split(self.dataset, test_size=0.3)

    def test_fit(self):
        # Testar se o modelo RandomForestClassifier treina corretamente
        random_forest = RandomForestClassifier(n_estimators=10, max_depth=5, seed=42)
        random_forest.fit(self.train_dataset)

        # Verificar se o número de árvores treinadas é igual a n_estimators
        self.assertEqual(len(random_forest.trees), random_forest.n_estimators)
        # Verificar se cada árvore usa um subconjunto correto de features
        for tree, features in random_forest.trees:
            self.assertEqual(len(features), random_forest.max_features)

    def test_predict(self):

        # Testar se o modelo gera previsões com o mesmo número de amostras que o dataset de teste
        random_forest = RandomForestClassifier(n_estimators=10, max_depth=5, seed=42)
        random_forest.fit(self.train_dataset)

        predictions = random_forest.predict(self.test_dataset)
        print(predictions)  # Inspecione o formato de retorno aqui

        self.assertIsInstance(predictions, np.ndarray)  # Verifica se é um array
        self.assertEqual(predictions.shape[0], self.test_dataset.X.shape[0])  # Corrige para garantir consistência

    def test_score(self):
        # Testar se a métrica de acurácia retorna o valor esperado
        random_forest = RandomForestClassifier(n_estimators=10, max_depth=5, seed=42)
        random_forest.fit(self.train_dataset)

        accuracy_ = random_forest.score(self.test_dataset)
        expected_accuracy = accuracy(self.test_dataset.y, random_forest.predict(self.test_dataset))

        print(round(accuracy_, 2))
        self.assertEqual(round(accuracy_, 2), round(expected_accuracy, 2))
