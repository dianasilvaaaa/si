from unittest import TestCase

import numpy as np
from datasets import DATASETS_PATH
import os
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.model_selection.split import stratified_train_test_split

class TestSplits(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_train_test_split(self):

        train, test = train_test_split(self.dataset, test_size = 0.2, random_state=123)
        test_samples_size = int(self.dataset.shape()[0] * 0.2)
        self.assertEqual(test.shape()[0], test_samples_size)
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - test_samples_size)

    def test_stratified_train_test_split(self):

        """
    Testa a função stratified_train_test_split para garantir que a divisão estratificada está correta.
    Verifica se a proporção das classes é mantida entre os datasets de treino e teste.
    """
    
    # Aplicar o split estratificado
        train, test = stratified_train_test_split(self.dataset, test_size=0.2, random_state=123)
    
    # Obter as proporções de classes no dataset original
        _, labels_counts = np.unique(self.dataset.y, return_counts=True)
        total_labels = np.sum(labels_counts)
        proportion = labels_counts / total_labels * 100
    
    # Obter as proporções de classes no dataset de treino
        _, labels_counts_train = np.unique(train.y, return_counts=True)
        total_labels_train = np.sum(labels_counts_train)
        proportion_train = labels_counts_train / total_labels_train * 100
    
    # Obter as proporções de classes no dataset de teste
        _, labels_counts_test = np.unique(test.y, return_counts=True)
        total_labels_test = np.sum(labels_counts_test)
        proportion_test = labels_counts_test / total_labels_test * 100
    
    # Verificar se o tamanho do dataset de teste é 20% do dataset original
        test_samples_size = int(self.dataset.shape()[0] * 0.2)
    
    # Assert para validar o tamanho dos datasets de treino e teste
        self.assertEqual(test.shape()[0], test_samples_size)
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - test_samples_size)
    
    # Assert para garantir que as proporções de classes nos datasets de treino e teste são semelhantes ao original
        self.assertTrue(np.allclose(proportion, proportion_train, rtol=1e-03))
        self.assertTrue(np.allclose(proportion, proportion_test, rtol=1e-03))