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


#train_test_split: O tamanho dos conjuntos de treino e teste está correto.
    def test_train_test_split(self):

        train, test = train_test_split(self.dataset, test_size = 0.2, random_state=123) #Divide o dataset em treino e teste, com 20% das amostras no conjunto de teste
        test_samples_size = int(self.dataset.shape()[0] * 0.2) #Calcula o número esperado de amostras no conjunto de teste.
        self.assertEqual(test.shape()[0], test_samples_size) #Verifica se o tamanho do conjunto de teste está correto.
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - test_samples_size) #Verifica se o tamanho do conjunto de treino está correto.


#stratified_train_test_split: A divisão mantém a proporção de classes entre os conjuntos de treino e teste.
    def test_stratified_train_test_split(self):

        """
    Testa a função stratified_train_test_split para garantir que a divisão estratificada está correta.
    Verifica se a proporção das classes é mantida entre os datasets de treino e teste.
    """
    
    # Aplicar o split estratificado
        train, test = stratified_train_test_split(self.dataset, test_size=0.2, random_state=123) # Divide o dataset em treino e teste usando divisão estratificada.
    
    # Obter as proporções de classes no dataset original
        _, labels_counts = np.unique(self.dataset.y, return_counts=True) #Obtém a contagem de cada classe no dataset original.
        total_labels = np.sum(labels_counts) #Soma o número total de amostras.
        proportion = labels_counts / total_labels * 100 #Calcula a proporção percentual de cada classe.
    
    # Obter as proporções de classes no dataset de treino
        _, labels_counts_train = np.unique(train.y, return_counts=True) #Obtém a contagem de cada classe no conjunto de treino.
        total_labels_train = np.sum(labels_counts_train) #Soma o número total de amostras no treino.
        proportion_train = labels_counts_train / total_labels_train * 100 #Calcula a proporção percentual de cada classe.
    
    # Obter as proporções de classes no dataset de teste
        _, labels_counts_test = np.unique(test.y, return_counts=True) #Obtém a contagem de cada classe no conjunto de teste.
        total_labels_test = np.sum(labels_counts_test) # Soma o número total de amostras no teste.
        proportion_test = labels_counts_test / total_labels_test * 100 #Calcula a proporção percentual de cada classe
    
    # Verificar se o tamanho do dataset de teste é 20% do dataset original
        test_samples_size = int(self.dataset.shape()[0] * 0.2) 
    
    # Assert para validar o tamanho dos datasets de treino e teste
        self.assertEqual(test.shape()[0], test_samples_size) #Verifica se o tamanho do conjunto de teste está correto.
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - test_samples_size) #Verifica se o tamanho do conjunto de treino está correto.
    
    # Assert para garantir que as proporções de classes nos datasets de treino e teste são semelhantes ao original
        self.assertTrue(np.allclose(proportion, proportion_train, rtol=1e-03)) #Verifica se as proporções das classes no treino são similares ao dataset original.
        self.assertTrue(np.allclose(proportion, proportion_test, rtol=1e-03)) #Verifica se as proporções das classes no teste são similares ao dataset original.