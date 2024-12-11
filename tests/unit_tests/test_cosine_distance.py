from unittest import TestCase

import numpy as np
from datasets import DATASETS_PATH

import os
from si.statistics.cosine_distance import cosine_distance
from si.io.csv_file import read_csv
from sklearn.metrics.pairwise import cosine_distances


class TestCosineDistance(TestCase):
    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)


    def test_cosine_distance(self):
        
        x= self.dataset.X[0,:] # Seleciona a primeira amostra como vetor x
        y = self.dataset.X[1:,:] # Seleciona as demais amostras como o conjunto de vetores y

        package = cosine_distance(x, y) # Calcula a distância do cosseno usando a implementação criada
        sk_learn = cosine_distances(x.reshape(1,-1),y) # Calcula a distância do cosseno usando a função do scikit-learn

        self.assertTrue(np.allclose(sk_learn[0],package))  # Verifica se os dois resultados são aproximadamente iguais
        