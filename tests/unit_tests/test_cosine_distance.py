from unittest import TestCase

import numpy as np
from datasets import DATASETS_PATH

import os
from si.statistics.cosine_distance import cosine_distance
from si.io.csv_file import read_csv
from sklearn.metrics.pairwise import cosine_distances


class TestCosineDistance(TestCase): #A classe encapsula os testes para validar o funcionamento da função cosine_distance
    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)


    def test_cosine_distance(self): 
        
        x= self.dataset.X[0,:] # Seleciona a primeira amostra como vetor x para o cálculo da distância.
        y = self.dataset.X[1:,:] # Seleciona as demais amostras como o conjunto de vetores y

        package = cosine_distance(x, y) # Calcula a distância do cosseno entre 𝑥 e 𝑦 usando a função personalizada cosine_distance
        sk_learn = cosine_distances(x.reshape(1,-1),y) # Calcula a distância do cosseno entre 𝑥 e 𝑦 usando a função equivalente do scikit-learn
        #x.reshape(1, -1): Ajusta o formato de 𝑥 para ser compatível com a função do scikit-learn (de 𝑑 para 1 × 𝑑)
        
        self.assertTrue(np.allclose(sk_learn[0],package))  # Verifica se os dois resultados são aproximadamente iguais
        #np.allclose: Compara os elementos dos dois arrays, verificando se a diferença está dentro de uma tolerância aceitável.

    """
Objetivo do Teste: Garantir que a função personalizada cosine_distance calcula corretamente a distância do cosseno, produzindo resultados equivalentes à implementação padrão do scikit-learn.

Calcula a distância do cosseno entre 𝑥 e 𝑦 usando duas abordagens:
A função personalizada (cosine_distance).
A função do scikit-learn (cosine_distances).
Compara os resultados para verificar se são equivalentes.
    """