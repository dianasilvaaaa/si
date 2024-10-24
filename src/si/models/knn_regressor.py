from typing import Callable, Union

import numpy as np

from si.base.estimator import Estimator
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance

class KNNRegressor(Model):
    """
A regressão KNN é uma técnica de aprendizado automática não paramétrica, utilizada em problemas de regressão. 
O método faz a previsão de uma nova amostra com base na similaridade, 
considerando os valores das k amostras mais próximas presentes nos dados de treino para realizar a predição.

"""
      

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance, **kwargs):

        """
        Inicia o classificador KNN
        parametros: 'K:int' é p número de k exemplos mais proximos a considerar
        'distance:callable' é a função que calcula a distancia entre uma amostra e as amostras no conjunto de dados de treino   

        """

        super().__init__(**kwargs)
        self.k = k
        self.distance = distance

        self.dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':

        """
Fit ajusta o modelo ao conjunto de dados fornecido
parametros: 'dataset:Dataset' é o conjunto de dados para ajustar o modelo (conjunto de dados de treino)
retorna: 'self: KNNRegressor' é o modelo ajustado
"""
        self.dataset = dataset
        return self
    
    def _get_closest_value(self, sample: np.array) -> Union[int,float]:

        """
Devolve a label mais proxima mais proxima da amostra dada
parametros: 'sample:np.array' amostra para obter o valor mais proximo
retorna: 'value: int or float' o valor mais proxima
"""
        # calcular a distância entre a amostra e o conjunto de dados de treino
        distances = self.distance(sample, self.dataset.X)

        # obter os k vizinhos mais próximos
        k_nearest_neighbors = np.argsort(distances)[:self.k]

        # obter os valores dos k vizinhos mais próximos
        k_nearest_neighbors_label_values = self.dataset.y[k_nearest_neighbors]

        # obter o valor médio dos k vizinhos mais próximos
        value = np.sum(k_nearest_neighbors_label_values) / self.k

        return value
    
    def _predict(self, dataset: Dataset) -> np.ndarray:

        """
        Prevê os valores das etiquetas do conjunto de dados fornecido
        parametros: 'Dataset:Dataset' conjunto de dados para prever os valores de conjunto de dados de teste
        retorna: 'predictions:np.ndarray' uma matriz de valores previstos para o conjunto de dados de teste

"""
        # calcula as previsões para cada linha (amostra) do conjunto de dados de teste
        predictions = np.array([self._get_closest_value(x) for x in dataset.X])
        return predictions 
    
    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:


        """
Calcula o raiz do erro quadrático médio entre os valores estimados e os valores reais de um determinado conjunto de dados
parametros: 'dataset:Dataset' o dataset para avaliar o modelo
retorna: 'float' Corresponde à raiz do erro quadrático médio do modelo para o conjunto de dados dado

"""

        return rmse(dataset.y, predictions)
    
        
        