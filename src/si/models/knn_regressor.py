from typing import Callable, Union

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance

class KNNRegressor(Model):
      

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance, **kwargs):


        super().__init__(**kwargs)
        self.k = k
        self.distance = distance

        self.dataset = None

    ########## NÃO ESTÁ TERMINADO ##########################
        