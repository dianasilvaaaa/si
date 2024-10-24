from unittest import TestCase

import numpy as np
from datasets import DATASETS_PATH

import os
from si.io.csv_file import read_csv

from si.statistics.sigmoid_function import sigmoid_function

class TestSigmoidFunction(TestCase):

    def setup(self):
        self.csv_file = os.path.join(DATASETS_PATH, "iris", "iris.csv")

        self.dataset = read_csv(filename=self.csv_file, features=True, label= True)

    def test_sigmoid_function(self):

        y = np.array([1, 2, 3])

        self.assertTrue(all(sigmoid_function(y) > 0))
        self.assertTrue(all(sigmoid_function(y) < 1))
