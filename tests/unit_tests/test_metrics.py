from unittest import TestCase

import numpy as np
from si.metrics.accuracy import accuracy
from si.metrics.mse import mse
from si.statistics.sigmoid_function import sigmoid_function

class TestMetrics(TestCase):

    def test_accuracy(self):

        y_true = np.array([0,1,1,1,1,1,0])
        y_pred = np.array([0,1,1,1,1,1,0])

        self.assertTrue(accuracy(y_true, y_pred)==1)

    def test_mse(self):

        y_true = np.array([0.1,1.1,1,1,1,1,0])
        y_pred = np.array([0,1,1.1,1,1,1,0])

        self.assertTrue(round(mse(y_true, y_pred), 3)==0.004)
        

    def test_sigmoid_function(self):

        x = np.array([1.9, 10.4, 75])

        x_sigmoid = sigmoid_function(x)

        self.assertTrue(all(x_sigmoid >= 0))
        self.assertTrue(all(x_sigmoid <= 1))