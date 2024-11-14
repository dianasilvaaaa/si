from unittest import TestCase
from datasets import DATASETS_PATH
import os
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.lasso_regression import LassoRegression
from si.metrics.mse import mse


class TestLassoRegressor(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_fit(self):

        lasso = LassoRegression()
        lasso.fit(self.train_dataset)

        self.assertEqual(lasso.theta.shape[0], self.train_dataset.shape()[1])
        self.assertNotEqual(lasso.theta_zero, None)
        self.assertNotEqual(len(lasso.cost_history), 0)
        self.assertNotEqual(len(lasso.mean), 0)
        self.assertNotEqual(len(lasso.std), 0)

    def test_predict(self):
        lasso = LassoRegression()
        lasso.fit(self.train_dataset)

        predictions = lasso.predict(self.test_dataset)

        self.assertEqual(predictions.shape[0], self.test_dataset.shape()[0])
    
    def test_score(self):
        lasso = LassoRegression()
        lasso.fit(self.train_dataset)
        
        score = lasso.score(self.test_dataset)

        self.assertEqual(round(score, 2), round(mse(self.test_dataset.y,lasso.predict(self.test_dataset)),2))