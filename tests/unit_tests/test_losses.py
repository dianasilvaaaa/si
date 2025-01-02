import os
from unittest import TestCase
from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.models.decision_tree_classifier import DecisionTreeClassifier
from datasets import DATASETS_PATH
from si.neural_networks.losses import BinaryCrossEntropy, MeanSquaredError, CategoricalCrossEntropy

class TestLosses(TestCase):

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_mean_squared_error_loss(self):

        error = MeanSquaredError().loss(self.dataset.y, self.dataset.y)

        self.assertEqual(error, 0)

    def test_mean_squared_error_derivative(self):

        derivative_error = MeanSquaredError().derivative(self.dataset.y, self.dataset.y)

        self.assertEqual(derivative_error.shape[0], self.dataset.shape()[0])

    def test_binary_cross_entropy_loss(self):

        error = BinaryCrossEntropy().loss(self.dataset.y, self.dataset.y)

        self.assertAlmostEqual(error, 0)

    def test_mean_squared_error_derivative(self):

        derivative_error = BinaryCrossEntropy().derivative(self.dataset.y, self.dataset.y)

        self.assertEqual(derivative_error.shape[0], self.dataset.shape()[0])

    def test_categorical_cross_entropy_loss(self):
        """
Este teste valida se a perda categórica cruzada é zero quando as previsões são iguais aos valores verdadeiros 
(y_pred == y_true). 

Na prática: Quando as previsões coincidem exatamente com os rótulos verdadeiros, a entropia cruzada deve 
ser zero, pois não há discrepância entre as distribuições. 

O teste usa assertAlmostEqual para comparar o valor calculado com o valor esperado (0), com uma margem 
de erro mínima. Isso verifica a consistência matemática da implementação.

        """
        error = CategoricalCrossEntropy().loss(self.dataset.y, self.dataset.y)
        self.assertAlmostEqual(error, 0)

    def test_categorical_cross_entropy_derivative(self):
        """
Este teste valida que a derivada da perda categórica cruzada é computada corretamente em termos de forma 
(shape). Especificamente, o teste verifica se a saída da derivada tem o mesmo número de amostras (linhas) 
que o conjunto de dados. 
 
O uso de assertEqual garante que o número de linhas na matriz derivada corresponde ao número de linhas em 
self.dataset. Este é um teste estrutural que confirma que a implementação lida corretamente com as 
dimensões do conjunto de dados.

        """
        derivative_error = CategoricalCrossEntropy().derivative(self.dataset.y, self.dataset.y)
        self.assertEqual(derivative_error.shape[0], self.dataset.shape()[0])