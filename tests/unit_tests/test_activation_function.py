from unittest import TestCase

from datasets import DATASETS_PATH

import os

from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.neural_networks.activation import ReLUActivation, SigmoidActivation,TanhActivation, SoftmaxActivation
import numpy as np

class TestSigmoidLayer(TestCase):

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_activation_function(self):

        sigmoid_layer = SigmoidActivation()
        result = sigmoid_layer.activation_function(self.dataset.X)
        self.assertTrue(all([i >= 0 and i <= 1 for j in range(result.shape[1]) for i in result[:, j]]))


    def test_derivative(self):
        sigmoid_layer = SigmoidActivation()
        derivative = sigmoid_layer.derivative(self.dataset.X)
        self.assertEqual(derivative.shape[0], self.dataset.X.shape[0])
        self.assertEqual(derivative.shape[1], self.dataset.X.shape[1])


class TestRELULayer(TestCase):

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_activation_function(self):

        relu_layer = ReLUActivation()
        result = relu_layer.activation_function(self.dataset.X)
        self.assertTrue(all([i >= 0 for j in range(result.shape[1]) for i in result[:, j]]))


    def test_derivative(self):
        sigmoid_layer = ReLUActivation()
        derivative = sigmoid_layer.derivative(self.dataset.X)
        self.assertEqual(derivative.shape[0], self.dataset.X.shape[0])
        self.assertEqual(derivative.shape[1], self.dataset.X.shape[1])


class TestTanhLayer(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")
        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_activation_function(self):
        tanh_layer = TanhActivation()
        result = tanh_layer.activation_function(self.dataset.X)
        self.assertTrue(all([-1 <= i <= 1 for j in range(result.shape[1]) for i in result[:, j]]))

    def test_derivative(self):
        tanh_layer = TanhActivation()
        derivative = tanh_layer.derivative(self.dataset.X)
        self.assertEqual(derivative.shape[0], self.dataset.X.shape[0])
        self.assertEqual(derivative.shape[1], self.dataset.X.shape[1])


class TestSoftmaxLayer(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")
        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_activation_function(self):
        """
• Objetivo: Garantir que a função de ativação Softmax produz probabilidades normalizadas, ou seja, cada linha 
da saída deve somar exatamente 1. 

• Importância: 
- A Softmax é usada para converter logits em probabilidades. Se as somas das linhas não forem próximas 
de 1, a saída não será interpretável como probabilidades. 

- Este teste detecta erros no cálculo de normalização, como esquecer de dividir pelo somatório ou 
problemas de precisão.


        """
        softmax_layer = SoftmaxActivation()
        result = softmax_layer.activation_function(self.dataset.X)
        # Each row in softmax output should sum to 1
        self.assertTrue(all([np.isclose(np.sum(row), 1.0) for row in result]))

    def test_derivative(self):

        """
Objetivo: Verificar se a derivada da Softmax tem o mesmo shape que a entrada. 

Importância: 
• Assim como na TanhActivation, a derivada da Softmax é usada no backpropagation. O shape da derivada 
deve coincidir com o da entrada para que a rede possa calcular os gradientes corretamente. 
• Este teste garante que o cálculo da derivada está alinhado com o fluxo esperado de dados durante o 
treinamento.

        """
        softmax_layer = SoftmaxActivation()
        derivative = softmax_layer.derivative(self.dataset.X)
        self.assertEqual(derivative.shape[0], self.dataset.X.shape[0])
        self.assertEqual(derivative.shape[1], self.dataset.X.shape[1])