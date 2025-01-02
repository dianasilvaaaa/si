import numpy as np
from unittest import TestCase
from si.neural_networks.layers import Dropout  

class TestDropoutLayer(TestCase):

    def setUp(self):
        self.probability = 0.5
        self.layer = Dropout(probability=self.probability)
        self.input_data = np.random.rand(4, 5)  # 4 exemplos com 5 características cada

    def test_forward_training_mode(self):
        output = self.layer.forward_propagation(self.input_data, training=True)
        self.assertEqual(output.shape, self.input_data.shape)
        # Verifica se há zeros na saída devido à máscara
        self.assertTrue(np.any(output == 0))
        # Verifica se o escalonamento foi aplicado corretamente
        scaling_factor = 1 / (1 - self.probability)
        self.assertTrue(np.all((output == 0) | (output >= self.input_data * scaling_factor)))


    def test_forward_inference_mode(self):
        output = self.layer.forward_propagation(self.input_data, training=False)
        # Verifica se a saída é igual à entrada (nenhuma modificação ocorre no modo de inferência)
        np.testing.assert_array_equal(output, self.input_data)


    def test_backward_propagation(self):
        # Simula a propagação direta no modo de treinamento para gerar a máscara
        self.layer.forward_propagation(self.input_data, training=True)
        output_error = np.random.rand(4, 5)# Simula o erro de saída
        input_error = self.layer.backward_propagation(output_error)
        expected_input_error = output_error * self.layer.mask
        np.testing.assert_array_equal(input_error, expected_input_error)

    def test_output_shape(self):
        self.layer.set_input_shape(self.input_data.shape)
        self.assertEqual(self.layer.output_shape(), self.input_data.shape)

    def test_parameters(self):
        self.assertEqual(self.layer.parameters(), 0)