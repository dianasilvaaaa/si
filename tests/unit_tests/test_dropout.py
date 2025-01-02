import numpy as np
from unittest import TestCase
from si.neural_networks.layers import Dropout  

class TestDropoutLayer(TestCase):

    def setUp(self):
        self.probability = 0.5
        self.layer = Dropout(probability=self.probability)
        self.input_data = np.random.rand(4, 5)  # 4 exemplos com 5 características cada

    def test_forward_training_mode(self):
        output = self.layer.forward_propagation(self.input_data, training=True) #Executa a propagação direta no modo de treinamento. 
        self.assertEqual(output.shape, self.input_data.shape) #Verifica se a forma da saída corresponde à forma da entrada. 
        # Verifica se há zeros na saída devido à máscara
        self.assertTrue(np.any(output == 0)) #  Garante que há valores zero na saída, indicando que a máscara foi aplicada corretamente.
        # Verifica se o escalonamento foi aplicado corretamente
        scaling_factor = 1 / (1 - self.probability) #Calcula o fator de escalonamento esperado.
        self.assertTrue(np.all((output == 0) | (output >= self.input_data * scaling_factor))) #Verifica se os valores da saída são zeros ou foram escalonados corretamente. 


    def test_forward_inference_mode(self):
        output = self.layer.forward_propagation(self.input_data, training=False) #Executa a propagação direta no modo de inferência. 
        # Verifica se a saída é igual à entrada (nenhuma modificação ocorre no modo de inferência)
        np.testing.assert_array_equal(output, self.input_data) #Verifica se a saída é idêntica à entrada, confirmando que o Dropout não altera os dados no modo de inferência. 


    def test_backward_propagation(self):
        # Simula a propagação direta no modo de treinamento para gerar a máscara
        self.layer.forward_propagation(self.input_data, training=True) #Realiza uma propagação direta no modo de treinamento para gerar a máscara necessária para o backward propagation. 
        output_error = np.random.rand(4, 5)# Simula o erro de saída. Gera uma matriz aleatória para simular o erro de saída.
        input_error = self.layer.backward_propagation(output_error) #Calcula o erro de entrada usando o método de retropropagação.
        expected_input_error = output_error * self.layer.mask #O erro esperado é o erro de saída multiplicado pela máscara, já que apenas os neurônios ativos contribuem. 
        np.testing.assert_array_equal(input_error, expected_input_error) #Verifica se o erro de entrada calculado corresponde ao esperado. 

    def test_output_shape(self):
        self.layer.set_input_shape(self.input_data.shape) #Define manualmente a forma de entrada da camada.
        self.assertEqual(self.layer.output_shape(), self.input_data.shape) #Verifica se a forma da saída corresponde à forma da entrada. 

    def test_parameters(self): #Método que retorna o número de parâmetros treináveis da camada
        self.assertEqual(self.layer.parameters(), 0) #Verifica se o número de parâmetros é zero, já que o Dropout não possui pesos ou biases.