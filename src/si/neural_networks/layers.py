from abc import ABCMeta, abstractmethod
import copy

import numpy as np

from si.neural_networks.optimizers import Optimizer


class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forward_propagation(self, input):
        raise NotImplementedError
    
    @abstractmethod
    def backward_propagation(self, error):
        raise NotImplementedError
    
    @abstractmethod
    def output_shape(self):
        raise NotImplementedError
    
    @abstractmethod
    def parameters(self):
        raise NotImplementedError
    
    def set_input_shape(self, input_shape):
        self._input_shape = input_shape

    def input_shape(self):
        return self._input_shape
    
    def layer_name(self):
        return self.__class__.__name__
    
class DenseLayer(Layer):
    """
    Dense layer of a neural network.
    """

    def __init__(self, n_units: int, input_shape: tuple = None):
        """
        Initialize the dense layer.

        Parameters
        ----------
        n_units: int
            The number of units of the layer, aka the number of neurons, aka the dimensionality of the output space.
        input_shape: tuple
            The shape of the input to the layer.
        """

        super().__init__()
        self.n_units = n_units #n_units: Número de neurônios ou dimensões no espaço de saída.
        self._input_shape = input_shape #input_shape: Formato da entrada para inicialização.

        ##Inicializa atributos para armazenar informações sobre a camada.
        self.input = None
        self.output = None
        self.weights = None
        self.biases = None

    def initialize(self, optimizer: Optimizer) -> 'DenseLayer':
        # initialize weights from a 0 centered uniform distribution (-0.5, 0.5)
        self.weights = np.random.rand(self.input_shape()[0], self.n_units) - 0.5 #Os pesos são gerados de forma aleatória no intervalo [-0.5, 0.5).
        # initialize biases to 0
        self.biases = np.zeros((1, self.n_units)) #Os biases são inicializados como zeros.
        #Duplica o otimizador para pesos e biases.
        self.w_opt = copy.deepcopy(optimizer) 
        self.b_opt = copy.deepcopy(optimizer)
        return self

    def parameters(self) -> int:

        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        #Calcula o número total de parâmetros da camada (pesos e vieses).
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation on the given input.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        self.input = input #self.input: Armazena o input recebido.
        self.output = np.dot(self.input, self.weights) + self.biases #Multiplica o input pelos pesos.
        return self.output #Soma os vieses ao resultado.
    

    def backward_propagation(self, output_error: np.ndarray) -> float:
        #Realiza o cálculo dos erros para retropropagação.
        """
        Perform backward propagation on the given output error.
        Computes the dE/dW, dE/dB for a given output_error=dE/dY.
        Returns input_error=dE/dX to feed the previous layer.
        Parameters
        ----------
        output_error: numpy.ndarray
            The output error of the layer.
        Returns
        -------
        float
            The input error of the layer.
        """
        # computes the layer input error (the output error from the previous layer),
        # dE/dX, to pass on to the previous layer
        # SHAPES: (batch_size, input_columns) = (batch_size, output_columns) * (output_columns, input_columns)
        input_error = np.dot(output_error, self.weights.T)  #Calcula o erro de entrada para camadas anteriores.

        # computes the weight error: dE/dW = X.T * dE/dY
        # SHAPES: (input_columns, output_columns) = (input_columns, batch_size) * (batch_size, output_columns)

        # Calcula os erros nos pesos e biases.
        weights_error = np.dot(self.input.T, output_error)
        # computes the bias error: dE/dB = dE/dY
        # SHAPES: (1, output_columns) = SUM over the rows of a matrix of shape (batch_size, output_columns)
        bias_error = np.sum(output_error, axis=0, keepdims=True)

        # updates parameters
        #Atualiza pesos e biases usando o otimizador.
        self.weights = self.w_opt.update(self.weights, weights_error)
        self.biases = self.b_opt.update(self.biases, bias_error)
        return input_error

    def output_shape(self) -> tuple:
        #Retorna o formato da saída da camada.
        """
        Returns the shape of the output of the layer.
        Returns
        -------
        tuple
            The shape of the output of the layer.
        """
        return (self.n_units,)


###### EX 12

class Dropout(Layer):
    #Esta classe implementa uma camada de Dropout, que ajuda a reduzir o overfitting durante o treinamento de redes neurais.
    """
    Dropout layer for neural networks.
    """

    def __init__(self, probability: float):
        """
        Initialize the dropout layer.

        Parameters
        ----------
        probability: float
            The dropout rate, between 0 and 1.
        """
        super().__init__()
        self.probability = probability 
        #Taxa de dropout, ou seja, a fração de unidades que serão "desligadas" (não usadas) durante o treinamento. Deve estar no intervalo [0, 1).
        
        self.mask = None #Máscara binária que determina quais unidades são mantidas (1) ou desligadas (0).
        #self.input e self.output: Variáveis para armazenar os dados de entrada e saída da camada.
        self.input = None
        self.output = None

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation on the given input.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        self.input = input #Armazena o input para uso posterior(backward_propagation)

        #durante o treinamento
        if training:
            # Compute scaling factor
            scaling_factor = 1 / (1 - self.probability) #Fator de escala para manter o valor esperado das ativações igual ao das ativações originais (compensa a fração "desligada").

            # Generate the mask
            self.mask = np.random.binomial(1, 1 - self.probability, size=input.shape) #Gera uma máscara binária com probabilidade de 1 igual a 1 - probability.

            # Apply the mask and scale the output
            self.output = input * self.mask * scaling_factor #Multiplica o input pela máscara e pelo fator de escala.
        else:
            # During inference, the input is not changed
            self.output = input #No modo de inferência, não é aplicado dropout, e o input é simplesmente retornado.
        return self.output #Retorna o output da camada.

    def backward_propagation(self, output_error: np.ndarray) -> np.ndarray:
        """
        Perform backward propagation on the given output error.

        Parameters
        ----------
        output_error: numpy.ndarray
            The output error of the layer.

        Returns
        -------
        numpy.ndarray
            The input error of the layer.
        """
        # Multiply the output error by the mask
        return output_error * self.mask
#O erro vindo da camada seguinte.
#self.mask: Multiplica o erro pela máscara, garantindo que os neurônios "desligados" durante a propagação direta não contribuam para os gradientes.
#Retorna o erro ajustado para a camada anterior.

    def output_shape(self) -> tuple:
        
        """
        Returns the shape of the output of the layer.

        Returns
        -------
        tuple
            The input shape (dropout does not change the shape of the data).
        """
        return self.input_shape() #O Dropout não altera o formato do dado, então o formato de saída é igual ao formato de entrada.

    def parameters(self) -> int:

    
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            Always returns 0 (dropout layers do not have learnable parameters).
        """

        ##Camadas de Dropout não têm parâmetros treináveis (como pesos ou biases), então o número de parâmetros é sempre 0.
        return 0