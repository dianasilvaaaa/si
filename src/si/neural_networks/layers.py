from abc import ABCMeta, abstractmethod
import copy
from si.neural_networks.optimizers import Optimizer
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
        self.n_units = n_units #O número de unidades (neurônios) na camada densa. Este valor define quantos neurônios estarão presentes na camada.
        self._input_shape = input_shape #A forma da entrada (geralmente, a quantidade de neurônios na camada anterior).

#Variáveis que armazenam as informações necessárias para a operação da camada. weights e biases são os parâmetros que a camada aprenderá durante o treinamento.
        self.input = None
        self.output = None
        self.weights = None
        self.biases = None

    def initialize(self, optimizer: Optimizer) -> 'DenseLayer':

        """
Este método é responsável por inicializar os pesos e vieses da camada. Os pesos são inicializados de forma aleatória, com valores entre -0.5 e 0.5, e os vieses são inicializados com zero. 
O optimizer (que será um objeto da classe Optimizer) é copiado para as variáveis w_opt e b_opt, para ser usado mais tarde no processo de otimização (ajuste dos parâmetros durante o treinamento).

        """
        # initialize weights from a 0 centered uniform distribution [-0.5, 0.5)
        self.weights = np.random.rand(self.input_shape()[0], self.n_units) - 0.5
        # initialize biases to 0
        self.biases = np.zeros((1, self.n_units))
        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)
        return self

    def parameters(self) -> int:
        """
        Este método retorna o número total de parâmetros da camada, ou seja, a quantidade de elementos em weights e biases. Ele é importante para saber quantos parâmetros a rede possui no total, o que afeta o treinamento

        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:

        """
Este é o método que realiza a propagação para frente. Ou seja, pega o input (entrada da camada) e calcula a saída da camada de acordo com a fórmula:

output = 𝑋⋅𝑊+𝑏 

X é a entrada,
𝑊 são os pesos,
𝑏 são os vieses.

Ele retorna a saída da camada (self.output), que será usada como entrada para a próxima camada ou como o resultado final.

        forward_propagation(self, input): Método abstrato. Este método será responsável 
        por implementar a propagação para frente, ou seja, como os dados entram na camada e saem dela após algum processamento.

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
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output
    

######## EX 12

class Dropout(Layer):
    """
    Dropout layer for a neural network.
    """

    def __init__(self, probability: float):
        """
        Initialize the dropout layer.

        Parameters
        ----------
        probability: float
            The dropout rate, a value between 0 and 1.

            
probability: O parâmetro probability define a taxa de "desligamento" ou dropout rate. 
Ele é um valor entre 0 e 1, onde 0 significa "sem dropout" e 1 significa "desligar todos os neurônios". 
Esse valor define a probabilidade de cada neurônio ser desligado durante o treinamento.

mask: A mask é uma matriz binária que armazena quais neurônios estão ativos ou inativos durante a execução do dropout. 
Quando um neurônio é "desligado", ele será multiplicado por zero na máscara.

input e output: São as variáveis que armazenam a entrada e a saída da camada, respectivamente.



        """
        super().__init__()
        self.probability = probability
        self.mask = None
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
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        self.input = input
        if training:
            scaling_factor = 1 / (1 - self.probability)
            self.mask = np.random.binomial(1, 1 - self.probability, size=input.shape)
            self.output = input * self.mask * scaling_factor
        else:
            self.output = input
        return self.output

    def backward_propagation(self, error: np.ndarray) -> np.ndarray:
        """
        Perform backward propagation on the given error.

        Parameters
        ----------
        error: numpy.ndarray
            The error from the subsequent layer.

        Returns
        -------
        numpy.ndarray
            The propagated error to the previous layer.
        """
        return error * self.mask  # Only propagate error for active neurons

    def output_shape(self) -> tuple:
        """
        Return the shape of the output.

        Returns
        -------
        tuple
            The shape of the output, which is the same as the input shape.
        """
        return self.input_shape()

    def parameters(self) -> int:
        """
        Return the number of learnable parameters in the layer.

        Returns
        -------
        int
            Always returns 0 as dropout layers do not have learnable parameters.
        """
        return 0
