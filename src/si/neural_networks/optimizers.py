import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, learning_rate: float):

        """
       Inicializa o otimizador com uma taxa de aprendizado específica.

        :param learning_rate: A taxa de aprendizado para o otimizador.

        """
        self.learning_rate = learning_rate
    @abstractmethod

    def update(self, w: np.array, grad_loss_w: np.array) -> np.array:

        
        """
        Método abstrato para atualizar os parâmetros de uma camada do modelo.

        :param parameters: Parâmetros atuais (pesos ou vieses) da camada do modelo.
        :param gradients: Gradientes da função de perda em relação aos parâmetros.
        :return: Parâmetros atualizados.
        """
        raise NotImplementedError
    
class SGD(Optimizer):
    def __init__(self, learning_rate: float, momentum:float = 0.0):

        super().__init__(learning_rate)
        self.momentum = momentum

        self.retained_gradient = None
    
    def update(self, w, grad_loss_w):
        
        if self.retained_gradient == None:
            self.retained_gradient = np.zeros(w.shape)

            self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w

            new_weights = w - self.learning_rate * self.retained_gradient

            return new_weights
        

