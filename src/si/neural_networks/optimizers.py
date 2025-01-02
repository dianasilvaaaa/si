from abc import abstractmethod
import numpy as np


class Optimizer:

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """
        Initialize the optimizer.

        Parameters
        ----------
        learning_rate: float
            The learning rate to use for updating the weights.
        momentum:
            The momentum to use for updating the weights.
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.retained_gradient = None

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w
        return w - self.learning_rate * self.retained_gradient

###### EX15 

class Adam(Optimizer):
    def __init__(self, learning_rate: float = 0.01, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-8):
        """
        Initialize the Adam optimizer.

        Parameters
        ----------
        learning_rate: float
            The learning rate to use for updating the weights.
        beta_1: float
            The exponential decay rate for the 1st moment estimates. Defaults to 0.9.
        beta_2: float
            The exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
        epsilon: float
            A small constant for numerical stability. Defaults to 1e-8.



A atualização dos pesos no Adam é ajustada para levar em conta essas estimativas, ajudando a lidar com 
gradientes que podem variar em magnitude. 

m: Estima a média móvel dos gradientes. Inicialmente, é uma matriz de zeros com o mesmo formato 
que os pesos w. 
v: Estima a média móvel do quadrado dos gradientes. Também começa como zeros. 
t: Contador do número de iterações, usado para corrigir vieses nas estimativas de momento. 
        """
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        
        self.m = None
        self.v = None
        self.t = 0

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer using the Adam optimizer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        # Verify if m and v are initialized, if not initialize them as matrices of zeros
        if self.m is None: 
#Verifica se m e v já foram inicializados. Caso contrário, os inicializa como matrizes de zeros do mesmo formato que os pesos w. 
            self.m = np.zeros_like(w)
        if self.v is None:
            self.v = np.zeros_like(w)

        # Update time stamp (t += 1)
        self.t += 1

        # Compute and update m(momentum m)
        #self.m: A média móvel anterior dos gradientes, ponderada por β1 . 
        #grad_loss_w: O gradiente atual, ponderado por 1−β1. 
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad_loss_w

        # Compute and update v(momentum v)
        #Similar ao primeiro momento, mas usando o quadrado dos gradientes para estimar a variância. 
        # Isso ajuda a ajustar a taxa de aprendizado com base na magnitude do gradiente. 
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (grad_loss_w ** 2)

        #Inicialmente, m e v são tendenciosos para valores próximos a zero (porque começam como zeros). 
        # A correção ajusta esse viés dividindo por 1−β1t e 1−β2t. 
        # Compute m_hat
        m_hat = self.m / (1 - self.beta_1 ** self.t)
        # Compute v_hat
        v_hat = self.v / (1 - self.beta_2 ** self.t)

        # Compute the moving averages and return the updated weights
        #m_hat: Gradiente corrigido (estimativa do gradiente médio). 
        #v_hat: Variância corrigida. 
        #Divide o gradiente médio pela raiz quadrada da variância corrigida, escalonando dinamicamente os passos com base na magnitude do gradiente. 
        return w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    