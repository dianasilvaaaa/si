import numpy as np


def sigmoid_function(X: np.array) -> float:

    """
    Calcula a função sigmoide para os valores de entrada X.
    
    Argumentos:
    X -- valor ou array de valores para os quais calcular a sigmoide.
    
    Retorno:
    A probabilidade de cada valor de X ser 1, utilizando a função sigmoide.
    """
    return 1 / (1 + np.exp(-X))

        