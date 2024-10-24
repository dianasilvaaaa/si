import numpy as np


def rmse(y_true:np.ndarray, y_pred: np.ndarray) -> float:

    """

 “"”
    Calcula a raiz do erro quadrático médio (rmse) entre os valores reais e os valores previstos.
    
    Parâmetros
    ----------
    y_verdadeiro: np.ndarray
        - Uma matriz que contém os valores verdadeiros do rótulo
    
    y_pred: np.ndarray
        - Uma matriz que contém os valores previstos para o rótulo

    Retorna
    -------
    flutuante
        - O valor RMSE entre os valores reais e previstos
    “"”

"""

    diferenca = y_true - y_pred
    erro_quadratico = diferenca ** 2
    media_erro = np.mean(erro_quadratico)
    raiz_erro = np.sqrt(media_erro)
    return raiz_erro

