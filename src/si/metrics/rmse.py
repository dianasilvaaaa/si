import numpy as np


def rmse(y_true:np.ndarray, y_pred: np.ndarray) -> float:

    """

 “"”
    Calcula a raiz do erro quadrático médio (rmse) entre os valores reais e os valores previstos.
    
    Parâmetros
    ----------
    y_true: np.ndarray
        - Uma matriz que contém os valores verdadeiros do rótulo
    
    y_pred: np.ndarray
        - Uma matriz que contém os valores previstos para o rótulo

    Retorna
    -------
    float:
        - O valor RMSE entre os valores reais e previstos
    “"”

"""

    diferenca = y_true - y_pred #Calcula a diferença entre os valores reais (y_true) e os valores previstos (y_pred). A diferença pode ser positiva ou negativa.
    erro_quadratico = diferenca ** 2 #Eleva ao quadrado os valores da diferença calculada. Isso elimina os sinais negativos e dá mais peso aos erros maiores.
    media_erro = np.mean(erro_quadratico) #Calcula a média (ou seja, a soma dos erros quadráticos dividida pelo número total de valores). A média representa o Mean Squared Error (MSE).
    raiz_erro = np.sqrt(media_erro) #Calcula a raiz quadrada da média do erro quadrático.
    #Este é o Root Mean Squared Error (RMSE), retornado pela função.

    return raiz_erro #Retorna o valor final de RMSE.

