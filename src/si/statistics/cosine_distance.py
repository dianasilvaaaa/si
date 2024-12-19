import numpy as np

"""
O código implementa uma função chamada cosine_distance que calcula a distância do cosseno entre um vetor 
𝑥 (uma única amostra) e múltiplos vetores 𝑦 (amostras de comparação). 
A distância do cosseno é uma métrica de similaridade vetorial que mede a diferença angular entre vetores, ignorando a magnitude


"""


def cosine_distance(x:np.ndarray, y:np.ndarray)->np.ndarray:
    """
    Gives the distance between X and the various samples in Y

    Parameters
    ----------
    x : np.ndarray
        - A single sample
    y : np.ndarray
        - Multiple samples
    
    Returns
    -------
    np.ndarray
        - An array containing the distance between X and the various samples in Y    
    """

    similarity = np.dot(x,y.T)/ (np.linalg.norm(x)*np.linalg.norm(y,axis= 1))
    #np.dot(x, y.T): Calcula o produto escalar de x com cada vetor em y
    #np.linalg.norm(x): calcula a norma (magnitude) do vetor x, que é a raiz quadrada da soma dos quadrados dos elementos de 𝑥
    #np.linalg.norm(y, axis=1): calcula a norma para cada vetor em y ao longo das linhas (amostras)
    #Divide o produto escalar pelo produto das normas, resultando na similaridade do cosseno
    
    #calculo da distancia:
    return 1-similarity #Subtrai a similaridade de 1 para obter a distância do cosseno