import numpy as np

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
    #np.linalg.norm(x): Norma (magnitude) do vetor x
    #np.linalg.norm(y, axis=1): Normas dos vetores em y, calculadas ao longo das linhas (amostras)
    #Divide o produto escalar pelo produto das normas, resultando na similaridade do cosseno
    
    #calculo da distancia:
    return 1-similarity #Subtrai a similaridade de 1 para obter a distância do cosseno