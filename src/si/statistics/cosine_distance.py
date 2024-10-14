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
    return 1-similarity