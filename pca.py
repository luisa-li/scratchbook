import numpy as np 

def pca(D: np.ndarray, components: int) -> np.ndarray:
    """
    Module for principal componenet analysis.

    Parameters
    ----------
    D : np.ndarray
        The array to perform PCA on
    components : int
        The number of principal components use.

    Returns
    -------
    np.ndarray
        The array D projected onto the principal components.
    """
    
    means = np.mean(D, axis=0)
    std = (sum((i - means)**2 for i in D)/len(D))**0.5
    centered_D = (D - means) / std 
    
    covs = np.cov(centered_D.T)
    eigenvalues, eigenvectors = np.linalg.eig(covs)
    
    idx = np.argsort(np.abs(eigenvalues)) 
    sorted_vecs = eigenvectors[:, idx]
    
    W = sorted_vecs[:components, :] 
    proj_D = centered_D.dot(W.T)
    
    return proj_D