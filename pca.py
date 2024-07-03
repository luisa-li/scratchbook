import numpy as np 
from sklearn.datasets import load_breast_cancer

def pca(D: np.ndarray, components: int):
    
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