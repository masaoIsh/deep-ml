import numpy as np


def cosine_similarity(row1, row2):
    norm_1 = np.sqrt(np.sum(row1**2, axis=0))
    norm_2 = np.sqrt(np.sum(row2**2, axis=0))
    
    if norm_1 == 0 or norm_2 == 0:
        return 0.0
    else:
        return np.dot(row1, row2) / (norm_1*norm_2)

def pairwise_cosine_similarity(X):
    # Your code here
    X = np.array(X, dtype=float)
    n, d = X.shape
    S = np.zeros((n ,n))

    for i in range(n):
        for j in range(n):
            S[i, j] = cosine_similarity(X[i], X[j])

    return S