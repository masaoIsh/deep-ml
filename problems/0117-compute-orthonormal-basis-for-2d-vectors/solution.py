import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    # Your code here
    if vectors == [[0, 0]]:
        return []
    
    vectors = np.array(vectors, dtype=float)
    v1 = vectors[0]
    v2 = vectors[1]
    proj_ontov1 = (np.dot(v1, v2) / np.dot(v1, v1)) * v1
    grammed = v2 - proj_ontov1
    orthogonals = [v1, grammed]

    normalized = []
    for vector in orthogonals:
        norm = np.linalg.norm(vector)
        if norm > tol:
            normalized.append(vector / norm)
    
    return normalized

    