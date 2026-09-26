import math

def orthogonal_projection(v, L):
	"""
	Compute the orthogonal projection of vector v onto line L.

	:param v: The vector to be projected
	:param L: The line vector defining the direction of projection
	:return: List representing the projection of v onto L
	"""
	dot_product = sum([v[i]*L[i] for i in range(len(v))])

	L_norm_sq = sum([L[i]**2 for i in range(len(L))])

	tomul = dot_product / (L_norm_sq)

	return [L[i]*tomul for i in range(len(L))]
	
