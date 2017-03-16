"""Convert between 2d and flattened triangular arrays and indices."""

import numpy as np
import numba as nb


from .math import tri_n, tri_root, tri_root_rem


@nb.vectorize(nopython=True)
def nb_mat_idx_to_triu(row, col, n):
	"""Numba implementation of :func:`.mat_idx_to_triu` (numpy ufunc).

	row must be < col.

	:param int row: Matrix row index.
	:param int col: Matrix column index.
	:param int n: Matrix size.
	:returns: Linear index of position in upper triangle.
	:rtype int:
	"""
	# The following methods utilizes the symmetry between the upper and lower
	# indexing methods - if
	#     (row, col) <-- triu --> (i)
	# then 
	#     (n - row - 1, n - col - 1) <-- tril --> (T_{n-1} - i - 1)
	# i.e., if you switch which triangle you're on and reverse all indices
	# between their highest and lowest value the relationship still holds.
	# 
	# return tri_n(n - 1) - nb_mat_idx_to_tril(n - row - 1, n - col - 1, n) - 1
	# return tri_n(n - 1) - tri_n(n - row - 2) - (n - col - 1) - 1
	# return tri_n(n - 1) - tri_n(n - row - 2) - n + col
	# return tri_n(n - 1) - tri_n(n - row - 2) - (n - 1) + col - 1
	return tri_n(n - 2) - tri_n(n - row - 2) + col - 1


@nb.vectorize(nopython=True)
def nb_mat_idx_to_tril(row, col, n):
	"""Numba implementation of :func:`.mat_idx_to_tril` (numpy ufunc).

	row must be > col.

	:param int row: Matrix row index.
	:param int col: Matrix column index.
	:param int n: Matrix size.
	:returns: Linear index of position in lower triangle.
	:rtype int:
	"""
	return tri_n(row - 1) + col


def _ensure_order(indices1, indices2):
	"""
	Ensure that all elements of the first index array are smaller than the
	corresponding elements in the second.

	Used by mat_idx_to_triu and mat_idx_to_tril to ensure row and column
	indices are in the correct order.

	:param indices1: Scalar or array of integer indices.
	:param indices2: Scalar or array of integer indices in the same shape as
		indices2.
	:returns: Two-tuple of arrays where all elements in the first are smaller
		than elements of the second.
	:raises ValueError: If any array elements are equal.
	"""

	swapped = indices1 >= indices2
	if np.any(swapped):
		if np.any(indices1 == indices2):
			raise ValueError("Can't get index along diagonal")

		temp = indices2
		indices2 = np.where(swapped, indices1, indices2)
		indices1 = np.where(swapped, temp, indices1)

	return indices1, indices2


def mat_idx_to_triu(row, col, n):
	"""Convert two-dimensional index to linear index of upper triangle.

	If col < row (lower triangle), they will be swapped.

	Function is vectorized, all arguments may be arrays.

	:param int row: Row index in matrix.
	:param int col: Column index in matrix.
	:param int n: Size of matrix.
	:rtype int:
	:raises ValueError: If row and column are equal (does not correspond to an
		element in the upper triangle).
	"""

	row, col = _ensure_order(row, col)
	return nb_mat_idx_to_triu(row, col, n)


def mat_idx_to_tril(row, col, n):
	"""Convert two-dimensional index to linear index of lower triangle.

	If col > row (upper triangle), they will be swapped.

	Function is vectorized, all arguments may be arrays.

	:param int row: Row index in matrix.
	:param int col: Column index in matrix.
	:param int n: Size of matrix.
	:rtype int:
	:raises ValueError: If row and column are equal (does not correspond to an
		element in the lower triangle).
	"""

	col, row = _ensure_order(col, row)
	return nb_mat_idx_to_tril(row, col, n)


@nb.jit(nopython=True)
def triu_idx_to_mat(i, n):
	"""Convert linear index of upper triangle to two-dimensional index.

	:param int i: Linear index of element in upper triangle.
	:param int n: Size of matrix.
	:returns: 2-tuple of ``(row, col)`` indices.
	:rtype: tuple
	"""
	tn = tri_n(n - 1)
	root, rem = tri_root_rem(tn - i - 1)
	return n - root - 2, n - rem - 1


@nb.jit(nopython=True)
def tril_idx_to_mat(i, n):
	"""Convert linear index of lower triangle to two-dimensional index.

	:param int i: Linear index of element in lower triangle.
	:param int n: Size of matrix.
	:returns: 2-tuple of ``(row, col)`` indices.
	:rtype: tuple
	"""
	root, rem = tri_root_rem(i)
	return root + 1, rem


@nb.jit(nopython=True)
def nb_matrix_to_triu(matrix, out):
	"""Numba implementation of matrix_to_tri() for ``upper=True.``

	:param matrix: Square array.
	:type matrix: numpy.ndarray.
	:param out: Linear array of correct length to write values to.
	:type matrix: numpy.ndarray
	"""
	N = matrix.shape[0]

	i = 0
	for row in range(N):
		for col in range(row + 1, N):
			out[i] = matrix[row, col]
			i += 1


@nb.jit(nopython=True)
def nb_matrix_to_tril(matrix, out):
	"""Numba implementation of matrix_to_tri() for ``upper=False.``

	:param matrix: Square array.
	:type matrix: numpy.ndarray.
	:param out: Linear array of correct length to write values to.
	:type matrix: numpy.ndarray
	"""
	N = matrix.shape[0]

	i = 0
	for row in range(N):
		for col in range(row):
			out[i] = matrix[row, col]
			i += 1


def matrix_to_tri(matrix, out=None, upper=False):
	"""Get flattened lower/upper triangle of a square matrix.

	Output will be an array containing the triangle's elements in "row-major"
	order, that is, first the elements of the 0th row in the specified triangle,
	then the 1st row, etc.

	:param matrix: Square array.
	:type matrix: numpy.ndarray.
	:param out: Existing array to write to, if any. Must be 1D with the correct
		number of elements (``tri_n(N - 1)`` where ``N`` is the size of the
		matrix). If None one will be created with the same data type as
		``matrix``.
	:type out: numpy.ndarray
	:param bool upper: If True get upper triangle of matrix, otherwise get
		lower triangle.
	:returns: 1D array containing the specified triangle of the matrix. Will be
		the same array as ``out`` if it was given.
	:rtype: numpy.ndarray
	"""

	N = matrix.shape[0]
	tri_len = tri_n(N - 1)

	if out is None:
		out = np.empty(tri_len, dtype=matrix.dtype)
	elif out.shape != (tri_len,):
		raise ValueError('"out" has incorrect shape')

	if upper:
		nb_matrix_to_triu(matrix, out)
	else:
		nb_matrix_to_tril(matrix, out)

	return out


@nb.jit(nopython=True)
def nb_triu_to_matrix(array, diag, out):
	"""Numba implementation of tri_to_matrix() for upper=True."""
	N = tri_root(len(array)) + 1

	i = 0
	for row in range(N):
		out[row, row] = diag
		for col in range(row + 1, N):
			out[row, col] = out[col, row] = array[i]
			i += 1


@nb.jit(nopython=True)
def nb_tril_to_matrix(array, diag, out):
	"""Numba implementation of tri_to_matrix() for upper=False."""
	N = tri_root(len(array)) + 1

	i = 0
	for row in range(N):
		out[row, row] = diag
		for col in range(row):
			out[row, col] = out[col, row] = array[i]
			i += 1


def tri_to_matrix(array, diag=0, out=None, upper=False):
	"""Convert flattened lower/upper triangle to full symmetrical square matrix.

	:param array: 1D array containing elements of matrix's lower/upper triangle,
		in same format as output of :func:`.matrix_to_tri`.
	:type array: numpy.ndarray
	:param diag: Number to fill diagonal with.
	:param out: Existing array to write to, if any. Must be square with the
		correct number of elements. If None one will be created with the same
		data type as ``array``.
	:type out: numpy.ndarray
	:param bool upper: Whether ``array`` contains the upper (True) or lower
		(False) triangle of the matrix.
	:returns: Full matrix. Will be the same array as ``out`` if it was given.
	:rtype: numpy.ndarray
	"""
	N = tri_root(len(array)) + 1

	if out is None:
		out = np.zeros((N, N), dtype=array.dtype)
	elif out.shape != (N, N):
		raise ValueError('"out" has incorrect shape')

	if upper:
		nb_triu_to_matrix(array, diag, out)
	else:
		nb_tril_to_matrix(array, diag, out)

	return out
