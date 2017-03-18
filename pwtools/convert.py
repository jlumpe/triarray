"""Convert between 2d and flattened triangular arrays and indices."""

import numpy as np
import numba as nb


from .math import tri_n, tri_root, tri_root_rem


@nb.vectorize([nb.intp(nb.intp, nb.intp, nb.intp)], nopython=True)
def mat_idx_to_tril_fast(row, col, n):
	"""mat_idx_to_tril_fast(row, col, n)

	Convert two-dimensional index to linear index of lower triangle.

	This is the fast implementation, which does not check the order of row and
	col.

	:param int row: Matrix row index.
	:param int col: Matrix column index. Must be less than row.
	:param int n: Matrix size.
	:returns: Linear index of position in lower triangle.
	:rtype int:
	"""
	return tri_n(row - 1) + col


@nb.vectorize([nb.intp(nb.intp, nb.intp, nb.intp)], nopython=True)
def mat_idx_to_tril(row, col, n):
	"""mat_idx_to_tril(row, col, n)

	Convert two-dimensional index to linear index of lower triangle.

	This is the safe implementation, which checks row and col are not equal and
	swaps them if they are not in the correct order.

	:param int row: Matrix row index.
	:param int col: Matrix column index. Must not be equal to row.
	:param int n: Matrix size.
	:returns: Linear index of position in lower triangle.
	:rtype int:
	"""
	if col == row:
		raise ValueError('Cannot get index along diagonal')

	if row > col:
		return mat_idx_to_tril_fast(row, col, n)
	else:
		return mat_idx_to_tril_fast(col, row, n)


@nb.vectorize([nb.intp(nb.intp, nb.intp, nb.intp)], nopython=True)
def mat_idx_to_triu_fast(row, col, n):
	"""mat_idx_to_triu_fast(row, col, n)

	Convert two-dimensional index to linear index of upper triangle.

	This is the fast implementation, which does not check the order of row and
	col.

	:param int row: Matrix row index.
	:param int col: Matrix column index. Must be greater than row.
	:param int n: Matrix size.
	:returns: Linear index of position in upper triangle.
	:rtype int:
	"""
	# Messy but fast implementation
	return (2 * n - row - 3) * row // 2 + col - 1


@nb.vectorize([nb.intp(nb.intp, nb.intp, nb.intp)], nopython=True)
def mat_idx_to_triu(row, col, n):
	"""mat_idx_to_triu(row, col, n)

	Convert two-dimensional index to linear index of upper triangle.

	This is the safe implementation, which checks row and col are not equal and
	swaps them if they are not in the correct order.

	:param int row: Matrix row index.
	:param int col: Matrix column index. Must not be equal to row.
	:param int n: Matrix size.
	:returns: Linear index of position in upper triangle.
	:rtype int:
	"""
	if col == row:
		raise ValueError('Cannot get index along diagonal')

	if row < col:
		return mat_idx_to_triu_fast(row, col, n)
	else:
		return mat_idx_to_triu_fast(col, row, n)


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
def matrix_to_tril(matrix, out):
	"""Get flattened lower triangle of a square matrix.

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


@nb.jit(nopython=True)
def matrix_to_triu(matrix, out):
	"""Get flattened upper triangle of a square matrix.

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
		matrix_to_triu(matrix, out)
	else:
		matrix_to_tril(matrix, out)

	return out


@nb.jit(nopython=True)
def tril_to_matrix(array, diag, out):
	"""Convert flattened lower triangle to full symmetrical square matrix.

	:param array: 1D array containing elements of matrix's lower triangle, in
		same format as output of :func:`.matrix_to_tril`.
	:type array: numpy.ndarray
	:param diag: Number to fill diagonal with.
	:param out: Existing array to write to. Must be square with the	correct
		number of elements.
	:type out: numpy.ndarray
	"""

	N = tri_root(len(array)) + 1

	i = 0
	for row in range(N):
		out[row, row] = diag
		for col in range(row):
			out[row, col] = out[col, row] = array[i]
			i += 1


@nb.jit(nopython=True)
def triu_to_matrix(array, diag, out):
	"""Convert flattened upper triangle to full symmetrical square matrix.

	:param array: 1D array containing elements of matrix's upper triangle, in
		same format as output of :func:`.matrix_to_triu`.
	:type array: numpy.ndarray
	:param diag: Number to fill diagonal with.
	:param out: Existing array to write to. Must be square with the	correct
		number of elements.
	:type out: numpy.ndarray
	"""

	N = tri_root(len(array)) + 1

	i = 0
	for row in range(N):
		out[row, row] = diag
		for col in range(row + 1, N):
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
		triu_to_matrix(array, diag, out)
	else:
		tril_to_matrix(array, diag, out)

	return out
