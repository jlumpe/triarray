"""Convert between 2d and condensed triangular arrays."""

import numpy as np
import numba as nb


from .math import tri_n, tri_root


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
