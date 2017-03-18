"""Convert between 2d and condensed triangular indices."""

import numba as nb


from .math import tri_n, tri_root_rem


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
