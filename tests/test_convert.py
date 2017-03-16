"""Test pwtools.convert."""

import pytest
import numpy as np

import pwtools as pw


def test_tri():
	"""Test tri_n and tri_root."""

	# Just a large range of values
	for n in range(0, 1000, 10):

		t_n = pw.tri_n(n)

		# Check by summing a range
		assert t_n == sum(range(1, n + 1))

		# Check root finding
		assert pw.tri_root(t_n) == n

		# Check root finding with remainder
		for i in range(n):
			assert pw.tri_root_rem(t_n + i) == (n, i)


def test_array_conversion(indices, index_matrix, upper):
	"""Test conversion between matrix and flattened triangle arrays."""

	mat_vals = pw.tri_to_matrix(indices, upper=upper)
	assert np.array_equal(mat_vals, index_matrix)
	assert mat_vals.dtype == indices.dtype

	tri_vals = pw.matrix_to_tri(index_matrix, upper=upper)
	assert np.array_equal(indices, tri_vals)
	assert tri_vals.dtype == index_matrix.dtype


@pytest.mark.parametrize('diag', [0, 1])
def test_matrix_diag(upper, diag):
	"""Test the diag argument of tri_to_matrix()."""

	n = 10

	tri_vals = np.zeros(pw.tri_n(n - 1))

	matrix = pw.tri_to_matrix(tri_vals, diag=diag, upper=upper)

	assert np.all(np.diagonal(matrix) == diag)


def test_index_conversion(n, index_matrix, upper):
	"""Test conversion between matrix and flattened triangle indices."""

	mat_to_tri = pw.mat_idx_to_triu if upper else pw.mat_idx_to_tril
	tri_to_mat = pw.triu_idx_to_mat if upper else pw.tril_idx_to_mat

	for row in range(n):

		if upper:
			col_range = range(row + 1, n)
		else:
			col_range = range(0, row)

		for col in (col_range):

			# Test (row, col) -> linear matches index from numpy method
			idx = index_matrix[row, col]
			assert mat_to_tri(row, col, n) == idx

			# Test reverse
			assert tri_to_mat(idx, n) == (row, col)

			# Try with row/column swapped
			assert mat_to_tri(col, row, n) == idx
