"""Test pwtools.convert."""

import pytest
import numpy as np

import pwtools as pw


def test_array_conversion(indices, index_matrix, upper, diag_val):
	"""Test conversion between matrix and flattened triangle arrays."""

	mat_vals = pw.tri_to_matrix(indices, upper=upper, diag=diag_val)
	assert np.array_equal(mat_vals, index_matrix)
	assert mat_vals.dtype == indices.dtype

	tri_vals = pw.matrix_to_tri(index_matrix, upper=upper)
	assert np.array_equal(indices, tri_vals)
	assert tri_vals.dtype == index_matrix.dtype


def test_index_conversion_scalar(n, index_matrix, upper):
	"""Test conversion between matrix and flattened triangle scalar indices."""

	mat_to_tri = pw.mat_idx_to_triu if upper else pw.mat_idx_to_tril
	mat_to_tri_fast = pw.mat_idx_to_triu_fast if upper else pw.mat_idx_to_tril_fast
	tri_to_mat = pw.triu_idx_to_mat if upper else pw.tril_idx_to_mat

	for row in range(n):

		for col in range(n):

			if row == col:
				# Test getting along diagonal raises exception
				with pytest.raises(ValueError):
					mat_to_tri(row, col)

			else:

				# Test (row, col) -> linear matches index from numpy method
				idx = int(index_matrix[row, col])
				assert mat_to_tri(row, col, n) == idx

				# If on the correct triangle
				if (row < col) == upper:

					# Test fast version
					assert mat_to_tri_fast(row, col, n) == idx

					# Test reverse
					assert tri_to_mat(idx, n) == (row, col)


def test_index_conversion_array(n, upper):
	"""Test conversion between matrix and flattened triangle array indices."""

	indices = np.arange(pw.tri_n(n - 1))

	if upper:
		rows, cols = np.triu_indices(n, k=1)

		calc_indices = pw.mat_idx_to_triu(rows, cols, n)

		calc_rows, calc_cols = pw.triu_idx_to_mat(indices, n)

	else:
		rows, cols = np.tril_indices(n, k=-1)

		calc_indices = pw.mat_idx_to_tril(rows, cols, n)

		calc_rows, calc_cols = pw.tril_idx_to_mat(indices, n)

	assert np.array_equal(calc_indices, indices)

	assert np.array_equal(calc_rows, rows)
	assert np.array_equal(calc_cols, cols)


def test_squareform_conversion(n):
	"""Test vs. scipy's squareform() function."""

	try:
		from scipy.spatial.distance import squareform
	except ImportError:
		pytest.skip('Could not import scipy')

	indices = np.arange(pw.tri_n(n - 1), dtype=np.double)
	matrix = squareform(indices)

	assert np.array_equal(pw.tri_to_matrix(indices, upper=True), matrix)
