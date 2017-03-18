"""Test pwtools.index."""

import pytest
import numpy as np

import pwtools as pw


def test_convert_scalar(n, index_matrix, upper):
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


def test_convert_array(n, upper):
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
