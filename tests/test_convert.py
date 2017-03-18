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


def test_squareform_conversion(n):
	"""Test vs. scipy's squareform() function."""

	try:
		from scipy.spatial.distance import squareform
	except ImportError:
		pytest.skip('Could not import scipy')

	indices = np.arange(pw.tri_n(n - 1), dtype=np.double)
	matrix = squareform(indices)

	assert np.array_equal(pw.tri_to_matrix(indices, upper=True), matrix)
