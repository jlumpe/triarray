"""Common fixtures for all tests."""


import pytest
import numpy as np

import pwtools as pw


@pytest.fixture(params=[10, 30, 100])
def n(request):
	"""Size of test matrix."""
	return request.param


@pytest.fixture(params=[False, True])
def upper(request):
	"""Whether to use upper triangle of matrix (True) or lower (False)."""
	return request.param


@pytest.fixture(params=['i4', 'i8', 'f4', 'f8', 'u4', 'u8'])
def dtype(request):
	"""Data type of test matrix."""
	return np.dtype(request.param)


@pytest.fixture()
def indices(n, dtype):
	"""Indices of each element in upper/lower triangle of test matrix."""
	size = pw.tri_n(n - 1)
	return np.arange(size, dtype=dtype)


@pytest.fixture()
def index_matrix(n, indices, upper):
	"""Full 2D test matrix containing indices in upper/lower triangle."""

	matrix = np.zeros((n, n), dtype=indices.dtype)

	# Use numpy functions because we want it to match the indexing order.
	if upper:
		rows, cols = np.triu_indices(n, k=1)
	else:
		rows, cols = np.tril_indices(n, k=-1)

	matrix[rows, cols] = indices
	matrix[cols, rows] = indices

	assert np.array_equal(matrix, matrix.T)

	return matrix
