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


@pytest.fixture(params=[0, 1])
def diag_val(request):
	"""Diagonal value of matrix."""
	return request.param


@pytest.fixture()
def index_matrix(n, indices, upper, diag_val):
	"""Full 2D test matrix containing indices in upper/lower triangle."""

	matrix = np.zeros((n, n), dtype=indices.dtype)

	# Use numpy functions because we want it to match the indexing order.
	if upper:
		rows, cols = np.triu_indices(n, k=1)
	else:
		rows, cols = np.tril_indices(n, k=-1)

	matrix[rows, cols] = indices
	matrix[cols, rows] = indices

	np.fill_diagonal(matrix, diag_val)

	assert np.array_equal(matrix, matrix.T)

	return matrix


@pytest.fixture()
def rand_matrix(n, indices):
	"""Random symmetric matrix with zero diagonal."""

	dtype = indices.dtype
	random = np.random.RandomState(0)

	if dtype.kind in 'iu':
		matrix = random.randint(100, size=(n, n)).astype(dtype=dtype)
	else:
		matrix = random.rand(n, n).astype(dtype=dtype)

	matrix += matrix.T.copy()

	np.fill_diagonal(matrix, 0)

	return matrix
