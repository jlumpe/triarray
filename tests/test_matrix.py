"""Test pwtools.matrix."""

import numpy as np
import pytest

import pwtools as pw


def check_getitem_same(mat1, mat2, index):
	"""Check indexing returns the same results between two matrix objects."""
	assert np.array_equal(mat1[index], mat2[index])


@pytest.fixture()
def trimatrix(indices, upper, diag_val):
	"""TriMatrix that should correspond to the index_matrix fixture."""
	return pw.TriMatrix(indices, upper=upper, diag_val=diag_val)


def test_attrs(n, indices, trimatrix, upper, diag_val):
	"""Test basic attributes."""
	assert trimatrix.size == n
	assert trimatrix.upper == upper
	assert trimatrix.diag_val == diag_val

	# Check array uses same memory
	# Not sure what trickery might happen with Numba but I think comparing the
	# memoryview objects in the data attribute should work fine
	assert indices.data == trimatrix.array.data


def test_invalid_array_size():
	"""Test constructor with invalid array size."""
	with pytest.raises(ValueError):
		pw.TriMatrix(np.arange(11))


def test_index_conversion(trimatrix, index_matrix):

	for row in range(trimatrix.size):
		for col in range(trimatrix.size):
			if row == col:

				# Can't get index along diagonal
				with pytest.raises(ValueError):
					trimatrix.flat_index(row, col)

			else:

				idx = trimatrix.flat_index(row, col)

				assert trimatrix.array[idx] == index_matrix[row, col]
				assert trimatrix.flat_index(row, col) == idx


def test_to_array(trimatrix, index_matrix):
	"""Test conversion to array using method and various indices."""
	assert np.array_equal(trimatrix.to_array(), index_matrix)
	assert np.array_equal(trimatrix[()], index_matrix)
	assert np.array_equal(trimatrix[:], index_matrix)
	assert np.array_equal(trimatrix[:, :], index_matrix)


def test_getitem_single(trimatrix, index_matrix, diag_val):
	"""Test getting a single element from the matrix."""

	for row in range(trimatrix.size):
		for col in range(trimatrix.size):
			assert trimatrix[row, col] == index_matrix[row, col]
			assert trimatrix.get_item(row, col) == index_matrix[row, col]


def test_get_row_single(trimatrix, index_matrix):
	"""Test various methods of getting single rows."""

	out = np.zeros(trimatrix.size, dtype=index_matrix.dtype)

	for row in range(trimatrix.size):

		row_vals = index_matrix[row]

		assert np.array_equal(trimatrix[row], row_vals)
		assert np.array_equal(trimatrix[row, :], row_vals)
		assert np.array_equal(trimatrix[:, row], row_vals)
		assert np.array_equal(trimatrix.get_row(row), row_vals)

		trimatrix.get_row(row, out=out)
		assert np.array_equal(out, row_vals)


def test_get_row_array(n, trimatrix, index_matrix):
	"""Test getting many rows by indexing with single integer array."""

	def check_rows(rows):
		check_getitem_same(trimatrix, index_matrix, rows)
		check_getitem_same(trimatrix, index_matrix, (rows, slice(None)))
		check_getitem_same(trimatrix, index_matrix, (slice(None), rows))

	# 1D array - every 10th row
	step = int(np.ceil(n / 10))
	rows = np.arange(0, n, step)
	check_rows(rows)

	# 2D array
	rows = np.arange(17 * 17).reshape(17, 17)
	rows = (rows * 11) % n
	check_rows(rows)

	# Degenerate case of empty 1D array
	rows = np.arange(0)
	check_rows(rows)


def test_getitem_scalar_array(n, trimatrix, index_matrix):
	"""Check indexing with a single integer and an array of integers."""

	def check_rows(rows):
		check_getitem_same(trimatrix, index_matrix, rows)
		check_getitem_same(trimatrix, index_matrix, (rows, slice(None)))
		check_getitem_same(trimatrix, index_matrix, (slice(None), rows))

	# 1D array - every 10th row
	step = int(np.ceil(n / 10))
	rows = np.arange(0, n, step)
	check_rows(rows)

	# 2D array
	rows = np.arange(17 * 17).reshape(17, 17)
	rows = (rows * 11) % n
	check_rows(rows)

	# Degenerate case of empty 1D array
	rows = np.arange(0)
	check_rows(rows)


def test_invalid_index(trimatrix):
	"""Test various invalid indices."""

	# Too many
	with pytest.raises(ValueError):
		trimatrix[:, :, :]

	# Float scalar
	with pytest.raises(TypeError):
		trimatrix[0.5]

	# Float array
	with pytest.raises(ValueError):
		trimatrix[np.linspace(0, 10)]


def test_index_out_of_range(trimatrix):
	"""Test row/column indices out of range result in an exception."""

	def check_bad_index(*index):
		with pytest.raises(ValueError) as exc_info:
			trimatrix[tuple(index)]
		assert str(exc_info.value) == 'Index out of range'

	full_slice = slice(None)
	valid_array = np.arange(0, trimatrix.size, 5)

	for bad_int in (-1, trimatrix.size):

		bad_array = valid_array.copy()
		bad_array.flat[-1] = bad_int

		for bad_index in [bad_int, bad_array]:

			check_bad_index(bad_index)

			check_bad_index(bad_index, 0)
			check_bad_index(0, bad_index)

			check_bad_index(bad_index, full_slice)
			check_bad_index(full_slice, bad_index)

			check_bad_index(bad_index, valid_array)
			check_bad_index(valid_array, bad_index)

			check_bad_index


def test_get_partial_row(trimatrix, index_matrix, upper):
	"""Check get_partial_row() and iter_partial_rows() methods."""

	def get_partial_row(row):
		return index_matrix[row, row + 1:] if upper else index_matrix[row, :row]

	for i in range(trimatrix.size):
		assert np.array_equal(trimatrix.get_partial_row(i), get_partial_row(i))

	for i, row in enumerate(trimatrix.iter_partial_rows()):
		assert np.array_equal(row, get_partial_row(i))


def test_index_int_types(trimatrix, index_matrix):
	"""
	Test indexing with Python integers and numpy integer objects of various
	types.
	"""

	# TODO - uint64 can't be converted to intp
	for np_type in (np.int32, np.uint32, np.int64):

		for i in range(trimatrix.size):

			assert trimatrix[0, i] == trimatrix[np_type(0), np_type(i)]

		indices = np.arange(trimatrix.size).astype(np_type)

		assert np.array_equal(trimatrix[0, indices], index_matrix[0, indices])
