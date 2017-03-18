"""Test triarray.io."""

import io

import pytest
import numpy as np

import triarray as tri


@pytest.fixture(params=[False, True])
def fobj(request, tmpdir, rand_matrix, upper):
	"""Open writeable binary file-like object with matrix written to it.

	True parameter for actual file, otherwise a BytesIO object.
	"""

	if request.param:
		fobj = open(tmpdir.join('tri_data'), 'wb+')
	else:
		fobj = io.BytesIO()

	tri.write_tri_file(fobj, rand_matrix, upper=upper)
	fobj.seek(0)

	yield fobj

	fobj.close()


@pytest.fixture()
def keep(n):
	"""Some non-regular pattern of rows to keep."""
	random = np.random.RandomState(0)
	return random.rand(n) < .75


def check_rows(row_iter, matrix, upper):
	"""Check rows from read_tri_file_rows() match matrix."""

	n = matrix.shape[0]
	dtype = matrix.dtype

	# Row to start at
	expected_row = 0 if upper else 1

	for row, row_vals in row_iter:

		# Check rows are sequential
		assert row == expected_row
		expected_row += 1

		# Check values are correct
		if upper:
			expected_vals = matrix[row, row + 1:]
		else:
			expected_vals = matrix[row, :row]

		assert np.array_equal(row_vals, expected_vals)

		# Check data type
		assert row_vals.dtype == dtype

	# Check we went through all the rows
	assert row == (n - 2 if upper else n - 1)


def test_read_rows(rand_matrix, fobj, upper):
	"""Test iterating over stored rows with read_tri_file_rows()."""

	n = rand_matrix.shape[0]
	dtype = rand_matrix.dtype

	row_iter = tri.read_tri_file_rows(fobj, n, dtype=dtype, upper=upper)

	check_rows(row_iter, rand_matrix, upper)


def test_read_rows_subset(rand_matrix, fobj, upper, keep):
	"""Test read_tri_file_rows() with a subset of rows."""

	n = rand_matrix.shape[0]

	matrix_subset = rand_matrix[np.ix_(keep, keep)]

	row_iter = tri.read_tri_file_rows(fobj, keep=keep, dtype=rand_matrix.dtype,
	                                 upper=upper)

	check_rows(row_iter, matrix_subset, upper)


@pytest.mark.parametrize('diag', [None, 0, 1])
def test_read_matrix(rand_matrix, fobj, upper, diag):
	"""Test reading full matrix with read_tri_file()."""

	n = rand_matrix.shape[0]
	dtype = rand_matrix.dtype

	read = tri.read_tri_file(fobj, n=n, dtype=dtype, upper=upper, diag=diag)

	# Check diagonal first
	assert np.all(np.diag(read) == (0 if diag is None else diag))

	# Set diagonal to match matrix before checking
	np.fill_diagonal(read, np.diagonal(rand_matrix))

	assert np.array_equal(rand_matrix, read)
	assert rand_matrix.dtype == dtype


@pytest.mark.parametrize('diag', [None, 0, 1])
def test_read_matrix_subset(rand_matrix, fobj, upper, keep, diag):
	"""Test read_tri_file() with a subset of rows."""

	n = rand_matrix.shape[0]
	dtype = rand_matrix.dtype

	matrix_subset = rand_matrix[np.ix_(keep, keep)]

	read = tri.read_tri_file(fobj, keep=keep, dtype=dtype, upper=upper, diag=diag)

	# Check diagonal first
	assert np.all(np.diag(read) == (0 if diag is None else diag))

	# Set diagonal to match matrix before checking
	np.fill_diagonal(read, np.diagonal(matrix_subset))

	assert np.array_equal(matrix_subset, read)
	assert matrix_subset.dtype == dtype


@pytest.mark.parametrize('new_dtype', ['i4', 'i8', 'f4', 'f8'])
def test_write_dtype(rand_matrix, upper, new_dtype):
	"""Test write_tri_file with different dtype arguments."""

	n = rand_matrix.shape[0]

	new_dtype = np.dtype(new_dtype)

	mat_as_type = rand_matrix.astype(new_dtype)
	
	with io.BytesIO() as fobj:

		tri.write_tri_file(fobj, rand_matrix, upper=upper, dtype=new_dtype)

		fobj.seek(0)
		read = tri.read_tri_file(fobj, n, dtype=new_dtype, upper=upper)

		assert read.dtype == new_dtype

		assert np.array_equal(read, mat_as_type)
