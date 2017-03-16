"""Read/write matrices to disk in condensed format."""

import io

import numpy as np

from .util import get_tqdm


def write_tri_file(fobj, matrix, upper=False, dtype=None):
	"""Write the lower/upper triangle of a matrix to a binary file object.

	:param fobj: Writable file-like object in binary mode.
	:param matrix: Square matrix.
	:type matrix: numpy.ndarray
	:param bool upper: Whether to write the upper (True) or lower (False)
		portion of the matrix.
	:param dtype: Numpy data type to write as. If None will use the data type
		of ``matrix``.
	"""

	n = matrix.shape[0]
	if matrix.shape != (n, n):
		raise ValueError('Matrix is not square')

	if dtype is not None:
		dtype = np.dtype(dtype)
		if dtype == matrix.dtype:
			dtype = None

	for row in range(n):

		if upper:
			row_vals = matrix[row, row + 1:]
		else:
			row_vals = matrix[row, :row]

		if row_vals.size == 0:
			continue

		if dtype is not None:
			row_vals = row_vals.astype(dtype)

		fobj.write(row_vals.tobytes())


def read_tri_file_rows(fobj, n=None, dtype='f8', upper=False, keep=None,
                       progress=False):
	"""
	Iterate over partial rows of a matrix stored in a file in flattened
	triangular format.

	Given a file containing distance matrix data in the non-redundant form
	created by :func:`.write_tri_file`, yields each partial row of the matrix
	stored in the file. These will be the portions of the rows to the left of
	the diagonal if ``upper`` is False or to the right if ``upper`` is True.
	Row portions of zero length will be skipped.

	:param fobj: Readable, seekable file-like object in binary mode.
	:param int n: Size of full matrix stored in file. Exactly one of ``n`` or
		``keep`` should be given.
	:param dtype: Numpy data type of the stored data.
	:type dtype: numpy.dtype
	:param bool upper: Whether the file contains the upper (True) or lower
		(False) triangle of the matrix. Should match value used when file was
		created by :func:`write_tri_file`.
	:param keep: If given, subset of rows/columns of matrix to pull from file.
		Should be a boolean array with length matching the size of the full
		stored matrix, with ``False`` values indicating a row/column should be
		skipped. The returned values will then be filtered to include only
		these row/columns of the full matrix. Exactly one of ``n`` or ``keep``
		should be given.
	:type keep: numpy.ndarray
	:param progress: If True will display a progress bar with tqdm, using either
		the standard or notebook version depending on whether the code is being
		executed in a Jupyter notebook or not. You may also pass any other
		function taking an iterable as an argument and returning another
		iterable in the same manner as :func:`tqdm.tqdm`.

	:returns: Generator yielding 2-tuples of ``int`` and :class:`numpy.ndarray`,
		the row index and row values. If ``keep`` was given instead of ``n`` the
		row indices will be for the filtered matrix, not the full matrix.
	"""

	if keep is not None:
		if n is not None:
			raise TypeError('Must give only one of "n" and "keep"')
		n = keep.size

	elif n is None:
		raise TypeError('Should give one of "n" or "keep"')

	# Use the appropriate flavor of tqdm if progress is True
	if progress is True:
		progress = get_tqdm()

	# Iterator over rows - wrap in progress if needed
	rowiter = range(n)
	if progress:
		rowiter = progress(rowiter)

	sub_row = 0
	for row in rowiter:

		# Range of columns to read
		col_begin = row + 1 if upper else 0
		col_end = n if upper else row

		row_len = col_end - col_begin

		if keep is not None and not keep[row]:
			# Row to be skipped, seek ahead to the next
			fobj.seek(row_len * dtype.itemsize, io.SEEK_CUR)

		else:

			if row_len > 0:
				# Read in all stores values of row
				data = fobj.read(row_len * dtype.itemsize)
				row_vals = np.frombuffer(data, dtype=dtype, count=row_len)

				if keep is None:
					# Yield all columns
					yield sub_row, row_vals

				else:
					# Yield subset of columns
					keep_row = keep[col_begin:col_end]
					if keep_row.sum() > 0:
						yield sub_row, row_vals[keep_row]

			sub_row += 1


def read_tri_file(fobj, n=None, dtype='f8', upper=False, keep=None,
                  diag=None, out=None, **kwargs):
	"""Read a full matrix from a file created by :func:`.write_tri_file`.

	:param fobj: Readable, seekable file-like object in binary mode.
	:param int n: Size of full matrix stored in file. Exactly one of ``n`` or
		``keep`` should be given.
	:param dtype: Numpy data type of the stored data.
	:type dtype: numpy.dtype
	:param bool upper: Whether the file contains the upper (True) or lower
		(False) triangle of the matrix. Should match value used when file was
		created by :func:`write_tri_file`.
	:param keep: If given, subset of rows/columns of matrix to pull from file.
		Should be a boolean array with length matching the size of the full
		stored matrix, with ``False`` values indicating a row/column should be
		skipped. ``read_tri_file(..., keep=keep)`` should be identical to
		``read_tri_file(...)[np.ix_(keep, keep)]``.Exactly one of ``n`` or
		``keep`` should be given.
	:type keep: numpy.ndarray
	:param diag: Value to fill diagonal with. If None and ``out`` is given will
		keep existing diagonal values in matrix, otherwise if ``out`` is omitted
		will be zero.
	:param out: Square array to write matrix values to. Should be of correct
		shape (``(n, n)`` where ``n`` is given explicitly or otherwise
		``n = sum(keep)``.
	:type out: numpy.ndarray
	:param \\**kwargs: Additional keyword arguments to
		:func:`.read_tri_file_rows`.

	:rtype: numpy.ndarray
	:returns: Square array containing matrix values. If ``out`` was given will
		be the same array, otherwise a new one will be created with the
		appropriate data type.
	"""

	if keep is not None:
		if n is not None:
			raise TypeError('Must give only one of "n" and "keep"')
		n = np.sum(keep)

	elif n is None:
		raise TypeError('Should give one of "n" or "keep"')

	# Create destination array if necessary
	if out is None:
		out = np.zeros((n, n), dtype=dtype)
	elif out.shape != (n, n):
		raise ValueError('"out" does not have the expected shape.')

	# Pass correct arguments to read_tri_file_rows()
	read_args = dict(fobj=fobj, dtype=dtype, upper=upper, **kwargs)
	if keep is None:
		read_args['n'] = n
	else:
		read_args['keep'] = keep

	# Read in rows
	for row, row_vals in read_tri_file_rows(**read_args):

		col_slice = slice(row + 1, None) if upper else slice(row)

		out[row, col_slice] = row_vals
		out[col_slice, row] = row_vals

	# Fill diagonal
	if diag is not None:
		np.fill_diagonal(out, diag)

	return out
