"""Simulate full symmetric matrices while storing only non-redundant data."""

from types import MappingProxyType
from enum import Enum

import numpy as np
import numba as nb


from .math import tri_n, tri_root


# Tuple of bases for builtin/numpy integer python types
INTEGER_TYPES = (int, np.integer)


# Mapping from numpy dtype to numba type objects for supported data types
NUMBA_TYPES = MappingProxyType({
	np.dtype(s): getattr(nb, s)
	for s in
	['i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8', 'f4', 'f8']
})


class NbTriMatrixBase:
	"""Base for Numba jitclass which implements most of :class:`.TriMatrix`.

	Numba doesn't support subclassing a jitclass, so the inheritance needs to be
	done before the decorator is applied. Child classes will implement methods
	based off whether the lower or upper triangle of the matrix is stored in the
	array.

	This is an abstract base class, but doesn't use :class:`abc.ABCMeta` because
	jitclass doesn't like it. Abstract methods simply raise
	``NotImplementedError``.

	Implementations should set the following attributes:

	.. attribute:: array

		1D array storing condensed triangle of matrix.

	.. attribute:: arraysize

		numba.intp, size of linear array.

	.. attribute:: size

		numba.intp, number of rows/columns in full matrix.

	.. attribute:: upper

		Boolean, whether the array stores the upper (True) or lower (False)
		triangle of the matrix.

	.. attribute:: diag_val

		Value to fill diagonal with.
	"""

	def row_start(self, row):
		"""
		Get index in flat matrix where the stored portion of the matrix row
		starts.

		:param int row: Matrix row index.
		:rtype: int
		"""
		raise NotImplementedError()

	def row_len(self, row):
		"""
		Get the length of the portion of the matrix row that is present in the
		array.

		:param int row: Matrix row index.
		:rtype: int
		"""
		raise NotImplementedError()

	def flat_index(self, row, col):
		"""Convert matrix index to array index.

		Row should be > col for lower triangle, < col for upper triangle. This
		is not actually validated.

		:param int row: Matrix row index.
		:param int col: Matrix column index.
		:rtype: int
		"""
		raise NotImplementedError()

	def get_item(self, row, col):
		"""Get a single element from the full matrix.

		:param int row: Matrix row index.
		:param int col: Matrix column index.
		"""
		if row == col:
			return self.diag_val
		elif (row < col) == self.upper:
			return self.array[self.flat_index(row, col)]
		else:
			return self.array[self.flat_index(col, row)]

	def get_partial_row(self, row):
		"""Get portion of the matrix row stored in the flat array.

		:param int row: Index of row to get.
		:returns: Writable view of portion of flat array.
		:rtype: np.ndarray
		"""
		start = self.row_start(row)
		stop = start + self.row_len(row)
		return self.array[start:stop]

	def get_row(self, row, out):
		"""Fill an array with the values of a row of the full matrix.

		:param int row: Index of row to get.
		:param out: Destination array to write to.
		:type out: numpy.ndarray
		"""
		raise NotImplementedError()

	def to_array(self, out):
		"""Fill a 2D array with the values of the full matrix.

		:param out: Destination array to write to.
		:type out: numpy.ndarray
		"""
		raise NotImplementedError()


class NbTriLMatrixBase(NbTriMatrixBase):
	"""
	Base for Numba jitclass which implements most of :class:`.TriMatrix` using
	the lower triangle.
	"""

	def __init__(self, array, diag_val):

		self.array = array
		self.arraysize = array.shape[0]

		self.size = tri_root(self.arraysize) + 1

		self.diag_val = diag_val

		self.upper = False

	def row_start(self, row):
		return tri_n(row - 1)

	def row_len(self, row):
		return row

	def flat_index(self, row, col):
		return tri_n(row - 1) + col

	def get_row(self, row, out):

		row_start = self.row_start(row)

		for col in range(row):
			out[col] = self.array[row_start + col]

		out[row] = self.diag_val

		for col in range(row + 1, self.size):
			out[col] = self.array[self.flat_index(col, row)]

	def to_array(self, out):

		i = 0
		for row in range(self.size):
			out[row, row] = self.diag_val

			for col in range(row):

				out[row, col] = self.array[i]
				out[col, row] = self.array[i]

				i += 1

		return out


class NbTriUMatrixBase(NbTriMatrixBase):
	"""
	Base for Numba jitclass which implements most of :class:`.TriMatrix` using
	the upper triangle.
	"""

	def __init__(self, array, diag_val):

		self.array = array
		self.arraysize = array.shape[0]

		self.size = tri_root(self.arraysize) + 1

		self.diag_val = diag_val

		self.upper = True

	def row_start(self, row):
		return self.arraysize - tri_n(self.size - row - 1)

	def row_len(self, row):
		return self.size - row - 1

	def flat_index(self, row, col):
		return self.row_start(row) + col - row - 1

	def get_row(self, row, out):

		for i in range(row):
			out[i] = self.array[self.flat_index(i, row)]

		out[row] = self.diag_val

		row_start = self.row_start(row)
		for i in range(self.size - row - 1):
			out[row + i + 1] = self.array[row_start + i]

		return out

	def to_array(self, out):

		i = 0
		for row in range(self.size):
			out[row, row] = self.diag_val

			for col in range(row + 1, self.size):

				out[row, col] = self.array[i]
				out[col, row] = self.array[i]

				i += 1

		return out


def _make_jitclass_for_type(base, nbtype):
	"""
	Apply jitclass to one of the base classes with the signature for the given
	array data type.
	"""

	spec = [
		('array', nbtype[:]),
		('arraysize', nb.intp),
		('size', nb.intp),
		('upper', nb.boolean),
		('diag_val', nbtype)
	]

	return nb.jitclass(spec)(base)


# Jitclasses of NbTriLMatrixBase for each supported array data type
_LOWER_JITCLASS_BY_TYPE = MappingProxyType({
	dtype: _make_jitclass_for_type(NbTriLMatrixBase, nbtype)
	for dtype, nbtype in NUMBA_TYPES.items()
})

# Jitclasses of NbTriUMatrixBase for each supported array data type
_UPPER_JITCLASS_BY_TYPE = MappingProxyType({
	dtype: _make_jitclass_for_type(NbTriUMatrixBase, nbtype)
	for dtype, nbtype in NUMBA_TYPES.items()
})


def _get_jitclass_for_dtype(dtype, upper):
	"""
	Get the correct jitclass of NbTriMatrixBase for the data type and triangle.
	"""
	if upper:
		return _UPPER_JITCLASS_BY_TYPE[dtype]
	else:
		return _LOWER_JITCLASS_BY_TYPE[dtype]


@nb.guvectorize(
	[
		(nbtype[:], nbtype[:], nb.intp[:], nb.intp[:], nbtype[:])
		for nbtype in NUMBA_TYPES.values()
	],
	'(n),(),(),()->()',
	nopython=True,
)
def getitem_advanced_int_lower(array, diag, row, col, out):
	"""getitem_advanced_int_lower(array, diag, row, col)

	Numpy ufunc. Get items from the condensed lower triangle of a symmetric
	matrix using the equivalent of numpy advanced indexing with integer arrays
	on both the rows and columns of the full matrix.

	This is meant to be vectorized on the ``row`` and ``col`` arguments only.

	Row/column indices do not need to fall on the lower triangle (may be on
	diagonal or upper), but are not bounds checked.

	:param array: Source array (1D array even for kernel).
	:param diag: Value to substitute for diagonal elements. Same type as
		``array``.
	:param int row: Row index.
	:param int col: Column index.
	"""
	r = row[0]
	c = col[0]

	if r == c:
		out[0] = diag[0]
	elif r > c:
		out[0] = array[tri_n(r - 1) + c]
	else:
		out[0] = array[tri_n(c - 1) + r]


@nb.guvectorize(
	[
		(nbtype[:], nb.intp[:], nbtype[:], nb.intp[:], nb.intp[:], nbtype[:])
		for nbtype in NUMBA_TYPES.values()
	],
	'(n),(),(),(),()->()',
	nopython=True,
)
def getitem_advanced_int_upper(array, size, diag, row, col, out):
	"""getitem_advanced_int_upper(array, size, diag, row, col)

	Numpy ufunc. Get items from the condensed upper triangle of a symmetric
	matrix using the equivalent of numpy advanced indexing with integer arrays
	on both the rows and columns of the full matrix.

	This is meant to be vectorized on the ``row`` and ``col`` arguments only.

	Row/column indices do not need to fall on the upper triangle (may be on
	diagonal or upper), but are not bounds checked.

	:param array: Source array (1D array even for kernel).
	:param int size: Size of full matrix.
	:param diag: Value to substitute for diagonal elements. Same type as
		``array``.
	:param int row: Row index.
	:param int col: Column index.
	"""
	r = row[0]
	c = col[0]

	if r == c:
		out[0] = diag[0]
	else:
		if r > c:
			r, c = c, r

		i = r * (2 * size[0] - r - 3) // 2 + c - 1

		out[0] = array[i]


@nb.jit(nopython=True)
def check_index_in_range(indices, size):
	"""Check an array of row/column indices are in the correct range.

	:param int indices: Array of integers. Rows or columns in matrix.
	:param int size: Size of matrix.
	:raises ValueError: If index is not >= 0 and < size.
	"""
	for i in indices.flat:
		if i < 0 or i >= size:
			raise ValueError('Index out of range')


# General types accepted as elements of an ndarray index
IndexType = Enum('IndexType', [
	'full_slice',
	'integer_scalar',
	'integer_array',
	'boolean_array',
])


class TriMatrix:
	"""Simulates a symmetric square array using only non-redundant data.

	Indexing this class should return the same result as with the equivalent
	full 2D matrix as a :class:`numpy.ndarray`.

	:param array: One-dimensional :class:`numpy.ndarray` containing condensed
		upper/lower triangle of full matrix. Length must be a triangular number.
	:type array: numpy.ndarray
	:param bool upper: True if ``array`` contains the upper triangle of the
		matrix, False otherwise.
	:param diag_val: Value of diagonal elements of matrix.

	.. attribute:: array

		One-dimensional :class:`numpy.ndarray` containing condensed upper/lower
		triangle of full matrix.

	.. attribute:: size

		Size of full matrix.

	.. attribute:: diag_val

		Value of all diagonal elements in matrix.

	.. attribute:: upper

		True if the upper triangle is stored in :attr:`array`, False if the
		lower triangle is stored.
	"""

	def __init__(self, array, upper=False, diag_val=0):

		NbMatrixClass = _get_jitclass_for_dtype(array.dtype, upper)
		self.nbmatrix = NbMatrixClass(array, diag_val)

		if self.nbmatrix.arraysize != tri_n(self.size - 1):
			raise ValueError('Size of array is not a triangular number')

	@property
	def array(self):
		return self.nbmatrix.array

	@property
	def size(self):
		return self.nbmatrix.size

	@property
	def diag_val(self):
		return self.nbmatrix.diag_val

	@property
	def upper(self):
		return self.nbmatrix.upper

	def _check_bounds(self, index):
		"""Check that a row/column index is within expected bounds."""
		if not 0 <= index < self.size:
			raise ValueError('Index out of range')

	def row_start(self, row):
		"""
		Get index in flat matrix where the stored portion of the matrix row
		starts.

		:param int row: Matrix row index.
		:rtype: int
		"""
		self._check_bounds(row)
		return self.nbmatrix.row_start(row)

	def row_len(self, row):
		"""
		Get the length of the portion of the matrix row that is present in the
		array.

		:param int row: Matrix row index.
		:rtype: int
		"""
		self._check_bounds(row)
		return self.nbmatrix.row_len(row)

	def flat_index(self, row, col):
		"""Convert matrix index to array index.

		:param int row: Matrix row index.
		:param int col: Matrix column index.
		:rtype: int
		"""
		self._check_bounds(row)
		self._check_bounds(col)

		if row == col:
			raise ValueError("Can't get index along diagonal.")

		if (row > col) == self.upper:
			row, col = col, row

		return self.nbmatrix.flat_index(row, col)

	def get_item(self, row, col):
		"""Get a single element from the full matrix.

		:param int row: Matrix row index.
		:param int col: Matrix column index.
		"""
		self._check_bounds(row)
		self._check_bounds(col)
		return self.nbmatrix.get_item(row, col)

	def get_partial_row(self, row):
		"""Get portion of the matrix row stored in the flat array.

		:param int row: Index of row to get.
		:returns: Writable view of portion of flat array.
		:rtype: np.ndarray
		"""
		self._check_bounds(row)
		return self.nbmatrix.get_partial_row(row)

	def get_row(self, row, out=None):
		"""Get a row of the full matrix.

		:param int row: Index of row to get.
		:param out: Destination array to write to, if any.
		:type out: numpy.ndarray
		:returns: numpy.ndarray
		"""
		self._check_bounds(row)

		if out is None:
			out = np.empty(self.size, dtype=self.array.dtype)

		self.nbmatrix.get_row(row, out)

		return out

	def iter_partial_rows(self):
		"""
		Iterate over rows of matrix and yield the slice that is stored in the
		condensed array.

		Equivalent to ``trimat.get_partial_row(i) for i in range(trimat.size)``.

		Yielded arrays are writable slices of :attr:`array`.

		:returns: Generator yielding class:`numpy.ndarray`.
		"""

		start = 0

		for row in range(self.size):

			row_len = self.row_len(row)

			yield self.array[start:start + row_len]

			start += row_len

	def to_array(self, out=None):
		"""Get the full matrix in numpy array format.

		:param out: Destination array to write to, if any.
		:type out: numpy.ndarray
		:returns: 2D array containing full matrix values. Will be the same array
			as ``out`` if it was given.
		:rtype: numpy.ndarray
		"""

		if out is None:
			out = np.empty((self.size, self.size), dtype=self.array.dtype)

		elif out.shape != (self.size, self.size):
			raise ValueError('out has incorrect shape')

		self.nbmatrix.to_array(out)

		return out

	def _process_index_elem(self, item):
		"""
		Validate an index item, return converted item and enum value
		representing item type or raise an exception.

		:returns: (item, item_type) tuple.
		"""

		if isinstance(item, slice):

			if item == slice(None):
				return item, IndexType.full_slice

			else:
				return np.arange(*item.indices(self.size)), IndexType.integer_array

		elif np.isscalar(item):

			if isinstance(item, INTEGER_TYPES):
				self._check_bounds(item)
				return item, IndexType.integer_scalar

			else:
				raise TypeError('Scalar index must be integer')

		else:

			# Convert to array if not already
			if not isinstance(item, np.ndarray):
				item = np.asanyarray(item)

			if item.dtype.kind in 'ui':
				# Integer array

				# Check in range
				check_index_in_range(item, self.size)

				return item, IndexType.integer_array

			elif item.dtype.kind == 'b':
				# Boolean array

				# TODO...
				# return item, IndexType.boolean_array
				raise NotImplementedError('Boolean indexing not yet supported')

			else:
				raise ValueError('Array indices must have integer or boolean dtype')

	def __getitem__(self, indices):

		# First check the case of getting a single item - this is most likely
		# to be used in a tight loop and so will be the place where all the
		# tests and conversions in this function will slow things down the most
		if (isinstance(indices, tuple) and len(indices) == 2 and
		    isinstance(indices[0], INTEGER_TYPES) and
		    isinstance(indices[1], INTEGER_TYPES)):

			return self.get_item(*indices)

		# Convert non-tuples to 1-tuples
		if not isinstance(indices, tuple):
			indices = (indices,)

		if len(indices) == 0:
			# Empty tuple: matrix[()] -> full array
			return self.to_array()

		elif len(indices) == 1:
			rowidx, rowtype = self._process_index_elem(indices[0])
			colidx = None
			coltype = IndexType.full_slice

		elif len(indices) == 2:
			rowidx, rowtype = self._process_index_elem(indices[0])
			colidx, coltype = self._process_index_elem(indices[1])

		else:
			raise ValueError('Too many indices for matrix')

		if rowtype is IndexType.full_slice:

			if coltype is IndexType.full_slice:
				# Full array
				return self.to_array()

			elif coltype is IndexType.integer_scalar:
				# Single row
				return self.get_row(colidx)

			else:
				# Integer array, get many rows
				return self._getitem_advanced_int_cols(colidx)

		elif rowtype is IndexType.integer_scalar:

			if coltype is IndexType.full_slice:
				# Single row
				return self.get_row(rowidx)

			# elif coltype is IndexType.integer_scalar:
			# 	# Single item
			# 	return self.get_item(rowidx, colidx)

			else:
				# Integer array
				# Single row, advanced integer indexing on columns
				return self._getitem_advanced_int_2d(rowidx, colidx)

		else:
			# Integer array

			if coltype is IndexType.full_slice:
				return self._getitem_advanced_int_rows(rowidx)

			elif coltype is IndexType.integer_scalar:
				# Advanced integer indexing on rows, single column
				return self._getitem_advanced_int_2d(rowidx, colidx)

			else:
				# Two integer arrays
				return self._getitem_advanced_int_2d(rowidx, colidx)

	def _getitem_advanced_int_rows(self, rows):
		"""Numpy advanced integer indexing on rows, full columns."""

		out = np.empty(rows.shape + (self.size,), dtype=self.array.dtype)

		for array_index, row in np.ndenumerate(rows):
			self.nbmatrix.get_row(row, out[array_index])

		return out

	def _getitem_advanced_int_cols(self, cols):
		"""Numpy advanced integer indexing on columns, full rows."""

		out = np.empty((self.size,) + cols.shape, dtype=self.array.dtype)

		row_out = np.empty(self.size, dtype=self.array.dtype)

		for array_index, row in np.ndenumerate(cols):
			# Can't do this, doesn't seem like it works with Numba
			#self.nbmatrix.get_row(row, out[:, array_index])
			self.nbmatrix.get_row(row, row_out)
			out[(slice(None),) + array_index] = row_out

		return out

	def _getitem_advanced_int_2d(self, rows, cols):
		"""Numpy advanced integer indexing on both rows and columns."""

		if self.upper:
			return getitem_advanced_int_upper(
				self.array,
				self.size,
				self.diag_val,
				rows,
				cols,
			)

		else:
			return getitem_advanced_int_lower(
				self.array,
				self.diag_val,
				rows,
				cols,
			)
