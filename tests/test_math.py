"""Test pwtools.math."""

import pytest
import numpy as np

import pwtools as pw


def test_tri_scalar():
	"""Test tri_n and inverses with scalar arguments."""

	# Just try a large range of values
	for n in range(0, 1000, 10):

		t_n = pw.tri_n(n)

		# Check by summing a range
		assert t_n == sum(range(1, n + 1))

		# Check root finding
		assert pw.tri_root(t_n) == n
		assert pw.tri_root_strict(t_n) == n

		# Check root finding up to next triangular number
		for i in range(1, n):

			# Strict should raise an exception
			with pytest.raises(ValueError):
				pw.tri_root_strict(t_n + i)

			# Truncated should return last triangular number
			assert pw.tri_root_trunc(t_n + i) == n

			# Try with remainder
			assert pw.tri_root_rem(t_n + i) == (n, i)


def test_tri_array():
	"""Test tri_n and inverses with array arguments."""

	n = np.arange(1000)

	t_n = pw.tri_n(n)

	# Check elements match scalar versions
	for n_i, tn_i in zip(n, t_n):
		assert pw.tri_n(n_i) == tn_i

	# Check inverse
	assert np.array_equal(pw.tri_root(t_n), n)
	assert np.array_equal(pw.tri_root_strict(t_n), n)
	assert np.array_equal(pw.tri_root_trunc(t_n), n)


def test_tri_root_improper():
	"""
	Test tri_root-related functions with arguments that are not triangular
	numbers.
	"""

	t = np.arange(1000)

	with pytest.raises(ValueError):
		pw.tri_root_strict(t)

	# Check the truncated version works as expected
	n = pw.tri_root_trunc(t)
	t_trunc = pw.tri_n(n)
	assert np.all((t_trunc <= t) & (t < pw.tri_n(n + 1)))

	# Check remainder function
	root, rem = pw.tri_root_rem(t)
	assert np.array_equal(root, n)
	assert np.array_equal(pw.tri_n(root) + rem, t)
