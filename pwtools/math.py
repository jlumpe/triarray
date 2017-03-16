"""Math stuff."""

import numpy as np
import numba as nb


@nb.vectorize(nopython=True)
def tri_n(n):
	"""Get the nth triangular number (numpy ufunc).

	:param int n: Nonnegative integer.
	:rtype: int
	"""
	return n * (n + 1) // 2


@nb.vectorize(nopython=True)
def tri_root(t):
	"""Get n such that t is the nth triangular number (numpy ufunc).

	:param int t: Triangular number.
	:rtype: int
	:raises ValueError: If t is not a triangular number.
	"""
	s = 8 * t + 1
	rs = int(round(np.sqrt(s)))
	if rs ** 2 != s:
		raise ValueError('Not a triangular number')
	return (rs - 1) // 2


@nb.jit(nopython=True)
def tri_root_rem(t):
	"""Get n and r such that ``t == tri_n(n) + r``.

	:param int t: Nonnegative integer
	:returns: (n, r) tuple.
	:rtype tuple:
	"""
	s = 8 * t + 1
	rs = int(round(np.sqrt(s)))
	if rs ** 2 > s:
		rs -= 1
	n = (rs - 1) // 2
	return n, t - tri_n(n)
