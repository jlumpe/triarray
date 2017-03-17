"""Math stuff."""

import numpy as np
import numba as nb


@nb.vectorize([nb.intp(nb.intp)], nopython=True)
def tri_n(n):
	"""tri_n(n)

	Numpy ufunc. Get the nth triangular number.

	:param int n: Nonnegative integer.
	:rtype: int
	"""
	return n * (n + 1) // 2


@nb.vectorize([nb.intp(nb.intp)], nopython=True)
def tri_root(t):
	"""tri_root(t)

	Numpy ufunc. Get n such that t is the nth triangular number.

	This is the fastest version of this function. Behavior is undefined when
	t is not a triangular number.

	:param int t: Triangular number.
	:rtype: int
	"""
	s = 8 * t + 1
	rs = nb.intp(np.sqrt(s) + .5)
	return (rs - 1) // 2


@nb.vectorize([nb.intp(nb.intp)], nopython=True)
def tri_root_strict(t):
	"""tri_root_stric(t)

	Numpy ufunc. Get n such that t is the nth triangular number, or raise an
	exception if t is not triangular.

	:param int t: Triangular number.
	:rtype: int
	:raises ValueError: If t is not a triangular number.
	"""
	s = 8 * t + 1
	rs = nb.intp(np.sqrt(s) + .5)
	if rs ** 2 != s:
		raise ValueError('Not a triangular number')
	return (rs - 1) // 2


@nb.vectorize([nb.intp(nb.intp)], nopython=True)
def tri_root_trunc(t):
	"""tri_root_trunc(t)

	Numpy ufunc. Get n such that t is >= the nth triangular number and < the
	(n+1)th triangular number.

	:param int t: Triangular number.
	:rtype: int
	:raises ValueError: If t is not a triangular number.
	"""
	s = 8 * t + 1
	rs = nb.intp(np.sqrt(s) + .5)
	if rs ** 2 > s:
		rs -= 1
	return (rs - 1) // 2


@nb.jit(nopython=True)
def tri_root_rem(t):
	"""Get n and r such that ``t == tri_n(n) + r``.

	:param t: Scalar or array of nonnegative integers.
	:returns: (n, r) tuple of arrays the same shape as ``t``.
	:rtype: tuple
	"""
	n = tri_root_trunc(t)
	return n, t - tri_n(n)
