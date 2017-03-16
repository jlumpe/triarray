"""Test pwtools.math."""

import pwtools as pw


def test_tri():
	"""Test tri_n and tri_root."""

	# Just try a large range of values
	for n in range(0, 1000, 10):

		t_n = pw.tri_n(n)

		# Check by summing a range
		assert t_n == sum(range(1, n + 1))

		# Check root finding
		assert pw.tri_root(t_n) == n

		# Check root finding with remainder
		for i in range(n):
			assert pw.tri_root_rem(t_n + i) == (n, i)
