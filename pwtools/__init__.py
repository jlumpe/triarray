"""Utilities for dealing with pairwise distance/similarity matrices."""

from .math import tri_n, tri_root, tri_root_rem
from .convert import (
	mat_idx_to_triu,
	mat_idx_to_tril,
	triu_idx_to_mat,
	tril_idx_to_mat,
	tri_to_matrix,
	matrix_to_tri,
)
