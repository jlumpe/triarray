"""Utilities for dealing with pairwise distance/similarity matrices."""

from .math import tri_n, tri_root, tri_root_strict, tri_root_trunc, tri_root_rem
from .convert import (
	mat_idx_to_triu,
	mat_idx_to_tril,
	triu_idx_to_mat,
	tril_idx_to_mat,
	tri_to_matrix,
	matrix_to_tri,
)
from .io import write_tri_file, read_tri_file_rows, read_tri_file
