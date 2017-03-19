.. py:currentmodule:: triarray



Python API
==========



Indexing
--------

Tools for converting between 2D indices of full matrices and 1D indices of
condensed arrays.

.. autofunction:: mat_idx_to_tril

.. autofunction:: mat_idx_to_triu

.. autofunction:: tril_idx_to_mat

.. autofunction:: triu_idx_to_mat



Array conversion
----------------

Convert full 2D matrices to and from condensed triangular format.

.. autofunction:: matrix_to_tri

.. autofunction:: tri_to_matrix



I/O
---

Read and write full matrices to and from disk in condensed format.

.. autofunction:: write_tri_file

.. autofunction:: read_tri_file_rows

.. autofunction:: read_tri_file



Matrix interface
----------------

.. autoclass:: TriMatrix
