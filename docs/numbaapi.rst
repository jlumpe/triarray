.. py:currentmodule:: triarray



Numba API
=========


Use these functions if you wish to extend **triarray** using Numba.


This package makes heavy use of `Numba <http://numba.pydata.org/>`_ to compile
Python functions into high-performance C code. Many of these compiled functions
are hidden behind Python functions to expose a more friendly API, but I chose
Numba over Cython for this purpose in part because it is much easier to extend
existing code (especially in the Jupyter Notebook).

If you plan on using this package in your own Numba code it will greatly
improve performance to use the Numba compiled functions directly.



TODO
----

Fill this out...
