# triarray

Python package for working with symmetric matrices in non-redundant format.


## Overview

The `triarray` package contains tools for working with large symmetric matrices while only storing the elements in the upper or lower triangle, thus halving memory requirements.

When storing symmetric matrices stored in standard array format about half of the elements are redundant, meaning you are using twice as much memory or disk space as you need to. This is especially common in scientific applications when working with large distance or similarity matrices.

Space can be saved by storing only the lower or upper triangle of the array, but standard operations like getting an element by row and column become awkward. ``triarray`` provides tools for working with data in this format.

``triarray`` uses [Numba](http://numba.pydata.org/)'s just-in-time compilation to generate high-performance C code that works with any data type and is easily extendable (including within a Jupyter notebook).


### Example

The `scipy.spatial.distance.pdist` function calculates pairwise distances between all rows of a matrix and returns only the upper triangle of the full distance matrix:

```python

import numpy as np
from scipy.spatial.distance import pdist

vectors = np.random.rand(1000, 10)

dists = pdist(vectors)  # Shape is (499500,) instead of (1000, 1000)

```

The `triarray.TriMatrix` class wraps a 1D Numpy array storing the condensed data and exposes an interface that lets you treat it as if it was still in matrix format:

```python

from triarray import TriMatrix

matrix = TriMatrix(dists, upper=True, diag_val=0)

matrix.size  # Number of rows/columns in matrix
>>> 1000

matrix[0, 1]  # Distance between 0th and 1st vector
>>> 1.1610289956390953

matrix[0, 0]  # Diagonals are zero
>>> 0.0

matrix[0]  # 0th row of matrix
>>> array([ 0.        ,  1.161029  ,  1.03467554,  1.32559121,  1.26185034,
    ...

```

It even supports Numpy's advanced indexing with integer arrays of arbitrary shape:

```python

rows, cols = np.ix_([0, 1, 2], [3, 4, 5])
rows, cols
>>> (array([[0],
            [1],
            [3]]), array([[4, 5, 6]]))
            
matrix[rows, cols]
>>> array([[ 1.26185034,  1.08800206,  1.30490993],
           [ 0.99262394,  1.33044029,  1.20373382],
           [ 1.42524039,  1.36195143,  1.70404005]])

```


## Requirements

* Numpy 1.11 or above
* Numba 0.30 or above


## Installation

The easiest way is to use pip:

    pip install triarray
    
or you can clone the repository and run the setup script:

    cd path/to/triarray
    python setup.py install


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
